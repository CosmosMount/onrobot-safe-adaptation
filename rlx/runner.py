import argparse
import importlib
import logging
import logging.handlers
import os
import subprocess
import sys
import time
from collections.abc import Mapping

import yaml
from absl import logging as absl_logging
from ml_collections import config_dict

from rlx.manager import (
    get_algorithm_config,
    get_algorithm_general_properties,
    get_algorithm_model_class,
    get_environment_config,
    get_environment_create_train_and_eval_env,
    get_environment_general_properties,
)
from rlx.types import DLFrameworkType, SimulationType


os.environ["WANDB__SERVICE_WAIT"] = "600"
absl_logging.set_verbosity(absl_logging.ERROR)

rlx_logger = logging.getLogger("rl_x")


class RunnerMode:
    TRAIN = "train"
    TEST = "test"
    SHOW_CONFIG = "show_config"


def load_yaml_config(config_path):
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ValueError("The YAML configuration root must be a mapping")
    return dict(config)


def get_config_section(config, section_name, required=False):
    section = config.get(section_name)
    if section is None:
        if required:
            raise ValueError(f"Missing required YAML section: {section_name}")
        return {}
    if not isinstance(section, Mapping):
        raise ValueError(f"YAML section '{section_name}' must be a mapping")
    return dict(section)


def parse_experiment_config(config):
    experiment = get_config_section(config, "experiment", required=True)
    missing_fields = [field for field in ("algorithm", "environment", "mode") if not experiment.get(field)]
    if missing_fields:
        raise ValueError(f"Missing required experiment field(s): {', '.join(missing_fields)}")

    valid_modes = (RunnerMode.TRAIN, RunnerMode.TEST, RunnerMode.SHOW_CONFIG)
    if experiment["mode"] not in valid_modes:
        raise ValueError(f"Invalid experiment mode '{experiment['mode']}'. Expected one of: {', '.join(valid_modes)}")
    return experiment


def config_to_dict(config):
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(f"Expected a mapping-like default config, got {type(config).__name__}")


def merge_config(default_config, overrides):
    merged = config_to_dict(default_config)

    def merge_values(target, source):
        for key, value in source.items():
            if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
                nested_target = dict(target[key])
                merge_values(nested_target, value)
                target[key] = nested_target
            else:
                target[key] = value

    merge_values(merged, overrides)
    return config_dict.ConfigDict(merged)


def get_explicit_config_params(config, prefix):
    params = []
    for key, value in config.items():
        param_name = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            params.extend(get_explicit_config_params(value, param_name))
        else:
            params.append(param_name)
    return params


def get_runner_default_config(runner_mode):
    return config_dict.ConfigDict({
        "mode": runner_mode,
        "track_console": False,
        "track_wandb": False,
        "wandb_entity": "placeholder",
        "project_name": "placeholder",
        "exp_name": "placeholder",
        "run_name": str(int(time.time())),
        "notes": "placeholder",
        "save_model": False,
        "load_model": "",
        "nr_test_episodes": 10,
        "jax_cache_dir": "/tmp/jax_cache",
        "jax_default_matmul_precision": "bfloat16",
        "jax_exec_time_optimization_effort": 0.0,
        "jax_memory_fitting_effort": 1.0,
    })


class Runner:
    def __init__(self, implementation_package_names=("rl_x",)):
        yaml_config = self.parse_arguments()
        experiment_config = parse_experiment_config(yaml_config)
        algorithm_name = experiment_config["algorithm"]
        environment_name = experiment_config["environment"]
        self._mode = experiment_config["mode"]

        runner_overrides = get_config_section(yaml_config, "runner")
        algorithm_overrides = get_config_section(yaml_config, "algorithm")
        environment_overrides = get_config_section(yaml_config, "environment")

        runner_config = merge_config(get_runner_default_config(self._mode), runner_overrides)
        runner_config.mode = self._mode

        self.import_environment(environment_name, implementation_package_names)
        environment_general_properties = get_environment_general_properties(environment_name)
        environment_config = merge_config(get_environment_config(environment_name), environment_overrides)
        self.environment_uses_isaac_lab = SimulationType.ISAAC_LAB == environment_general_properties.simulation_type
        if self.environment_uses_isaac_lab:
            self.initialize_isaac_lab(environment_config)

        self.import_algorithm(algorithm_name, implementation_package_names)
        algorithm_general_properties = get_algorithm_general_properties(algorithm_name)
        algorithm_config = merge_config(get_algorithm_config(algorithm_name), algorithm_overrides)

        self.check_compatibility(environment_general_properties, algorithm_general_properties)
        self.configure_dl_frameworks(
            runner_config,
            algorithm_config,
            environment_config,
            algorithm_general_properties,
            environment_general_properties,
        )

        self._model_class = get_algorithm_model_class(algorithm_name)
        self._create_train_and_eval_env = get_environment_create_train_and_eval_env(environment_name)
        self._explicit_algorithm_params = get_explicit_config_params(algorithm_overrides, "algorithm")

        self._config = config_dict.ConfigDict()
        self._config.experiment = config_dict.ConfigDict(experiment_config)
        self._config.runner = runner_config
        self._config.algorithm = algorithm_config
        self._config.environment = environment_config
        self.config = self._config

        self.configure_logging()


    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
        args, remaining_args = parser.parse_known_args()
        sys.argv = [sys.argv[0], *remaining_args]
        return load_yaml_config(args.config)


    def initialize_isaac_lab(self, environment_config):
        from isaaclab.app import AppLauncher

        launcher_args = argparse.Namespace()
        launcher_args.disable_fabric = environment_config.get("disable_fabric", None)
        launcher_args.num_envs = environment_config.get("nr_envs", None)
        launcher_args.task = environment_config.get("name", None)
        launcher_args.headless = not environment_config.get("render", False)
        launcher_args.livestream = environment_config.get("livestream", None)
        launcher_args.enable_cameras = environment_config.get("enable_cameras", None)
        launcher_args.xr = environment_config.get("xr", None)
        device = environment_config.get("device", None)
        launcher_args.device = "cuda:0" if device == "gpu" else device
        launcher_args.cpu = environment_config.get("cpu", None)
        launcher_args.verbose = environment_config.get("verbose", None)
        launcher_args.info = environment_config.get("info", None)

        for argument_name in ("experience", "rendering_mode", "kit_args"):
            argument_value = environment_config.get(argument_name, None)
            if argument_value is not None:
                setattr(launcher_args, argument_name, argument_value)

        launcher_args.anim_recording_enabled = environment_config.get("anim_recording_enabled", None)
        launcher_args.anim_recording_start_time = environment_config.get("anim_recording_start_time", None)
        launcher_args.anim_recording_stop_time = environment_config.get("anim_recording_stop_time", None)

        app_launcher = AppLauncher(launcher_args)
        self.isaac_simulation_app = app_launcher.app


    def check_compatibility(self, environment_properties, algorithm_properties):
        if environment_properties.action_space_type not in algorithm_properties.action_space_types:
            raise ValueError(f"Incompatible action space type. Environment: {environment_properties.action_space_type}, Algorithm: {algorithm_properties.action_space_types}")
        if environment_properties.observation_space_type not in algorithm_properties.observation_space_types:
            raise ValueError(f"Incompatible observation space type. Environment: {environment_properties.observation_space_type}, Algorithm: {algorithm_properties.observation_space_types}")
        if environment_properties.data_interface_type not in algorithm_properties.data_interface_types:
            raise ValueError(f"Incompatible data interface type. Environment: {environment_properties.data_interface_type}, Algorithm: {algorithm_properties.data_interface_types}")


    def configure_dl_frameworks(self, runner_config, algorithm_config, environment_config, algorithm_properties, environment_properties):
        algorithm_uses_torch = DLFrameworkType.TORCH == algorithm_properties.deep_learning_framework_type
        algorithm_uses_jax = DLFrameworkType.JAX == algorithm_properties.deep_learning_framework_type
        environment_uses_jax = SimulationType.JAX_BASED == environment_properties.simulation_type
        environment_uses_torch = environment_properties.simulation_type in (SimulationType.ISAAC_LAB, SimulationType.MANISKILL, SimulationType.WARP)

        if algorithm_uses_torch:
            import torch

            torch.set_float32_matmul_precision("high")
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, message=".*is deprecated, please use.*")

            if environment_uses_torch:
                self.check_device_compatibility(algorithm_config, environment_config)

        if algorithm_uses_jax or environment_uses_jax:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            import jax

            jax.config.update("jax_default_matmul_precision", runner_config.jax_default_matmul_precision)
            jax.config.update("jax_exec_time_optimization_effort", float(runner_config.jax_exec_time_optimization_effort))
            jax.config.update("jax_memory_fitting_effort", float(runner_config.jax_memory_fitting_effort))
            jax.config.update("jax_compilation_cache_dir", runner_config.jax_cache_dir)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

            algorithm_device = algorithm_config.get("device", None) if algorithm_uses_jax else None
            environment_device = environment_config.get("device", None) if environment_uses_jax else None
            self.check_device_compatibility(algorithm_config if algorithm_uses_jax else {}, environment_config if environment_uses_jax else {})
            if (algorithm_device or environment_device) == "cpu":
                jax.config.update("jax_platform_name", "cpu")
            try:
                jax.default_backend()
            except Exception:
                pass


    def check_device_compatibility(self, algorithm_config, environment_config):
        algorithm_device = algorithm_config.get("device", None)
        environment_device = environment_config.get("device", None)
        if algorithm_device and environment_device and algorithm_device != environment_device:
            raise ValueError("Incompatible device types between algorithm and environment")


    def configure_logging(self):
        rlx_logger.setLevel(logging.INFO)
        rlx_logger.propagate = False
        rlx_logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s", "%m-%d %H:%M:%S"))
        memory_handler = logging.handlers.MemoryHandler(100, logging.ERROR, console_handler)
        rlx_logger.addHandler(memory_handler)

        def info(message, flush=True, *args, **kwargs):
            if rlx_logger.isEnabledFor(logging.INFO):
                rlx_logger._log(logging.INFO, message, args, stacklevel=2, **kwargs)
            if flush:
                rlx_logger.handlers[0].flush()

        rlx_logger.info = info

        def handle_exception(exception_type, exception_value, exception_traceback):
            if issubclass(exception_type, KeyboardInterrupt):
                sys.__excepthook__(exception_type, exception_value, exception_traceback)
                return
            rlx_logger.error("Uncaught exception", exc_info=(exception_type, exception_value, exception_traceback))

        sys.excepthook = handle_exception


    def import_environment(self, environment_name, implementation_package_names):
        for package_name in implementation_package_names:
            module_name = f"{package_name}.environments.{environment_name}"
            try:
                return importlib.import_module(module_name)
            except ModuleNotFoundError as error:
                if not (error.name == module_name or module_name.startswith(f"{error.name}.")):
                    raise
        raise ModuleNotFoundError(f"Environment '{environment_name}' not found in any of the implementation packages: {implementation_package_names}")


    def import_algorithm(self, algorithm_name, implementation_package_names):
        for package_name in implementation_package_names:
            module_name = f"{package_name}.algorithms.{algorithm_name}"
            try:
                return importlib.import_module(module_name)
            except ModuleNotFoundError as error:
                if not (error.name == module_name or module_name.startswith(f"{error.name}.")):
                    raise
        raise ModuleNotFoundError(f"Algorithm '{algorithm_name}' not found in any of the implementation packages: {implementation_package_names}")


    def run(self):
        if self._mode == RunnerMode.SHOW_CONFIG:
            main_function = self.show_config
        elif self._mode == RunnerMode.TRAIN:
            main_function = self.train
        elif self._mode == RunnerMode.TEST:
            main_function = self.test
        else:
            raise ValueError("Invalid mode")

        try:
            main_function()
        except KeyboardInterrupt:
            rlx_logger.warning("KeyboardInterrupt")


    def show_config(self):
        rlx_logger.info("\n" + str(self._config))


    def get_run_path(self):
        return os.path.abspath(f"runs/{self._config.runner.project_name}/{self._config.runner.exp_name}/{self._config.runner.run_name}")


    def initialize_wandb(self, run_path):
        import wandb

        wandb.init(
            entity=self._config.runner.wandb_entity,
            project=self._config.runner.project_name,
            group=self._config.runner.exp_name,
            name=self._config.runner.run_name,
            notes=self._config.runner.notes,
            config=self._config.to_dict(),
            monitor_gym=True,
            save_code=True,
        )
        wandb.define_metric("*", step_metric="global_step")

        python_packages = subprocess.check_output(["pip", "freeze"]).decode().split("\n")
        python_packages = [package.split("==") for package in python_packages if package]
        wandb.config["python_packages"] = {package[0]: package[1] for package in python_packages if len(package) == 2}

        try:
            project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
            git_diff = subprocess.check_output(["git", "diff"], cwd=project_dir).decode()
            diff_path = os.path.join(run_path, "diff.patch")
            with open(diff_path, "w", encoding="utf-8") as diff_file:
                diff_file.write(git_diff)
            wandb.save(diff_path, base_path=run_path)
            wandb.config["git_commit_hash"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_dir).decode().strip()
        except Exception as error:
            rlx_logger.warning(f"Could not log git diff and commit hash: {error}")

        if "SLURM_JOB_ID" in os.environ:
            wandb.config["SLURM_JOB_ID"] = os.environ["SLURM_JOB_ID"]
        return wandb


    def train(self):
        run_path = self.get_run_path()
        if self._config.runner.save_model or self._config.runner.track_wandb:
            os.makedirs(run_path, exist_ok=True)

        wandb = self.initialize_wandb(run_path) if self._config.runner.track_wandb else None
        train_env, eval_env = self._create_train_and_eval_env(self._config)

        if self._config.runner.load_model:
            model = self._model_class.load(self._config, train_env, eval_env, run_path, None, self._explicit_algorithm_params)
        else:
            model = self._model_class(self._config, train_env, eval_env, run_path, None)

        try:
            model.train()
        except Exception:
            rlx_logger.error("Uncaught exception", exc_info=True)
        finally:
            train_env.close()
            eval_env.close()
            if wandb:
                wandb.finish()
            if self.environment_uses_isaac_lab:
                self.isaac_simulation_app.close()


    def test(self):
        if self._config.runner.track_wandb:
            raise ValueError("Wandb is not supported in test mode")
        if self._config.runner.save_model:
            raise ValueError("Saving model is not supported in test mode")

        run_path = self.get_run_path()
        train_env, eval_env = self._create_train_and_eval_env(self._config)
        if self._config.runner.load_model:
            model = self._model_class.load(self._config, train_env, eval_env, run_path, None, self._explicit_algorithm_params)
        else:
            model = self._model_class(self._config, train_env, eval_env, run_path, None)

        try:
            model.test(self._config.runner.nr_test_episodes)
        except Exception:
            rlx_logger.error("Uncaught exception", exc_info=True)
        finally:
            train_env.close()
            eval_env.close()
            if self.environment_uses_isaac_lab:
                self.isaac_simulation_app.close()
