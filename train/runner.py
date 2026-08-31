import argparse
import logging
import os
import subprocess
from pathlib import Path
from xml.etree import ElementTree

from train.config import DEFAULT_MUJOCO_SCENE, PROJECT_ROOT


SUPPORTED_ALGORITHMS = frozenset({"sqrl_sac"})
SUPPORTED_ENVIRONMENTS = frozenset({"go2"})


def _validate_experiment_config(raw_config, command):
    """Validate selectors that used to be silently ignored by the runner."""

    experiment = raw_config.get("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("configuration experiment section must be a mapping")
    unknown = set(experiment) - {"algorithm", "environment"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(
            f"unsupported experiment keys: {names}; the CLI command selects the phase"
        )

    algorithm_name = str(experiment.get("algorithm", "sqrl_sac")).strip().lower()
    if algorithm_name not in SUPPORTED_ALGORITHMS:
        supported = ", ".join(sorted(SUPPORTED_ALGORITHMS))
        raise ValueError(
            f"unsupported experiment.algorithm={algorithm_name!r}; supported: {supported}"
        )
    environment_name = str(experiment.get("environment", "go2")).strip().lower()
    if environment_name not in SUPPORTED_ENVIRONMENTS:
        supported = ", ".join(sorted(SUPPORTED_ENVIRONMENTS))
        raise ValueError(
            f"unsupported experiment.environment={environment_name!r}; supported: {supported}"
        )
    return algorithm_name, "test" if command in ("zero-shot", "eval") else "train"


def main(argv=None):
    parser = argparse.ArgumentParser(description="SQRL reproduction with local Go2 environments")
    parser.add_argument("command", choices=("sim", "pretrain", "zero-shot", "finetune", "eval", "show-config"))
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--unitree-root", default=os.environ.get("UNITREE_MUJOCO_ROOT"))
    parser.add_argument("--scene", default=str(DEFAULT_MUJOCO_SCENE))
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s - %(message)s")
    robot_root = PROJECT_ROOT / "assets" / "robots" / "go2"
    robot_xml = robot_root / "mjcf" / "go2.xml"
    required_assets = [
        robot_root / "mjcf" / "scene.xml",
        robot_xml,
        robot_root / "usd" / "go2.usd",
        robot_root / "usd" / "Props" / "instanceable_meshes.usd",
    ]
    if robot_xml.is_file():
        robot_tree = ElementTree.parse(robot_xml)
        mesh_dir = robot_xml.parent / robot_tree.getroot().find("compiler").get("meshdir")
        required_assets.extend(mesh_dir / mesh.get("file") for mesh in robot_tree.findall(".//mesh"))
    missing_assets = [str(path.relative_to(PROJECT_ROOT)) for path in required_assets if not path.is_file()]
    if missing_assets:
        raise FileNotFoundError("missing local Go2 assets: " + ", ".join(missing_assets))

    if args.command == "sim":
        candidates = [Path(args.unitree_root).expanduser() if args.unitree_root else None,
                      PROJECT_ROOT.parent / "modules" / "unitree_mujoco",
                      PROJECT_ROOT.parent.parent / "modules" / "unitree_mujoco"]
        unitree_root = next((path.resolve() for path in candidates if path is not None and (path / "simulate").is_dir()), None)
        if unitree_root is None:
            raise FileNotFoundError("unitree_mujoco was not found; pass --unitree-root or set UNITREE_MUJOCO_ROOT")
        simulate_root = unitree_root / "simulate"
        executable = simulate_root / "build" / "unitree_mujoco"
        if not executable.is_file():
            raise FileNotFoundError(f"simulator executable not found: {executable}")
        scene = Path(args.scene).expanduser().resolve()
        if not scene.is_file():
            raise FileNotFoundError(f"MuJoCo scene not found: {scene}")
        canonical_scene = DEFAULT_MUJOCO_SCENE.resolve()
        if scene != canonical_scene:
            raise ValueError(
                "Strict SQRL transfer only supports the validated canonical "
                f"MuJoCo scene {canonical_scene}; got {scene}"
            )
        from train.mujoco.pytorch.sdk import validate_go2_mjcf_contract

        validate_go2_mjcf_contract(scene)
        environment = os.environ.copy()
        environment["ORSA_STRICT_LOCKSTEP"] = "1"
        library_root = simulate_root / "mujoco" / "lib"
        environment["LD_LIBRARY_PATH"] = str(library_root) + (
            f":{environment['LD_LIBRARY_PATH']}" if environment.get("LD_LIBRARY_PATH") else ""
        )
        command = [str(executable), "-r", "go2", "-s", str(scene),
                   "-i", str(args.domain_id), "-n", args.interface]
        logging.getLogger("sqrl_runner").info("starting simulator: %s", " ".join(command))
        return subprocess.run(command, cwd=simulate_root, env=environment, check=False).returncode

    import yaml
    from ml_collections import config_dict

    source_phase = args.command in ("pretrain", "show-config")
    if source_phase:
        from train.isaac.pytorch.config import get_config as get_environment_config

        environment = get_environment_config("go2_sqrl.isaac_lab").to_dict()
        environment.update({"terrain_mode": "flat", "domain_randomization": True})
    else:
        from train.mujoco.pytorch.environment import get_config as get_environment_config

        environment = get_environment_config("go2_sqrl.sdk2_mujoco").to_dict()
        environment.update({"domain_id": args.domain_id, "interface": args.interface})
        if args.command in ("finetune", "eval"):
            environment["target_velocity_x"] = 0.6

    raw_config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file) or {}
        if not isinstance(raw_config, dict):
            raise ValueError("configuration root must be a mapping")
    unknown_sections = set(raw_config) - {"experiment", "algorithm", "environment", "runner"}
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(f"unsupported configuration sections: {names}")
    algorithm_name, runner_mode = _validate_experiment_config(raw_config, args.command)

    algorithm = {
        "name": algorithm_name,
        "device": "gpu",
        "compile_mode": "reduce-overhead",
        "bf16_mixed_precision_training": False,
        "learning_rate": 3e-4,
        "policy_lr": 3e-4,
        "qtask_lr": 3e-4,
        "qsafe_lr": 3e-4,
        "entropy_lr": 3e-4,
        "dual_learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "task_buffer_size": 1_000_000,
        "safety_buffer_size": 100_000,
        "batch_size": 256,
        "safety_batch_size": 256,
        "learning_starts": 5_000,
        "n_pre": 500_000,
        "n_target": 500_000,
        "task_utd_ratio": 1.0,
        "gamma": 0.99,
        "safe_gamma": 0.7,
        "tau": 0.005,
        "safe_tau": 0.005,
        "epsilon_safe": 0.1,
        "max_safe_action_samples": 100,
        "max_safety_trajectories": int(environment.get("nr_safety_envs", 1)),
        "k": int(environment.get("nr_safety_envs", 1)),
        "n_off": 1,
        "target_entropy": "auto",
        "log_std_min": -20,
        "log_std_max": 2,
        "nr_hidden_units": 256,
        "initial_nu": 0.0,
        "logging_frequency": 50_000,
    }
    runner = {
        "output_dir": f"runs/sqrl/{args.command}",
        "evaluation_episodes": 5,
        "mode": runner_mode,
    }

    def merge(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(algorithm, raw_config.get("algorithm", {}))
    merge(environment, raw_config.get("environment", {}))
    merge(runner, raw_config.get("runner", {}))
    configured_mode = str(runner.get("mode", runner_mode)).strip().lower()
    if configured_mode != runner_mode:
        raise ValueError(
            f"runner.mode={configured_mode!r} conflicts with command {args.command!r}; "
            f"expected {runner_mode!r}"
        )
    runner["mode"] = runner_mode
    configured_algorithm = str(algorithm.get("name", algorithm_name)).strip().lower()
    if configured_algorithm != algorithm_name:
        raise ValueError(
            "algorithm.name and experiment.algorithm must select the same implementation"
        )
    algorithm["name"] = algorithm_name
    if "k" not in raw_config.get("algorithm", {}):
        algorithm["k"] = int(environment.get("nr_safety_envs", 1))
    if "max_safety_trajectories" not in raw_config.get("algorithm", {}):
        algorithm["max_safety_trajectories"] = int(algorithm["k"])
    config = config_dict.ConfigDict({"algorithm": algorithm, "environment": environment, "runner": runner})
    from train.core.base import validate_environment_contract

    validate_environment_contract(config.environment)
    if args.command == "show-config":
        print(config)
        return 0

    import torch

    if config.algorithm.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.algorithm.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.set_float32_matmul_precision("high")
    if config.algorithm.bf16_mixed_precision_training and device.type != "cuda":
        raise ValueError("bf16 requires a CUDA training device")

    app = None
    train_env = eval_env = None
    try:
        if args.command == "pretrain":
            from isaaclab.app import AppLauncher

            launcher = argparse.Namespace(
                disable_fabric=config.environment.disable_fabric,
                num_envs=config.environment.nr_envs,
                task=config.environment.name,
                headless=not bool(config.environment.render),
                livestream=config.environment.livestream,
                enable_cameras=config.environment.enable_cameras,
                xr=config.environment.xr,
                device="cuda:0" if config.environment.device == "gpu" else config.environment.device,
                cpu=config.environment.cpu,
                verbose=config.environment.verbose,
                info=config.environment.info,
                experience=config.environment.experience,
                rendering_mode=config.environment.rendering_mode,
                kit_args=config.environment.kit_args,
                anim_recording_enabled=config.environment.anim_recording_enabled,
                anim_recording_start_time=config.environment.anim_recording_start_time,
                anim_recording_stop_time=config.environment.anim_recording_stop_time,
            )
            app = AppLauncher(launcher).app
            from sqrl.sac.pytorch.workflow import SQRLWorkflow
            from train.isaac.pytorch.environment import Go2IsaacEnv

            train_env = Go2IsaacEnv(config)
            workflow = SQRLWorkflow(config, train_env, device)
            workflow.pretrain()
            checkpoint = Path(config.runner.output_dir) / "final.model"
            workflow.save(str(checkpoint), "pretrain")
            logging.getLogger("sqrl_runner").info("saved pretrain checkpoint: %s", checkpoint)
        else:
            if not args.checkpoint or not Path(args.checkpoint).is_file():
                raise FileNotFoundError("zero-shot, finetune and eval require --checkpoint <model>")
            from sqrl.sac.pytorch.workflow import SQRLWorkflow
            from train.mujoco.pytorch.environment import Go2MujocoEnv

            train_env, eval_env = Go2MujocoEnv.create_pair(config)
            env = eval_env if args.command in ("zero-shot", "eval") else train_env
            workflow = SQRLWorkflow(config, env, device)
            checkpoint_phase = workflow.load(args.checkpoint, transfer=args.command in ("zero-shot", "finetune"))
            expected_phase = "finetune" if args.command == "eval" else "pretrain"
            if checkpoint_phase != expected_phase:
                raise ValueError(f"{args.command} requires a {expected_phase} checkpoint, got {checkpoint_phase}")
            if args.command == "finetune":
                workflow.finetune()
                checkpoint = Path(config.runner.output_dir) / "final.model"
                workflow.save(str(checkpoint), "finetune")
                logging.getLogger("sqrl_runner").info("saved finetune checkpoint: %s", checkpoint)
            else:
                workflow.evaluate(config.runner.evaluation_episodes)
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None and eval_env is not train_env:
            eval_env.close()
        if app is not None:
            app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
