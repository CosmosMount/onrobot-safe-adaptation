"""Strictly isolated Isaac environments for SQRL pre-training."""

from __future__ import annotations

import argparse

from ml_collections import config_dict

from train.core.environment import Go2Environment
from train.core.process_environment import EnvironmentProcess


def _worker_config(config_data, role, nr_envs):
    config = config_dict.ConfigDict(config_data)
    config.environment.nr_envs = int(nr_envs)
    if role == "safety":
        config.environment.seed = int(config.environment.seed) + 1
    return config


def _launcher_args(config):
    environment = config.environment
    return argparse.Namespace(
        disable_fabric=environment.disable_fabric,
        num_envs=environment.nr_envs,
        task=environment.name,
        headless=not bool(environment.render),
        livestream=environment.livestream,
        enable_cameras=environment.enable_cameras,
        xr=environment.xr,
        device="cuda:0" if environment.device == "gpu" else environment.device,
        cpu=environment.cpu,
        verbose=environment.verbose,
        info=environment.info,
        experience=environment.experience,
        rendering_mode=environment.rendering_mode,
        kit_args=environment.kit_args,
        anim_recording_enabled=environment.anim_recording_enabled,
        anim_recording_start_time=environment.anim_recording_start_time,
        anim_recording_stop_time=environment.anim_recording_stop_time,
    )


class _IsaacRuntime:
    """Own one child-local AppLauncher and vector environment."""

    def __init__(self, app, environment):
        self.app = app
        self.environment = environment

    def reset(self):
        return self.environment.reset()

    def step(self, actions):
        return self.environment.step(actions)

    def close(self):
        try:
            self.environment.close()
        finally:
            self.app.close()


def _create_isaac_runtime(config_data, role, nr_envs):
    """Launch Isaac before importing modules that require its runtime."""

    config = config_dict.ConfigDict(config_data)
    if int(config.environment.nr_envs) != int(nr_envs):
        raise ValueError(f"{role} worker environment width changed before startup")
    from isaaclab.app import AppLauncher

    app = AppLauncher(_launcher_args(config)).app
    try:
        from train.isaac.pytorch.environment import Go2IsaacEnv

        environment = Go2IsaacEnv(config)
    except BaseException:
        app.close()
        raise
    return _IsaacRuntime(app, environment)


class IsaacProcessEnvironment(Go2Environment):
    """Standard vector-env endpoint backed by one isolated Isaac process."""

    def __init__(
        self,
        config,
        role,
        nr_envs,
        *,
        environment_factory=_create_isaac_runtime,
    ):
        self.role = str(role)
        if self.role not in {"task", "safety"}:
            raise ValueError("Isaac worker role must be 'task' or 'safety'")
        config_data = (
            config.to_dict() if hasattr(config, "to_dict") else dict(config)
        )
        worker_config = _worker_config(config_data, self.role, nr_envs)
        super().__init__(worker_config, nr_envs)
        self._process = EnvironmentProcess(
            environment_factory,
            (worker_config.to_dict(), self.role, self.nr_envs),
            name=f"isaac-{self.role}-environment",
        )

    @property
    def pid(self):
        return self._process.pid

    def start(self):
        self._process.start()

    def wait_until_ready(self):
        self._process.wait_until_ready()

    def reset(self, *, seed=None, options=None):
        if seed is not None or options is not None:
            raise ValueError(
                "Remote Isaac reset does not accept seed/options overrides"
            )
        return self._process.reset()

    def step(self, actions):
        return self._process.step(actions)

    def close(self):
        self._process.close()


class IsaacPretrainEnvironments:
    """Own independent task and safety environments and their lifetimes."""

    def __init__(self, config, *, environment_factory=_create_isaac_runtime):
        environment = config.environment
        nr_task_envs = int(environment.nr_task_envs)
        nr_safety_envs = int(environment.nr_safety_envs)
        if nr_task_envs < 1 or nr_safety_envs < 1:
            raise ValueError("Task and safety worker sizes must both be positive")
        if nr_task_envs + nr_safety_envs != int(environment.nr_envs):
            raise ValueError("Task and safety worker sizes must sum to nr_envs")

        self.task = None
        self.safety = None
        try:
            self.task = IsaacProcessEnvironment(
                config,
                "task",
                nr_task_envs,
                environment_factory=environment_factory,
            )
            self.safety = IsaacProcessEnvironment(
                config,
                "safety",
                nr_safety_envs,
                environment_factory=environment_factory,
            )
            self.task.start()
            self.task.wait_until_ready()
            self.safety.start()
            self.safety.wait_until_ready()
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            if self.task is not None:
                self.task.close()
        finally:
            if self.safety is not None:
                self.safety.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
