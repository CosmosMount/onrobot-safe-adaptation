"""Backend-neutral Go2 vector-environment base class."""

import numpy as np
import torch

from .actions import project_actions_from_observation, project_action_targets_tensor
from .contracts import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    OBSERVATION_SPEC,
    validate_environment_contract,
)


# Common vector-environment interface implemented by both backends.
class Go2Environment:
    """Shared vector-environment contract for the Isaac and MuJoCo backends."""

    def __init__(self, config, nr_envs):
        from types import SimpleNamespace
        from algorithms.types import ActionSpaceType, ObservationSpaceType
        from gymnasium.spaces import Box

        self.config = config.environment
        validate_environment_contract(self.config)
        self.nr_envs = self.num_envs = int(nr_envs)
        self.single_observation_space = Box(low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.single_action_space = Box(low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.policy_observation_indices = np.arange(OBSERVATION_SIZE)
        self.critic_observation_indices = np.arange(OBSERVATION_SIZE)
        self.safety_critic_observation_indices = np.arange(OBSERVATION_SIZE)
        self.general_properties = SimpleNamespace(
            action_space_type=ActionSpaceType.CONTINUOUS,
            observation_space_type=ObservationSpaceType.FLAT_VALUES,
            observation_space_shape=self.single_observation_space.shape,
            policy_observation_indices=self.policy_observation_indices,
        )

    @staticmethod
    def project_actions(states, actions):
        if torch.is_tensor(actions):
            if not torch.is_tensor(states):
                states = torch.as_tensor(states, dtype=actions.dtype, device=actions.device)
            else:
                states = states.to(dtype=actions.dtype, device=actions.device)
            if states.shape[-1] != OBSERVATION_SIZE:
                raise ValueError(f"Observation must end in {OBSERVATION_SIZE} values, got {states.shape}")
            return project_action_targets_tensor(states[..., OBSERVATION_SPEC.previous_action_q_target], actions)[0]
        return project_actions_from_observation(states, actions)

    @staticmethod
    def get_final_observation_at_index(info, index):
        return info["final_observation"][index]

    @staticmethod
    def get_final_info_value_at_index(info, key, index):
        final_info = info["final_info"][index]
        if final_info is None:
            raise KeyError(f"No final info for environment {index}")
        return final_info[key]

    @staticmethod
    def get_logging_info_dict(info):
        ignored = {"failure", "applied_action", "final_observation", "final_info"}
        return {key: np.asarray(value).reshape(-1).tolist() for key, value in info.items()
                if key not in ignored and not isinstance(value, list)}

    def checkpoint_manifest(self, normalizer=None):
        from .estimation import velocity_estimator_config_from
        from .task import build_manifest

        return build_manifest(
            normalizer,
            fall_angle_threshold=float(self.config.fall_angle_threshold),
            fall_min_base_clearance=float(self.config.fall_min_base_clearance),
            fall_consecutive_frames=int(self.config.fall_consecutive_frames),
            target_velocity_x=float(self.config.target_velocity_x),
            domain_randomization=bool(self.config.domain_randomization),
            velocity_estimator_config=velocity_estimator_config_from(self.config),
        )

    def validate_checkpoint_manifest(self, manifest, normalizer=None):
        from .task import validate_manifest

        validate_manifest(manifest, self.checkpoint_manifest(normalizer))

    def validate_transfer_checkpoint_manifest(self, manifest, normalizer=None):
        from .task import validate_transfer_manifest

        validate_transfer_manifest(manifest, self.checkpoint_manifest(normalizer))

