"""Factory kept free of SDK imports until the environment is actually reset."""

from .env import Go2SDKMujocoEnv
from .reset_controller import MujocoResetController
from .sdk_client import SDKClient


def create_train_and_eval_env(config):
    client = SDKClient(config.environment.domain_id, config.environment.interface)
    reset_controller = MujocoResetController(
        window_title=config.environment.mujoco_window_title
    )
    train_env = Go2SDKMujocoEnv(
        config, client=client, role="train", reset_controller=reset_controller
    )
    eval_env = Go2SDKMujocoEnv(
        config, client=client, role="eval", reset_controller=reset_controller
    )
    return train_env, eval_env
