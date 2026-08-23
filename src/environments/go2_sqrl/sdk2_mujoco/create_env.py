"""Factory kept free of SDK imports until the environment is actually reset."""

from .env import Go2SDKMujocoEnv
from .sdk_client import SDKClient


def create_train_and_eval_env(config):
    client = SDKClient(config.environment.domain_id, config.environment.interface)
    train_env = Go2SDKMujocoEnv(config, client=client, role="train")
    eval_env = Go2SDKMujocoEnv(config, client=client, role="eval")
    return train_env, eval_env

