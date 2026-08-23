"""Isaac creation remains lazy so registration does not import Kit modules."""


def create_train_and_eval_env(config):
    from .env import Go2IsaacEnv

    train_env = Go2IsaacEnv(config)
    # Isaac Lab owns one SimulationContext. The second adapter intentionally
    # shares it; partition-aware rollout code treats the pools as one scene.
    eval_env = Go2IsaacEnv(config, backend=train_env.backend)
    return train_env, eval_env

