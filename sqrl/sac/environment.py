class SQRLPretrainEnvPair:
    """Create independent task and safety instances with ``env_factory(role)``."""

    def __init__(self, env_factory):
        self.task_env = env_factory("task")
        self.safety_env = env_factory("safety")
        if self.task_env is self.safety_env:
            raise ValueError("env_factory must return independent task and safety environment instances")
        if type(self.task_env) is not type(self.safety_env):
            raise TypeError("task and safety environments must be instances of the same class")
