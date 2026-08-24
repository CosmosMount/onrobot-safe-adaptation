from rl_x.algorithms.droq.flax.default_config import get_config as get_droq_config
from rl_x.algorithms.droq_qsafe.pytorch.default_config import add_qsafe_defaults


def get_config(algorithm_name):
    return add_qsafe_defaults(get_droq_config(algorithm_name), algorithm_name)


__all__ = ["get_config"]
