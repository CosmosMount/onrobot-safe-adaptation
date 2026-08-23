from rl_x.environments.environment_manager import (
    extract_environment_name_from_file,
    register_environment,
)

from .create_env import create_train_and_eval_env
from .default_config import get_config
from .general_properties import GeneralProperties


GO2_SQRL_SDK2_MUJOCO = extract_environment_name_from_file(__file__)
register_environment(
    GO2_SQRL_SDK2_MUJOCO,
    get_config,
    create_train_and_eval_env,
    GeneralProperties,
)

