from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.sac_qsafe.pytorch.sac_qsafe import SAC_QSafe
from rl_x.algorithms.sac_qsafe.pytorch.default_config import get_config
from rl_x.algorithms.sac.pytorch.general_properties import GeneralProperties


SAC_QSAFE_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(SAC_QSAFE_PYTORCH, get_config, SAC_QSafe, GeneralProperties)
