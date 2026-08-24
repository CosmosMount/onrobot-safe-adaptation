from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.droq_qsafe.pytorch.droq_qsafe import DroQ_QSafe
from rl_x.algorithms.droq_qsafe.pytorch.default_config import get_config
from rl_x.algorithms.droq_qsafe.pytorch.general_properties import GeneralProperties


DROQ_QSAFE_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(DROQ_QSAFE_PYTORCH, get_config, DroQ_QSafe, GeneralProperties)
