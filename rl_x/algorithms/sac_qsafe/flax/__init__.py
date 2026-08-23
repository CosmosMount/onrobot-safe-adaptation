from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.sac_qsafe.flax.sac_qsafe import SAC_QSafe
from rl_x.algorithms.sac_qsafe.flax.default_config import get_config
from rl_x.algorithms.sac.flax.general_properties import GeneralProperties


SAC_QSAFE_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(SAC_QSAFE_FLAX, get_config, SAC_QSafe, GeneralProperties)
