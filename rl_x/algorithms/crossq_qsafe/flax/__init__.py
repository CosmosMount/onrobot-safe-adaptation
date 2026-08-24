from rl_x.algorithms.algorithm_manager import (
    extract_algorithm_name_from_file,
    register_algorithm,
)
from rl_x.algorithms.crossq_qsafe.flax.crossq_qsafe import CrossQ_QSafe
from rl_x.algorithms.crossq_qsafe.flax.default_config import get_config
from rl_x.algorithms.crossq_qsafe.flax.general_properties import GeneralProperties


CROSSQ_QSAFE_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(CROSSQ_QSAFE_FLAX, get_config, CrossQ_QSafe, GeneralProperties)
