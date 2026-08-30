from enum import Enum

class DLFrameworkType(Enum):
    TORCH = 0
    JAX = 1

class ActionSpaceType(Enum):
    CONTINUOUS = 0
    DISCRETE = 1

class DataInterfaceType(Enum):
    LIST = 0
    NUMPY = 1
    TORCH = 2
    JAX = 3

class ObservationSpaceType(Enum):
    FLAT_VALUES = 0
    IMAGES = 1

class SimulationType(Enum):
    DEFAULT = 0
    JAX_BASED = 1
    ISAAC_LAB = 2
    MANISKILL = 3
    WARP = 4
