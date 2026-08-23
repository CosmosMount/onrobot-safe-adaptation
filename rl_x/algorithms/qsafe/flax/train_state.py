from flax.training.train_state import TrainState
import flax.struct


class QSafeTrainState(TrainState):
    target_params: flax.core.FrozenDict
