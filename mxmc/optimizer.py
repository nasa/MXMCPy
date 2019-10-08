from .mfmc import MFMC
from .mlmc import MLMC
from .model_selection import AutoModelSelection

ALGORITHM_MAP = {"mfmc": MFMC, "mlmc": MLMC}


class Optimizer():

    def __init__(self, *args, **kwargs):  # TODO: what should we do here?
        self._args = args
        self._kwargs = kwargs

    def optimize(self, algorithm, target_cost, auto_model_selection=False):
        optimizer = ALGORITHM_MAP[algorithm](*self._args, **self._kwargs)
        if auto_model_selection:
            optimizer = AutoModelSelection(optimizer)
        return optimizer.optimize(target_cost=target_cost)
