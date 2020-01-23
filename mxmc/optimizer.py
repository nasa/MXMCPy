from .acvis import ACVIS
from .acvmf import ACVMF
from .mfmc import MFMC
from .mlmc import MLMC
from .acvkl_enumerator import ACVKLEnumerator
from .model_selection import AutoModelSelection

ALGORITHM_MAP = {"mfmc": MFMC, "mlmc": MLMC, "acvmf": ACVMF, "acvis": ACVIS,
                 "acvkl": ACVKLEnumerator}


class Optimizer:

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def optimize(self, algorithm, target_cost, auto_model_selection=False):
        optimizer = ALGORITHM_MAP[algorithm](*self._args, **self._kwargs)
        if auto_model_selection:
            optimizer = AutoModelSelection(optimizer)
        return optimizer.optimize(target_cost=target_cost)
