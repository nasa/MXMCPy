from .optimizers.mfmc import MFMC
from .optimizers.mlmc import MLMC

from .optimizers.approximate_control_variates.generalized_independent_samples.impl_optimizers import *
from .optimizers.approximate_control_variates.generalized_multifidelity.impl_optimizers import *
from .optimizers.approximate_control_variates.generalized_recursive_difference.impl_optimizers import *

from .optimizers.model_selection import AutoModelSelection

ALGORITHM_MAP = {"mfmc": MFMC, "mlmc": MLMC, "acvmfu": ACVMFU, "acvmf": ACVMF,
                 "acvmfmc": ACVMFMC, "acvkl": ACVKL,
                 "gmfsr": GMFSR, "gmfmr": GMFMR,
                 "acvis": ACVIS, "gissr": GISSR, "gismr": GISMR,
                 "wrdiff": WRDiff, "grdsr": GRDSR, "grdmr": GRDMR}


class Optimizer:

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def optimize(self, algorithm, target_cost, auto_model_selection=False):
        optimizer = ALGORITHM_MAP[algorithm](*self._args, **self._kwargs)
        if auto_model_selection:
            optimizer = AutoModelSelection(optimizer)
        return optimizer.optimize(target_cost=target_cost)
