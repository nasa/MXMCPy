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
    '''
    User interface for accessing all MXMC variance minimization optimizers for
    computing optimal sample allocations.
    '''
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def optimize(self, algorithm, target_cost, auto_model_selection=False):
        '''
        Performs variance minimization optimization to determine the optimal
        sample allocation across available models within a specified target
        computational cost.

        :param algorithm: name of method to use for optimization (e.g., "mlmc",
            "mfmc", "acvkl").
        :type algorithm: string
        :param target_cost: total target cost constraint so that optimizer finds
            sample allocation that requires less than or equal computation time.
        :type target_cost: float
        :param auto_model_selection: flag to use automatic model selection in
            optimization to test all subsets of models for best set.
        :type auto_model_selection: Boolean

        :Returns: An OptimizationResult namedtuple with entries for cost,
            variance, and sample_array. cost (float) is expected cost of all 
            model evaluations prescribed in sample_array (np.array). variance
            is the minimized variance from the optimization.
        '''
        optimizer = ALGORITHM_MAP[algorithm](*self._args, **self._kwargs)
        if auto_model_selection:
            optimizer = AutoModelSelection(optimizer)
        return optimizer.optimize(target_cost=target_cost)
