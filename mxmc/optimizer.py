from mxmc.optimizers.mfmc import MFMC
from mxmc.optimizers.mlmc import MLMC

from mxmc.optimizers.approximate_control_variates.generalized_independent_samples.impl_optimizers import *  # noqa: E501, F403
from mxmc.optimizers.approximate_control_variates.generalized_multifidelity.impl_optimizers import *  # noqa: E501, F403
from mxmc.optimizers.approximate_control_variates.generalized_recursive_difference.impl_optimizers import *  # noqa: E501, F403

from mxmc.optimizers.model_selection import AutoModelSelection

ALGORITHM_MAP = {"mfmc": MFMC, "mlmc": MLMC, "acvmfu": ACVMFU,     # noqa: F405
                 "acvmf": ACVMF, "acvmfmc": ACVMFMC,               # noqa: F405
                 "acvkl": ACVKL, "gmfsr": GMFSR, "gmfmr": GMFMR,   # noqa: F405
                 "acvis": ACVIS, "gissr": GISSR, "gismr": GISMR,   # noqa: F405
                 "wrdiff": WRDiff, "grdsr": GRDSR,                 # noqa: F405
                 "grdmr": GRDMR}                                   # noqa: F405


class Optimizer:
    '''
    User interface for accessing all MXMC variance minimization optimizers for
    computing optimal sample allocations.

    :param model_costs: cost (run time) of all models
    :type model_costs: list of floats
    :param covariance: Covariance matrix defining covariance (of outputs) among         all available models. Size MxM where M is # models.
    :type covariance: 2D np.array

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
        :param target_cost: total target cost constraint so that optimizer
            finds sample allocation that requires less than or equal
            computation time.
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
