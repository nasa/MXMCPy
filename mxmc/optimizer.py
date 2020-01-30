from .optimizers.mfmc import MFMC
from .optimizers.mlmc import MLMC
from .optimizers.approximate_control_variates.generalized_multifidelity.acvmf \
    import ACVMF
from .optimizers.approximate_control_variates.generalized_multifidelity.acvmfu\
    import ACVMFU
from .optimizers.approximate_control_variates.generalized_multifidelity.acvmfmc\
    import ACVMFMC
from .optimizers.approximate_control_variates.generalized_multifidelity.gmfmr \
    import GMFMR
from .optimizers.approximate_control_variates.generalized_multifidelity.gmfsr \
    import GMFSR
from .optimizers.approximate_control_variates.generalized_multifidelity.acvkl \
    import ACVKL
from mxmc.optimizers.approximate_control_variates.generalized_independent_samples.acvis \
    import ACVIS
from mxmc.optimizers.approximate_control_variates.generalized_independent_samples.gissr \
    import GISSR
from mxmc.optimizers.approximate_control_variates.generalized_independent_samples.gismr \
    import GISMR

from .optimizers.model_selection import AutoModelSelection

ALGORITHM_MAP = {"mfmc": MFMC, "mlmc": MLMC, "acvmfu": ACVMFU, "acvmf": ACVMF,
                 "acvmfmc": ACVMFMC, "acvkl": ACVKL,
                 "gmfsr": GMFSR, "gmfmr": GMFMR,
                 "acvis": ACVIS, "gissr": GISSR, "gismr": GISMR}


class Optimizer:

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def optimize(self, algorithm, target_cost, auto_model_selection=False):
        optimizer = ALGORITHM_MAP[algorithm](*self._args, **self._kwargs)
        if auto_model_selection:
            optimizer = AutoModelSelection(optimizer)
        return optimizer.optimize(target_cost=target_cost)
