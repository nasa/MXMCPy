from .grd_optimizer import GRDOptimizer
from ..recursion_enumerator import SREnumerator
from ..recursion_enumerator import MREnumerator


__all__ = ['GRDMR', 'GRDSR', 'WRDiff']


class GRDMR(MREnumerator):

    def _get_sub_optimizer(self, *args, **kwargs):
        return GRDOptimizer(*args, **kwargs)


class GRDSR(SREnumerator):

    def _get_sub_optimizer(self, *args, **kwargs):
        return GRDOptimizer(*args, **kwargs)


class WRDiff(GRDOptimizer):

    def __init__(self, model_cost, covariance, *args, **kwargs):
        recursion_refs = [i for i in range(len(model_cost) - 1)]
        super().__init__(model_cost, covariance, recursion_refs=recursion_refs,
                         *args, **kwargs)
