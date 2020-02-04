from .gis_optimizer import GISOptimizer
from ..recursion_enumerator import SREnumerator
from ..recursion_enumerator import MREnumerator


__all__ = ['ACVIS', 'GISSR', 'GISMR']


class ACVIS(GISOptimizer):

    def __init__(self, model_cost, covariance, *args, **kwargs):
        recursion_refs = [0] * (len(model_cost) - 1)
        super().__init__(model_cost, covariance, recursion_refs=recursion_refs,
                         *args, **kwargs)


class GISSR(SREnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GISOptimizer(*args, **kwargs)


class GISMR(MREnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GISOptimizer(*args, **kwargs)
