from .grd_optimizer import GRDOptimizer
from ..recursion_enumerator import MREnumerator


class GRDMR(MREnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GRDOptimizer(*args, **kwargs)