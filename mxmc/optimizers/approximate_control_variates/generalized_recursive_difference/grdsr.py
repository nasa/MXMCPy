from .grd_optimizer import GRDOptimizer
from ..recursion_enumerator import SREnumerator


class GRDSR(SREnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GRDOptimizer(*args, **kwargs)