from .gis_optimizer import GISOptimizer
from ..recursion_enumerator import MREnumerator


class GISMR(MREnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GISOptimizer(*args, **kwargs)