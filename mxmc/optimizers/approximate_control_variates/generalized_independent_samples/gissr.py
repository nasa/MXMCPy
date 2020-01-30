from .gis_optimizer import GISOptimizer
from ..recursion_enumerator import SREnumerator


class GISSR(SREnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GISOptimizer(*args, **kwargs)