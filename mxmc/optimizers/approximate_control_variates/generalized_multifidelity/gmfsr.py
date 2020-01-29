from .gmf_unordered import GMFUnordered
from ..recursion_enumerator import SREnumerator


class GMFSR(SREnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GMFUnordered(*args, **kwargs)