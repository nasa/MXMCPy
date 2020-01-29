from .gmf_unordered import GMFUnordered
from ..recursion_enumerator import MREnumerator


class GMFMR(MREnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GMFUnordered(*args, **kwargs)