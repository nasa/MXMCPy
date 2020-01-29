from .gmf_ordered import GMFOrdered
from ..recursion_enumerator import KLEnumerator


class ACVKL(KLEnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GMFOrdered(*args, **kwargs)