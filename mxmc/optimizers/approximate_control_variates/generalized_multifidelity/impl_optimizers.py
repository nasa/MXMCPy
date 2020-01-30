from .gmf_ordered import GMFOrdered
from .gmf_unordered import GMFUnordered
from ..recursion_enumerator import KLEnumerator
from ..recursion_enumerator import SREnumerator
from ..recursion_enumerator import MREnumerator


class ACVKL(KLEnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GMFOrdered(*args, **kwargs)


class ACVMFMC(GMFOrdered):

    def __init__(self, model_cost, covariance, *args, **kwargs):
        recursion_refs = [i for i in range(len(model_cost) - 1)]
        super().__init__(model_cost, covariance, recursion_refs=recursion_refs,
                         *args, **kwargs)


class ACVMF(GMFOrdered):

    def __init__(self, model_cost, covariance, *args, **kwargs):
        recursion_refs = [0] * (len(model_cost) - 1)
        super().__init__(model_cost, covariance, recursion_refs=recursion_refs,
                         *args, **kwargs)


class ACVMFU(GMFUnordered):

    def __init__(self, model_cost, covariance, *args, **kwargs):
        recursion_refs = [0] * (len(model_cost) - 1)
        super().__init__(model_cost, covariance, recursion_refs=recursion_refs,
                         *args, **kwargs)


class GMFSR(SREnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GMFUnordered(*args, **kwargs)


class GMFMR(MREnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GMFUnordered(*args, **kwargs)

