from itertools import combinations
import numpy as np
from abc import abstractmethod

from .optimizer_base import OptimizerBase
from mxmc.generalized_multi_fidelity import GMFUnordered, GMFOrdered


class NoMatchingCombosError(RuntimeError):
    pass


class RecursionEnumerator(OptimizerBase):

    def optimize(self, target_cost):
        if target_cost < np.sum(self._model_costs):
            return self._get_invalid_result()
        if self._num_models == 1:
            return self._get_monte_carlo_result(target_cost)

        best_result = None
        for recursion_refs in self._recursion_iterator():
            sub_opt = self._get_sub_optimizer(self._model_costs,
                                              self._covariance,
                                              recursion_refs=recursion_refs)

            sub_opt_result = sub_opt.optimize(target_cost)
            if best_result is None \
                    or sub_opt_result.variance < best_result.variance:
                best_result = sub_opt_result

        if best_result is None:
            error_msg = "No potential recursion enumerations"
            raise NoMatchingCombosError(error_msg)

        return best_result

    @abstractmethod
    def _get_sub_optimizer(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _recursion_iterator(self):
        raise NotImplementedError


class KLEnumerator(RecursionEnumerator):
    def _recursion_iterator(self):
        for k in range(1, self._num_models):
            if k == self._num_models - 1:
                recursion_refs = [0] * k
                yield recursion_refs
            else:
                for l in range(1, k + 1):
                    recursion_refs = [0] * k + [l] * (self._num_models - k - 1)
                    yield recursion_refs


class ACVKL(KLEnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        return GMFOrdered(*args, **kwargs)


        # def _kl_enumerator(self):
        #     for k in self._k_subset_enumerator():
        #         if len(k) == self._num_models:
        #             yield k, None
        #         else:
        #             for l in k.intersection(self._l_models):
        #                 yield k, l
        #
        # def _k_subset_enumerator(self):
        #     for i in range(len(self._k_models), 0, -1):
        #         for subset in combinations(self._k_models, i):
        #             subset = set(subset)
        #             subset.add(0)
        #             yield subset


