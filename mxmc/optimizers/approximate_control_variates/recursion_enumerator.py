from itertools import combinations
import numpy as np
from abc import abstractmethod

from ..optimizer_base import OptimizerBase


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


class SREnumerator(RecursionEnumerator):
    def _recursion_iterator(self):
        for subset in self._subsets_excluding_0():
            if len(subset) == self._num_models - 1:
                recursion_refs = [0] * (self._num_models - 1)
                yield recursion_refs
            else:
                for ref in subset:
                    recursion_refs = [0 if i in subset else ref
                                      for i in range(1, self._num_models)]
                    yield recursion_refs

    def _subsets_excluding_0(self):
        for i in range(self._num_models, 0, -1):
            for subset in combinations(range(1, self._num_models), i):
                subset = set(subset)
                yield subset


class MREnumerator(RecursionEnumerator):
    def _recursion_iterator(self):
        starting_refs = [None] * (self._num_models - 1)
        ref_iter = MREnumerator._recursive_refs(starting_refs)
        possibilities = [i for i in ref_iter]
        possibilities = list(set(possibilities))
        for i in possibilities:
            yield list(i)

    @staticmethod
    def _recursive_refs(refs, last=0, banned=None):
        if banned is None:
            banned = []
        possible_indices = [i for i, r in enumerate(refs) if r is None]
        if len(possible_indices) == 0:
            yield tuple(refs)
            return

        for ind in possible_indices:
            possible_values = [0] + [i + 1 for i, r in enumerate(refs)
                                     if r is not None]
            possible_values = [i for i in possible_values if
                               i not in refs[ind:]]
            possible_values = [i for i in possible_values if i not in banned]
            for val in possible_values:
                new_refs = list(refs)
                new_refs[ind] = val
                new_banned = list(banned)
                if val != last:
                    new_banned += [last]
                for r in MREnumerator._recursive_refs(new_refs, val,
                                                      new_banned):
                    yield r
