from itertools import combinations
import numpy as np

from .optimizer_base import OptimizerBase
from .acvkl import ACVKL


class NoMatchingCombosError(RuntimeError):
    pass


class ACVKLEnumerator(OptimizerBase):

    def __init__(self, model_costs, covariance=None, k_models=None,
                 l_models=None, *args, **kwargs):
        super().__init__(model_costs, covariance, k_models=k_models,
                         *args, **kwargs)

        self._k_models, self._l_models = \
            self._make_model_specifications_feasible(k_models, l_models)

    def _make_model_specifications_feasible(self, k_models, l_models):
        feasible_k_models = set(range(1, self._num_models))
        if k_models is not None:
            feasible_k_models = set(k_models).intersection(feasible_k_models)
        feasible_l_models = set(feasible_k_models)
        if l_models is not None:
            feasible_l_models = set(l_models).intersection(feasible_l_models)
        return feasible_k_models, feasible_l_models

    def optimize(self, target_cost):
        if target_cost < np.sum(self._model_costs):
            return self._get_invalid_result()
        if self._num_models == 1:
            return self._get_monte_carlo_result(target_cost)

        best_result = None
        for k, l in self._kl_enumerator():
            sub_opt = ACVKL(self._model_costs, self._covariance, k=k, l=l)
            sub_opt_result = sub_opt.optimize(target_cost)
            if best_result is None \
                    or sub_opt_result.variance < best_result.variance:
                best_result = sub_opt_result

        if best_result is None:
            error_msg = "Specified acvkl parameters lead to no valid " + \
                        "combinations of k and l"
            raise NoMatchingCombosError(error_msg)

        return best_result

    def _kl_enumerator(self):
        for k in self._k_subset_enumerator():
            if len(k) == self._num_models:
                yield k, None
            else:
                for l in k.intersection(self._l_models):
                    yield k, l

    def _k_subset_enumerator(self):
        for i in range(len(self._k_models), 0, -1):
            for subset in combinations(self._k_models, i):
                subset = set(subset)
                subset.add(0)
                yield subset


