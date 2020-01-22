from itertools import combinations

import numpy as np

from .optimizer_base import OptimizationResult, InconsistentModelError


class AutoModelSelection:
    def __init__(self, optimizer):
        self._optimizer = optimizer

    def optimize(self, target_cost):
        best_indices = None
        best_result = self._optimizer.get_invalid_result()
        num_models = self._optimizer.get_num_models()

        for indices in self._get_subsets_of_model_indices(num_models):
            best_result, best_indices = \
                self._test_candidate_optimizer(target_cost, indices,
                                               best_result, best_indices)

        if best_indices is None:
            return best_result

        sample_array = \
            self._gen_sample_array(best_result, best_indices, num_models)

        estimator_variance = best_result.variance
        actual_cost = best_result.cost

        return OptimizationResult(actual_cost, estimator_variance,
                                  sample_array)

    def _test_candidate_optimizer(self, target_cost, indices,
                                  best_result, best_indices):

        candidate_optimizer = self._optimizer.subset(indices)
        try:
            opt_result = candidate_optimizer.optimize(target_cost)
        except InconsistentModelError:
            return best_result, best_indices

        if opt_result.variance < best_result.variance:
            return opt_result, indices

        return best_result, best_indices

    @staticmethod
    def _gen_sample_array(result, indices, num_models):

        sample_array = np.zeros((len(result.sample_array), num_models * 2))
        for i, index in enumerate(indices):

            sample_array[:, index * 2: index * 2 + 2] = \
                result.sample_array[:, i * 2: i * 2 + 2]

        return sample_array

    @staticmethod
    def _get_subsets_of_model_indices(num_models):

        index_list = list(range(1, num_models))
        for subset_length in reversed(index_list):
            for subset in combinations(index_list, subset_length):
                yield [0] + list(subset)

        yield [0]
