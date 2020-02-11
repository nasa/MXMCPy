from itertools import combinations

import numpy as np

from .optimizer_base import InconsistentModelError
from mxmc.optimizers.optimization_result import OptimizationResult


class AutoModelSelection:
    def __init__(self, optimizer):
        self._optimizer = optimizer

    def optimize(self, target_cost):
        best_indices = None
        best_result = self._optimizer._get_invalid_result()
        num_models = self._optimizer.get_num_models()

        for indices in self._get_subsets_of_model_indices(num_models):
            best_result, best_indices = \
                self._test_candidate_optimizer(target_cost, indices,
                                               best_result, best_indices)

        if best_indices is None:
            return best_result

        best_sample_array = best_result.allocation.compressed_allocation
        sample_array = self._gen_sample_array(best_sample_array, best_indices,
                                              num_models)

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
    def _gen_sample_array(result_sample, indices, num_models):

        sample_array = np.zeros((result_sample.shape[0], num_models * 2),
                                dtype=int)
        sample_array[:, indices * 2] = result_sample[:, 0:len(indices)*2:2]
        sample_array[:, indices * 2 + 1] = result_sample[:, 1:len(indices)*2:2]

        return sample_array

    @staticmethod
    def _get_subsets_of_model_indices(num_models):

        indices = list(range(1, num_models))
        for subset_length in reversed(indices):
            for subset in combinations(indices, subset_length):

                all_indices = [0] + list(subset)
                yield np.array(all_indices, dtype=int)

        yield np.zeros(1, dtype=int)
