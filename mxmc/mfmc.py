from itertools import combinations

import numpy as np

from .optimizer import OptimizationResult, Optimizer, InconsistentModelError


class MFMC(Optimizer):
    def __init__(self, model_costs, covariance, *_, **__):
        super().__init__(model_costs, covariance)
        stdev = np.sqrt(np.diag(covariance))
        correlations = covariance[0] / stdev[0] / stdev
        self._model_order_map = list(range(self._num_models))
        self._model_order_map.sort(key=lambda x: correlations[x], reverse=True)
        self._ordered_corr = correlations[self._model_order_map]
        self._ordered_corr = np.append(self._ordered_corr, 0.)
        self._ordered_cost = model_costs[self._model_order_map]
        self._ordered_stdev = stdev[self._model_order_map]

    def optimize(self, target_cost):
        if target_cost < self._model_costs[0]:
            return self.get_invalid_result()

        if not self._model_indices_are_consistent():
            raise InconsistentModelError("Inconsistent Models")

        sample_group_sizes = self._calculate_sample_group_sizes(target_cost)
        estimator_variance = \
            self._calculate_estimator_variance(sample_group_sizes)
        actual_cost = np.dot(self._ordered_cost, sample_group_sizes)
        allocation = self._make_allocation(sample_group_sizes)
        return OptimizationResult(actual_cost, estimator_variance, allocation)

    def _model_indices_are_consistent(self):
        for j in range(1, self._num_models):
            cost_ratio = self._ordered_cost[j-1] / self._ordered_cost[j]
            req_cost_ratio = (self._ordered_corr[j-1]**2 -
                              self._ordered_corr[j]**2) / \
                             (self._ordered_corr[j]**2 -
                              self._ordered_corr[j+1]**2)
            if cost_ratio <= req_cost_ratio:
                return False
        return True

    def _calculate_sample_group_sizes(self, target_cost):
        sample_ratios = self._calculate_sample_ratios()
        num_hifi_samples = target_cost / np.dot(self._ordered_cost,
                                                sample_ratios)
        sample_group_sizes = num_hifi_samples * sample_ratios
        sample_group_sizes = np.floor(sample_group_sizes)
        return sample_group_sizes

    def _calculate_sample_ratios(self):
        if self._num_models == 1:
            return np.array([1])
        sample_ratios = np.sqrt(self._ordered_cost[0]
                                * (self._ordered_corr[1:-1] ** 2 -
                                   self._ordered_corr[2:] ** 2)
                                / (self._ordered_cost[1:]
                                * (1 - self._ordered_corr[1] ** 2)))
        sample_ratios = np.insert(sample_ratios, 0, 1)
        return sample_ratios

    def _calculate_estimator_variance(self, sample_group_sizes):
        alphas = self._calculate_optimal_alphas()
        estimator_variance = self._ordered_stdev[0] ** 2 \
                             / sample_group_sizes[0]
        for i in range(1, self._num_models):
            estimator_variance += (1 / sample_group_sizes[i - 1]
                                   - 1 / sample_group_sizes[i]) \
                                  * (alphas[i] ** 2
                                     * self._ordered_stdev[i] ** 2
                                     - 2 * alphas[i] * self._ordered_corr[i]
                                     * self._ordered_stdev[0]
                                     * self._ordered_stdev[i])
        return estimator_variance

    def _calculate_optimal_alphas(self):
        alpha_star = self._ordered_corr[:-1] * self._ordered_stdev[0] / \
                     self._ordered_stdev
        return alpha_star

    def _make_allocation(self, sample_nums):
        allocation = np.zeros((self._num_models, 2 * self._num_models),
                              dtype=int)
        allocation[0, 0] = sample_nums[0]
        allocation[1:, 0] = [sample_nums[i] - sum(sample_nums[:i])
                             for i in range(1, len(sample_nums))]

        for i in range(len(sample_nums)):
            for k, j in enumerate(self._model_order_map[i:]):
                allocation[i, 1 + 2 * j] = 1
                if k > 0:
                    allocation[i, 2 * j] = 1
        return allocation

    @staticmethod
    def get_unique_subsets(master_set):
        index_list = range(len(master_set))
        for i in range(len(index_list), 0, -1):
            for j in combinations(index_list, i):
                yield [master_set[k] for k in j]
