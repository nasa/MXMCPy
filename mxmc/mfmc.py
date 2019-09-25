from collections import namedtuple
from itertools import combinations

import numpy as np

OptimizationResult = namedtuple('OptimizationResult',
                                'cost variance sample_array')


class MFMC:
    def __init__(self, covariance, model_costs):
        if len(covariance) != len(model_costs):
            raise ValueError("Covariance and model cost dimensions must match")
        if not np.allclose(covariance.transpose(), covariance):
            raise ValueError("Covariance array must be symmetric")
        self._model_costs = model_costs
        self._num_models = len(self._model_costs)
        self._stdevs = np.sqrt(np.diag(covariance))
        self._correlations = covariance[0] / self._stdevs[0] / self._stdevs

    def optimize(self, target_cost):
        if target_cost < self._model_costs[0]:
            allocation = np.ones((1, 2 * self._num_models))
            allocation[0, 0] = 0
            return OptimizationResult(0, np.inf, allocation)

        best_rel_variance = np.sqrt(self._model_costs[0])
        best_indices = [0]
        lofi_model_indices_sets = self.get_unique_subsets(range(1, self._num_models))
        for indices in lofi_model_indices_sets:
            indices += [0]
            indices.sort(key=lambda x: self._correlations[x], reverse=True)
            print(indices)

            if not self._model_indices_are_consistent(indices):
                continue

            rel_variance = 0
            for j in range(len(indices)):
                corr = self._correlations[indices[j]]
                corr_plus_1 = 0 if j == len(indices) - 1 else self._correlations[indices[j+1]]
                rel_variance += np.sqrt(self._model_costs[indices[j]] *
                                        (corr**2 - corr_plus_1**2))

            if rel_variance < best_rel_variance:
                best_rel_variance = rel_variance
                best_indices = indices

        correlations = self._correlations[best_indices]
        model_costs = self._model_costs[best_indices]
        stdevs = self._stdevs[best_indices]
        sample_nums = self._calculate_sample_nums(target_cost, correlations, model_costs)
        estimator_variance = self._calculate_estimator_variance(sample_nums, correlations, stdevs)

        actual_cost = np.dot(model_costs, sample_nums)
        allocation = self._make_allocation(sample_nums, len(best_indices))
        return OptimizationResult(actual_cost, estimator_variance, allocation)

    def _model_indices_are_consistent(self, indices):
        for j in range(1, len(indices)):
            corr_minus_1 = self._correlations[indices[j-1]]
            corr = self._correlations[indices[j]]
            corr_plus_1 = 0 if j == len(indices) -1 else self._correlations[indices[j+1]]
            if self._model_costs[indices[j-1]] / self._model_costs[indices[j]] \
                    <= (corr_minus_1**2 - corr**2)/(corr**2 - corr_plus_1**2):
                return False
        return True

    def _calculate_sample_nums(self, target_cost, correlations, model_costs):
        sample_ratios = self._calculate_sample_ratios(correlations, model_costs)
        num_hifi_samples = target_cost / np.dot(model_costs, sample_ratios)
        sample_nums = num_hifi_samples * sample_ratios
        sample_nums = np.floor(sample_nums)
        return sample_nums

    def _calculate_sample_ratios(self, correlations, model_costs):
        print(correlations)
        corr_i = correlations[1:]
        corr_i_plus_1 = correlations[2:]
        corr_i_plus_1 = np.append(corr_i_plus_1, 0)
        sample_ratios = np.sqrt(model_costs[0]
                                * (corr_i ** 2 - corr_i_plus_1 ** 2)
                                / (model_costs[1:]
                                * (1 - correlations[1] ** 2)))
        sample_ratios = np.insert(sample_ratios, 0, 1)
        return sample_ratios

    def _calculate_estimator_variance(self, sample_nums, correlations, stdevs):
        alphas = self._calculate_optimal_alphas(correlations, stdevs)
        estimator_variance = stdevs[0] ** 2 / sample_nums[0]
        for i in range(1, len(correlations)):
            estimator_variance += (1/sample_nums[i - 1] - 1/sample_nums[i]) \
                                  * (alphas[i] ** 2 * stdevs[i] ** 2
                                     - 2*alphas[i] * correlations[i]
                                     * stdevs[0] * stdevs[i])
        return estimator_variance

    def _calculate_optimal_alphas(self, correlations, stdevs):
        alpha_star = np.zeros(len(correlations))
        alpha_star[1:] = [correlations[i] * stdevs[0] / stdevs[i] for i in
                          range(1, len(correlations))]
        return alpha_star

    def _make_allocation(self, sample_nums, num_models):
        allocation = np.zeros((num_models, 2 * num_models),
                              dtype=int)
        allocation[0, 0] = sample_nums[0]
        allocation[1:, 0] = [sample_nums[i] - sum(sample_nums[:i])
                             for i in range(1, num_models)]
        for i in range(num_models):
            allocation[i, 1 + 2 * i:] = 1
        return allocation

    @staticmethod
    def get_unique_subsets(master_set):
        index_list = range(len(master_set))
        for i in range(len(index_list), 0, -1):
            for j in combinations(index_list, i):
                yield [master_set[k] for k in j]