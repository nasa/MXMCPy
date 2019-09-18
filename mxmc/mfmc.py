from collections import namedtuple

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
        if target_cost < sum(self._model_costs):
            allocation = np.ones((1, 2 * self._num_models))
            allocation[0, 0] = 0
            return OptimizationResult(0, np.inf, allocation)

        sample_nums = self._calculate_sample_nums(target_cost)

        actual_cost = np.dot(self._model_costs, sample_nums)
        estimator_variance = self._calculate_estimator_variance(sample_nums)
        allocation = self._make_allocation(sample_nums)

        return OptimizationResult(actual_cost, estimator_variance, allocation)

    def _calculate_sample_nums(self, target_cost):
        sample_ratios = self._calculate_sample_ratios()
        num_hifi_samples = target_cost / np.dot(self._model_costs,
                                                sample_ratios)
        sample_nums = num_hifi_samples * sample_ratios
        sample_nums = np.floor(sample_nums)
        return sample_nums

    def _calculate_sample_ratios(self):
        corr_i = self._correlations[1:]
        corr_i_plus_1 = self._correlations[2:]
        corr_i_plus_1 = np.append(corr_i_plus_1, 0)
        sample_ratios = np.sqrt(self._model_costs[0]
                                * (corr_i ** 2 - corr_i_plus_1 ** 2)
                                / (self._model_costs[1:]
                                * (1 - self._correlations[1] ** 2)))
        sample_ratios = np.insert(sample_ratios, 0, 1)
        return sample_ratios

    def _calculate_estimator_variance(self, sample_nums):
        alphas = self._calculate_optimal_alphas()
        estimator_variance = self._stdevs[0] ** 2 / sample_nums[0]
        for i in range(1, self._num_models):
            estimator_variance += (1/sample_nums[i - 1] - 1/sample_nums[i]) \
                                  * (alphas[i] ** 2 * self._stdevs[i] ** 2
                                     - 2*alphas[i] * self._correlations[i]
                                     * self._stdevs[0] * self._stdevs[i])
        return estimator_variance

    def _calculate_optimal_alphas(self):
        alpha_star = np.zeros(self._num_models)
        alpha_star[1:] = [self._correlations[i] * self._stdevs[0]
                          / self._stdevs[i] for i in
                          range(1, self._num_models)]
        return alpha_star

    def _make_allocation(self, sample_nums):
        allocation = np.zeros((self._num_models, 2 * self._num_models),
                              dtype=int)
        allocation[0, 0] = sample_nums[0]
        allocation[1:, 0] = [sample_nums[i] - sum(sample_nums[:i])
                             for i in range(1, self._num_models)]
        for i in range(self._num_models):
            allocation[i, 1 + 2 * i:] = 1
        return allocation
