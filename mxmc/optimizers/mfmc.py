import numpy as np

from mxmc.sample_allocations.acv_sample_allocation import ACVSampleAllocation
from .optimizer_base import OptimizerBase, InconsistentModelError,\
                            OptimizationResult


class MFMC(OptimizerBase):

    def __init__(self, model_costs, covariance):

        super().__init__(model_costs, covariance)
        self._update_covariance_dimension()
        stdev = self._calculate_stdevs()
        correlations = (covariance[0] / stdev[0]).reshape((-1, 1)) / stdev
        aggregate_correlations = \
            self._calc_aggregate_correlations(correlations, stdev)

        self._model_order_map = list(range(self._num_models))
        self._model_order_map.sort(
                key=lambda x: abs(aggregate_correlations[x]), reverse=True)

        self._ordered_agg_corr = \
            aggregate_correlations[self._model_order_map]
        self._ordered_agg_corr = np.append(self._ordered_agg_corr, 0.)
        self._ordered_cost = self._model_costs[self._model_order_map]
        self._ordered_corr = correlations[self._model_order_map]
        self._ordered_corr = np.vstack((self._ordered_corr,
                                        np.zeros(self._ordered_corr.shape[1])))
        self._ordered_stdev = stdev[self._model_order_map]

        self._alloc_class = ACVSampleAllocation

    def _update_covariance_dimension(self):
        if self._covariance.ndim == 2:
            self._covariance = self._covariance[:, :, None]

    def _calculate_stdevs(self):
        stdev = []
        for i in range(self._covariance.shape[2]):
            stdev.append(np.sqrt(np.diag(self._covariance[:, :, i])))
        return np.array(stdev).T

    @staticmethod
    def _calc_aggregate_correlations(correlations, stdev):
        aggregate_correlations = np.sqrt(
                np.sum(correlations ** 2 * stdev ** 2, axis=1)
                / np.sum(stdev ** 2, axis=1))
        return aggregate_correlations

    def optimize(self, target_cost):
        if target_cost < self._model_costs[0]:
            return self._get_invalid_result()

        if not self._model_indices_are_consistent():
            raise InconsistentModelError("Inconsistent Models")

        sample_group_sizes = self._calculate_sample_group_sizes(target_cost)
        estimator_variance = \
            self._calculate_estimator_variance(sample_group_sizes)
        actual_cost = np.dot(self._ordered_cost, sample_group_sizes)
        comp_allocation = self._make_allocation(sample_group_sizes)

        allocation = self._alloc_class(comp_allocation)

        return OptimizationResult(actual_cost, estimator_variance, allocation)

    def _model_indices_are_consistent(self):
        for j in range(1, self._num_models):
            cost_ratio = self._ordered_cost[j - 1] / self._ordered_cost[j]
            denominator = self._ordered_agg_corr[j] ** 2 \
                          - self._ordered_agg_corr[j + 1] ** 2
            if np.isclose(denominator, 0, atol=1e-16):
                return False
            numerator = self._ordered_agg_corr[j - 1] ** 2 \
                        - self._ordered_agg_corr[j] ** 2
            req_cost_ratio = numerator / denominator
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
                                * (self._ordered_agg_corr[1:-1] ** 2 -
                                   self._ordered_agg_corr[2:] ** 2)
                                / (self._ordered_cost[1:]
                                   * (1 - self._ordered_agg_corr[1] ** 2)))
        sample_ratios = np.insert(sample_ratios, 0, 1)
        return sample_ratios

    def _calculate_estimator_variance(self, sample_group_sizes):
        alphas = self._calculate_optimal_alphas()
        estimator_variance = self._ordered_stdev[0] ** 2 \
            / sample_group_sizes[0]
        if self._num_models > 1:
            estimator_variance += np.sum((1 / sample_group_sizes[:-1][:, None]
                                          -1 / sample_group_sizes[1:][:, None])
                                         * (alphas[1:] ** 2
                                            * self._ordered_stdev[1:] ** 2
                                            + 2 * alphas[1:]
                                            * self._ordered_corr[1:-1]
                                            * self._ordered_stdev[0]
                                            * self._ordered_stdev[1:]))

        return estimator_variance

    def _calculate_optimal_alphas(self):
        alpha_star = - self._ordered_corr[:-1] * self._ordered_stdev[0] / \
                     self._ordered_stdev
        return alpha_star

    def _make_allocation(self, sample_nums):
        allocation = np.zeros((self._num_models, 2 * self._num_models),
                              dtype=int)
        allocation[0, 0] = sample_nums[0]
        allocation[1:, 0] = [sample_nums[i] - sample_nums[i - 1]
                             for i in range(1, len(sample_nums))]

        for i in range(len(sample_nums)):
            for k, j in enumerate(self._model_order_map[i:]):
                allocation[i, 1 + 2 * j] = 1
                if k > 0:
                    allocation[i, 2 * j] = 1
        return allocation
