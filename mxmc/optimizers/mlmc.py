"""c
Implementation of an optimizer using the Multi-Level Monte Carlo (MLMC) method
to find the sample allocation that yields the smallest variance for a target
cost.
"""
import numpy as np
import warnings

from .optimizer_base import OptimizerBase
from mxmc.optimizers.optimizer_base import OptimizationResult
from mxmc.sample_allocations.mlmc_sample_allocation import MLMCSampleAllocation


class MLMC(OptimizerBase):
    """
    Class that implements the Multi-Level Monte Carlo (MLMC) optimizer for
    determining an optimal sample allocation across models to minimize
    estimator variance.
    NOTE:

    MLMC optimizer assumes that the high-fidelity model corresponds to the
    finest discretization and is therefore the most time consuming, so the
    first entry in the model_costs array must be the maximum.

    mlmc_variances is an array of variances of the differences between
    models on adjacent levels (except the lowest fidelity / fastest model,
    which is just the output variance), this input must be provided while
    the covariance input is not used. The array does not need to be ordered
    from high to low fidelity, but it must be arranged according to the
    model_costs array.
    """

    def __init__(self, model_costs, covariance=None):
        super().__init__(model_costs, covariance)
        self._validate_inputs(model_costs)
        self._level_costs = self._get_level_costs(self._model_costs)
        sorted_cov = self._sort_covariance_by_cost(covariance)
        self._mlmc_variances = self._get_variances_from_covariance(sorted_cov)
        self._alloc_class = MLMCSampleAllocation

    @staticmethod
    def _validate_inputs(model_costs):
        if model_costs[0] != np.max(model_costs):
            raise ValueError("First model must have highest cost for MLMC")

    def _get_level_costs(self, model_costs):

        sort_indices, model_costs_sorted = self._sort_model_costs(model_costs)
        level_costs_sorted = self._sum_adjacent_model_costs(model_costs_sorted)
        level_costs = self._unsort_level_costs(level_costs_sorted,
                                               sort_indices)
        self._cost_sort_indices = np.flip(sort_indices)  # to unsort later
        return level_costs

    def _sort_covariance_by_cost(self, cov_matrix):
        indices = np.ix_(self._cost_sort_indices, self._cost_sort_indices)
        cov_matrix_sorted = cov_matrix[indices]
        return cov_matrix_sorted

    def _get_variances_from_covariance(self, cov_matrix):
        vars_ = []
        for i in range(0, cov_matrix.shape[0] - 1):
            var = cov_matrix[i, i] + \
                  cov_matrix[i+1, i+1] - \
                  2 * cov_matrix[i, i+1]
            vars_.append(var)
        vars_.append(cov_matrix[-1, -1])

        sort_indices = self._cost_sort_indices.argsort()
        return np.array(vars_)[sort_indices]

    @staticmethod
    def _sort_model_costs(model_costs):

        sort_indices = np.argsort(model_costs)
        model_costs_sort = np.flip(model_costs[sort_indices])
        return sort_indices, model_costs_sort

    def _sum_adjacent_model_costs(self, model_costs_sorted):

        level_costs_sorted = np.copy(model_costs_sorted)
        for i in range(0, self._num_models - 1):
            level_costs_sorted[i] = model_costs_sorted[i] \
                                    + model_costs_sorted[i + 1]
        return level_costs_sorted

    def _unsort_level_costs(self, level_costs_sort, sort_indices):

        level_costs = np.zeros(self._num_models)
        level_costs_sort = np.flip(level_costs_sort)
        for i in range(0, self._num_models):
            level_costs[sort_indices[i]] = level_costs_sort[i]
        return level_costs

    def optimize(self, target_cost):

        if self._target_cost_is_too_small(target_cost):
            return self._get_invalid_result()

        return self._compute_optimization_result(target_cost)

    def _target_cost_is_too_small(self, target_cost):
        return target_cost < np.min(self._model_costs)

    def _compute_optimization_result(self, target_cost):

        samples_per_level = self._get_num_samples_per_level(target_cost)
        actual_cost = np.dot(samples_per_level, self._level_costs)

        nonzero_sample_nums = np.where(samples_per_level != 0)
        estimator_variance = np.sum(self._mlmc_variances[nonzero_sample_nums] /
                                    samples_per_level[nonzero_sample_nums])

        comp_allocation = self._make_allocation(samples_per_level)

        allocation = self._alloc_class(comp_allocation)

        return OptimizationResult(actual_cost, estimator_variance, allocation)

    def _get_num_samples_per_level(self, target_cost):

        mu_mlmc = self._calculate_mlmc_mu(target_cost)
        var_to_cost_ratios = self._mlmc_variances / self._level_costs
        samples_per_level = mu_mlmc * np.sqrt(var_to_cost_ratios)

        samples_per_level_ints = \
            self._adjust_samples_per_level(target_cost, samples_per_level)
        return samples_per_level_ints

    def _calculate_mlmc_mu(self, target_cost):

        mu_mlmc = 0.
        for var_k, cost_k in zip(self._mlmc_variances, self._level_costs):
            mu_mlmc += np.sqrt(var_k * cost_k)
        mu_mlmc = target_cost / mu_mlmc
        return mu_mlmc

    def _adjust_samples_per_level(self, target_cost, base_samples_per_level):

        samples_per_level = np.array(base_samples_per_level, dtype=int)
        remaining_cost = target_cost - \
            np.dot(samples_per_level, self._level_costs)

        sample_size_remainders = base_samples_per_level - samples_per_level
        sample_priority_indices = np.flip(np.argsort(sample_size_remainders))

        for sample_index in sample_priority_indices:

            level_cost = self._level_costs[sample_index]
            remainder = sample_size_remainders[sample_index]

            if level_cost <= remaining_cost and remainder > 0.:

                samples_per_level[sample_index] += 1
                remaining_cost -= self._level_costs[sample_index]

        return samples_per_level

    def _make_allocation(self, num_samples_per_level):

        allocation = np.zeros((self._num_models, 2 * self._num_models),
                              dtype=int)
        allocation[:, 0] = num_samples_per_level[self._cost_sort_indices]

        for model_index in range(1, self._num_models):
            cost_index = self._cost_sort_indices[model_index]
            allocation[model_index - 1, 2 * cost_index] = 1
            allocation[model_index, 2 * cost_index + 1] = 1
        allocation[0, 1] = 1

        if allocation[0, 0] == 0:

            msg1 = "No samples are allocated for the highest fidelity model!\n"
            msg2 = "Is your target cost too low?"
            warnings.warn(msg1 + msg2)

        return allocation
