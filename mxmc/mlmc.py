'''
Implementation of an optimizer using the Multi-Level Monte Carlo (MLMC) method
to find the sample allocation that yields the smallest variance for a target
cost.
'''
import numpy as np

from .optimizer_base import OptimizationResult, OptimizerBase

class MLMC(OptimizerBase):
    '''
    Class that implements the Multi-Level Monte Carlo (MLMC) optimizer for
    determining an optimal sample allocation across models to minimize estimator
    variance.
    NOTE:
        *MLMC optimizer assumes that the high-fidelity model corresponds to the
        finest discretization and is therefore the most time consuming, so the
        first entry in the model_costs array must be the maximum.
        *mlmc_variances is an array of variances of the differences between
        models on adjacent levels (except the lowest fidelity / fastest model,
        which is just the output variance), this input must be provided while
        the covariance input is not used. The array does not need to be ordered
        from high to low fidelity, but it must be arranged according to the
        model_costs array.
    '''

    def __init__(self, model_costs, covariance=None, vardiff_matrix=None):

        super().__init__(model_costs, covariance)
        self._validate_inputs(model_costs, vardiff_matrix)
        self._level_costs = self._get_level_costs(model_costs)
        self._vardiff_matrix = vardiff_matrix
        sorted_vardiff = self._sort_vardiff_by_cost(vardiff_matrix)
        self._mlmc_variances = self._get_variances_from_vardiff(sorted_vardiff)

    def _validate_inputs(self, model_costs, vardiff_matrix):
        if vardiff_matrix is None:
            raise ValueError("Must specify vardiff_matrix")
        if model_costs[0] != np.max(model_costs):
            raise ValueError("First model must have highest cost for MLMC")
        self._validate_variance_matrix(vardiff_matrix)

    def _get_level_costs(self, model_costs):

        sort_indices, model_costs_sorted = self._sort_model_costs(model_costs)
        level_costs_sorted = self._sum_adjacent_model_costs(model_costs_sorted)
        level_costs = self._unsort_level_costs(level_costs_sorted, sort_indices)
        self._cost_sort_indices = np.flip(sort_indices) # to unsort later
        return level_costs

    def _sort_vardiff_by_cost(self, vardiff_matrix):
        indices = np.ix_(self._cost_sort_indices, self._cost_sort_indices)
        vardiff_matrix = vardiff_matrix[indices]
        return vardiff_matrix

    def _get_variances_from_vardiff(self, vardiff_matrix):
        var = []
        for i in range(0, vardiff_matrix.shape[0] -1):
            var.append(vardiff_matrix[i, i + 1])
        var.append(vardiff_matrix[-1, -1])

        sort_indices = self._cost_sort_indices.argsort()
        return np.array(var)[sort_indices]

    def _sort_model_costs(self, model_costs):

        sort_indices = np.argsort(model_costs)
        model_costs_sort = np.flip(model_costs[sort_indices])
        return sort_indices, model_costs_sort

    def _sum_adjacent_model_costs(self, model_costs_sorted):

        level_costs_sorted = np.copy(model_costs_sorted)
        for i in range(0, self._num_models-1):
            level_costs_sorted[i] = model_costs_sorted[i] \
                                                + model_costs_sorted[i+1]
        return level_costs_sorted

    def _unsort_level_costs(self, level_costs_sort, sort_indices):

        level_costs = np.zeros(self._num_models)
        level_costs_sort = np.flip(level_costs_sort)
        for i in range(0, self._num_models):
            level_costs[sort_indices[i]] = level_costs_sort[i]
        return level_costs

    def optimize(self, target_cost):

        if self._target_cost_is_too_small(target_cost):
            return self.get_invalid_result()

        return self._compute_optimization_result(target_cost)

    def _target_cost_is_too_small(self, target_cost):
        return target_cost < np.min(self._model_costs)

    def _compute_optimization_result(self, target_cost):

        samples_per_level = self._get_num_samples_per_level(target_cost)
        actual_cost = np.dot(samples_per_level, self._level_costs)
        estimator_variance = np.sum(self._mlmc_variances/samples_per_level)
        allocation = self._get_allocation_array(samples_per_level)
        return OptimizationResult(actual_cost, estimator_variance, allocation)

    def _get_num_samples_per_level(self, target_cost):

        mu_mlmc = self._calculate_mlmc_mu(target_cost)
        var_to_cost_ratios = self._mlmc_variances / self._level_costs
        samples_per_level = mu_mlmc * np.sqrt(var_to_cost_ratios)
        samples_per_level_ints = self._adjust_samples_per_level(target_cost,
                                                             samples_per_level)
        return samples_per_level_ints

    def _calculate_mlmc_mu(self, target_cost):

        mu_mlmc = 0.
        for var_k, cost_k in zip(self._mlmc_variances, self._level_costs):
            mu_mlmc += np.sqrt(var_k * cost_k)
        mu_mlmc = target_cost / mu_mlmc
        return mu_mlmc

    def _adjust_samples_per_level(self, target_cost, samples_per_level):
        '''
        TODO - going to need to make this more complex/robust in order to yield
        # samples that come as close as possible to the target cost without
        going over after rounding to integers. For now, just make sure we
        are rounding to integers
        '''
        sample_ints = [int((samples)) for samples in samples_per_level]
        samples_per_level_ints = np.array(sample_ints)
        return samples_per_level_ints

    def _get_allocation_array(self, num_samples_per_level):

        allocation = np.zeros((self._num_models, 2*self._num_models), dtype=int)
        allocation[:, 0] = num_samples_per_level[self._cost_sort_indices]

        for model_index in range(1, self._num_models):
            cost_index = self._cost_sort_indices[model_index]
            allocation[model_index-1, 2*cost_index] = 1
            allocation[model_index, 2*cost_index+1] = 1
        allocation[0, 1] = 1

        return allocation

    def subset(self, model_indices):
        subset_costs = np.copy(self._model_costs[model_indices])
        subset_vardiff_matrix = \
            self._get_subset_of_matrix(self._vardiff_matrix, model_indices)
        return self.__class__(subset_costs,
                              vardiff_matrix=subset_vardiff_matrix)
