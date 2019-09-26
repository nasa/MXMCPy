
import numpy as np

from .optimizer import OptimizationResult, Optimizer


class MLMC(Optimizer):

    def __init__(self, model_costs, covariance=None, mlmc_variances=None):

        super().__init__(model_costs, covariance, mlmc_variances)
        self._validate_inputs(model_costs, mlmc_variances)
        self._mlmc_variances = mlmc_variances
        self._level_costs = self._get_level_costs(model_costs)

    def _validate_inputs(self, model_costs, mlmc_variances):

        if mlmc_variances is None:
            raise ValueError("Must specify mlmc_variances")
        if model_costs[0] !=  np.max(model_costs):
            raise ValueError("First model must have highest cost for MLMC")

    def _get_level_costs(self, model_costs):

        sort_indices, model_costs_sorted = self._sort_model_costs(model_costs)
        level_costs_sorted = self._sum_adjacent_model_costs(model_costs_sorted)
        level_costs = self._unsort_level_costs(level_costs_sorted, sort_indices)
        self._cost_sort_indices = np.flip(sort_indices) # to unsort later
        return level_costs
        
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
            return self._make_invalid_result()
        else:
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
        var_to_cost_ratio = self._mlmc_variances / self._level_costs
        num_samples_per_level = mu_mlmc*np.sqrt(var_to_cost_ratio)
        return num_samples_per_level

    def _calculate_mlmc_mu(self, target_cost):

        mu_mlmc = 0.
        for var_k, cost_k in zip(self._mlmc_variances, self._level_costs):
            mu_mlmc += np.sqrt(var_k * cost_k)
        mu_mlmc = target_cost / mu_mlmc
        return mu_mlmc

    def _get_allocation_array(self, num_samples_per_level):

        allocation = np.zeros((self._num_models, 2*self._num_models))
        allocation[:,0] = num_samples_per_level[self._cost_sort_indices]

        for model_index in range(1, self._num_models):
            cost_index = self._cost_sort_indices[model_index]
            allocation[model_index-1, 2*cost_index] = 1
            allocation[model_index, 2*cost_index+1] = 1
        allocation[0, 1] = 1
    
        return allocation


