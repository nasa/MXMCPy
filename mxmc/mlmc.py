
import numpy as np

from .optimizer import OptimizationResult, Optimizer


class MLMC(Optimizer):
    def __init__(self, model_costs, covariance=None, mlmc_variances=None):

        super().__init__(model_costs, covariance, mlmc_variances)

        if mlmc_variances is None:
            raise ValueError("Must specify mlmc_variances")
        self._mlmc_variances = mlmc_variances
        self._level_costs = self._get_level_costs(model_costs)

    def _get_level_costs(self, model_costs):
        '''
        To get level costs from model costs, we need to sort costs to descending
        order, then sum model costs on adjacent levels (except for last/fastest
        model), then unsort the summed costs back to the original order
        '''

        sort_indices = np.argsort(model_costs)
        level_costs_sort = np.flip(model_costs[sort_indices])

        for i in range(0, self._num_models-1):
            level_costs_sort[i] = level_costs_sort[i] + level_costs_sort[i+1]

        #Unsort the costs back to original order & store inverse map for later
        level_costs = np.zeros(self._num_models)
        level_costs_sort = np.flip(level_costs_sort)
        for i in range(0, self._num_models):
            level_costs[sort_indices[i]] = level_costs_sort[i]

        self._cost_sort_indices = np.flip(sort_indices)

        return level_costs
        

    def optimize(self, target_cost):

        if target_cost < np.min(self._model_costs):
            return self._make_invalid_result()
        else:
            mu_mlmc = 0.
            for var_k, cost_k in zip(self._mlmc_variances, self._level_costs):
                mu_mlmc += np.sqrt(var_k * cost_k)
            mu_mlmc = target_cost / mu_mlmc
    
            num_samples_per_level = mu_mlmc*np.sqrt(self._mlmc_variances / self._level_costs)

            actual_cost = np.dot(num_samples_per_level, self._level_costs)
            
            estimator_variance = np.sum(self._mlmc_variances / num_samples_per_level)

            allocation = self._get_allocation_array(num_samples_per_level)

            return OptimizationResult(actual_cost, estimator_variance, allocation)

    def _get_allocation_array(self, num_samples_per_level):


        allocation = np.zeros((self._num_models, 2*self._num_models))

        allocation[:,0] = num_samples_per_level[self._cost_sort_indices]

        print("sort index = ", self._cost_sort_indices)
        for model_index in range(1, self._num_models):
            cost_index = self._cost_sort_indices[model_index]
            allocation[model_index-1, 2*cost_index] = 1
            allocation[model_index, 2*cost_index+1] = 1
        allocation[0, 1] = 1
    
        return allocation


