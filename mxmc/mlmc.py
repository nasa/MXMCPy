
import numpy as np

from .optimizer import OptimizationResult, Optimizer


class MLMC(Optimizer):
    def __init__(self, model_costs, covariance=None, mlmc_variances=None):

        super().__init__(model_costs, covariance, mlmc_variances)

        if mlmc_variances is None:
            raise ValueError("Must specify mlmc_variances")
        self._mlmc_variances = mlmc_variances
        self._level_costs = np.zeros(len(self._model_costs))
        for i in range(0, len(self._model_costs)-1):
            self._level_costs[i] = self._model_costs[i] + self._model_costs[i+1]
        self._level_costs[-1] = self._model_costs[-1]

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
            allocation = np.zeros((self._num_models, 2*self._num_models))

            allocation[:,0] = num_samples_per_level
            for model_index in range(self._num_models-1):
                allocation[model_index, 2*model_index+1] = 1
                allocation[model_index, 2*model_index+2] = 1
            allocation[-1,-1] = 1

            return OptimizationResult(actual_cost, estimator_variance, allocation)
