import numpy as np
from scipy import optimize as scipy_optimize

from .optimizer_base import OptimizerBase, OptimizationResult

class ACVMF(OptimizerBase):

    def __init__(self, model_costs, covariance, *args, **kwargs):
        super().__init__(model_costs, covariance, *args, **kwargs)

    def optimize(self, target_cost):
        if target_cost < np.sum(self._model_costs):
            return self.get_invalid_result()

        ratios = self._solve_opt_problem(target_cost)
        
        N = target_cost/(np.dot(self._model_costs[1:], ratios) + self._model_costs[0])
        print("N = ", N)
        sample_nums = np.array([np.floor(N)] + list(np.floor(N*ratios)))
        allocation = self._make_allocation(sample_nums)

        ratios = sample_nums[1:]/sample_nums[0]
        variance = self._compute_objective_function(ratios, target_cost)
        cost = np.dot(sample_nums, self._model_costs)

        return OptimizationResult(cost, variance, allocation)

    def _solve_opt_problem(self, target_cost):
        initial_guess = np.array(range(2,self._num_models+1))
        constraints = [self._get_cost_constraint(target_cost)]
        ratios = scipy_optimize.minimize(self._compute_objective_function, 
                                         initial_guess, (target_cost, ), 
                                         method='SLSQP', constraints=constraints)
        print("ratios = ", ratios)
        return initial_guess

    def _get_cost_constraint(self, target_cost):
        
        def cost_constraint(ratios):
            return target_cost/(np.dot(self._model_costs[1:], ratios) + self._model_costs[0]) - 1
        constraint_dict = {"type": "ineq", "fun": cost_constraint}
        return constraint_dict

    def _compute_objective_function(self, ratios, target_cost):

        N = target_cost/(np.dot(self._model_costs[1:], ratios) +\
             self._model_costs[0])
        big_C = self._covariance[1:, 1:]
        c_bar = self._covariance[0, 1:] / np.sqrt(self._covariance[0,0])

        F = np.zeros((self._num_models-1, self._num_models-1))
        np.fill_diagonal(F, (ratios-1)/ratios)

        for i in range(self._num_models-2):
            for j in range(i+1, self._num_models-1):
                min_ratio = np.min([ratios[i], ratios[j]])
                F[i, j] = (min_ratio - 1)/ min_ratio
                F[j, i] = F[i, j]

        a = np.diag(F)*c_bar
        R_squared = np.dot(a, np.linalg.solve(big_C*F, a))
        variance = self._covariance[0,0]/N*(1-R_squared)
        return variance

    def _make_allocation(self, sample_nums):

        allocation = np.zeros((self._num_models, 2 * self._num_models),
                              dtype=int)
        allocation[0,1:] = np.ones(2*self._num_models-1, dtype=int)
        allocation[0, 0] = sample_nums[0]
        allocation[1:, 0] = [sample_nums[i] - sample_nums[i - 1]
                             for i in range(1, len(sample_nums))]        
        
        for k in range(self._num_models-1):
            col_index = 2*k + 3
            allocation[:k+2, col_index] = 1
        return allocation
