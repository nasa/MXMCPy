import numpy as np
from scipy import optimize as scipy_optimize
import torch

from .optimizer_base import OptimizerBase, OptimizationResult

class ACVMF(OptimizerBase):

    def __init__(self, model_costs, covariance, *args, **kwargs):
        super().__init__(model_costs, covariance, *args, **kwargs)

    def optimize(self, target_cost):
        if target_cost < np.sum(self._model_costs):
            return self.get_invalid_result()
        if self._num_models == 1:
            return self._get_monte_carlo_opt_result(target_cost)

        sample_nums = self._solve_opt_problem(target_cost)
        sample_nums = np.floor(sample_nums)
        allocation = self._make_allocation(sample_nums)
        variance, _ = self._compute_objective_function(sample_nums, target_cost)
        cost = self._get_total_cost(sample_nums)

        return OptimizationResult(cost, variance, allocation)

    def _get_monte_carlo_opt_result(self, target_cost):

        sample_nums = np.floor(np.array([target_cost/self._model_costs[0]]))
        variance = self._covariance[0,0]/sample_nums[0]
        cost = self._get_total_cost(sample_nums)
        allocation = self._make_allocation(sample_nums)
        return OptimizationResult(cost, variance, allocation)

    def _solve_opt_problem(self, target_cost):
        
        initial_guess = np.ones(self._num_models)
        constraints = [self._get_cost_constraint(target_cost)]
        bounds = [(1, np.inf) for i in range(self._num_models)]
        options = {"disp": False}
        opt_result = scipy_optimize.minimize(self._compute_objective_function, 
                                             initial_guess, (target_cost, ), 
                                             constraints=constraints,
                                             bounds=bounds, jac=True,
                                             method='SLSQP',
                                             options=options)
        return opt_result.x

    def _get_total_cost(self, sample_nums):

        total_sample_nums = np.zeros(len(sample_nums))
        total_sample_nums[0] = sample_nums[0]
        total_sample_nums[1:] = sample_nums[0] + sample_nums[1:]
        cost = np.dot(total_sample_nums, self._model_costs)
        return cost

    def _get_cost_constraint(self, target_cost):

        def constraint_func(sample_nums):
            N = sample_nums[0]
            cost = N*self._model_costs[0]
            for i in range(self._num_models-2):
                cost +=  self._model_costs[i+1]*(N + sample_nums[i+1])
            return target_cost - cost

        constraint_dict = {"type": "ineq", "fun": constraint_func}
        return constraint_dict

    def _compute_objective_function(self, sample_nums, target_cost):

        sample_nums = torch.tensor(sample_nums, requires_grad=True)
        covariance = torch.tensor(self._covariance)
        model_costs = torch.tensor(self._model_costs)
        variance = self._compute_acv_estimator_variance(covariance, sample_nums)
        variance.backward()       
        return (variance.detach().numpy(), sample_nums.grad.detach().numpy())

    def _compute_acv_estimator_variance(self, covariance, sample_nums):

        big_C = covariance[1:, 1:]
        c_bar = covariance[0, 1:] / torch.sqrt(covariance[0,0])
        ratios = self._compute_ratios_from_sample_nums(sample_nums)
        F = self._compute_acv_F_matrix(ratios)
        a = (torch.diag(F)*c_bar).reshape((-1,1))
        alpha, _ = torch.solve(a, big_C*F)
        R_squared = torch.dot(a.flatten(),  alpha.flatten())
        variance = covariance[0,0]/sample_nums[0]*(1-R_squared)
        return variance

    def _compute_ratios_from_sample_nums(self, sample_nums):
        ratios = torch.zeros(len(sample_nums)-1, dtype=torch.double)
        N = sample_nums[0]
        for i in range(self._num_models-1):
            ratios[i] = 1. + sample_nums[i+1]/N
        return ratios

    def _compute_acv_F_matrix(self, ratios):

        F = torch.zeros((self._num_models-1, self._num_models-1))
        for i in range(self._num_models-1):
            F[i, i] = (ratios[i]-1)/ratios[i]
        for i in range(self._num_models-2):
            for j in range(i+1, self._num_models-1):
                min_ratio = torch.min(ratios[[i,j]])
                F[i, j] = (min_ratio - 1)/ min_ratio
                F[j, i] = F[i, j]
        return F

    def _make_allocation(self, sample_nums):

        allocation = np.zeros((self._num_models, 2 * self._num_models),
                              dtype=int)
        allocation[0,1:] = np.ones(2*self._num_models-1, dtype=int)
        allocation[0, 0] = sample_nums[0]

        deltaNs_unused = list(sample_nums[1:])
        deltaNs_used = []

        for i in range(self._num_models-1):
            min_ind = deltaNs_unused.index(min(deltaNs_unused))
            true_ind = np.argwhere(sample_nums[1:] == min(deltaNs_unused))
            min_deltaN = deltaNs_unused[min_ind]
            allocation[i+1, 0] = min_deltaN - sum(deltaNs_used)
            allocation[:i+2, 3+2*true_ind] = 1
            deltaNs_used.append(deltaNs_unused.pop(min_ind))

        return allocation
