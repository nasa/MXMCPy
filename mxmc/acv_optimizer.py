from abc import abstractmethod

import numpy as np
import torch
from scipy import optimize as scipy_optimize

from .optimizer_base import OptimizerBase, OptimizationResult

TORCHDTYPE = torch.double


class ACVOptimizer(OptimizerBase):

    def optimize(self, target_cost):
        if target_cost < np.sum(self._model_costs):
            return self._get_invalid_result()
        if self._num_models == 1:
            return self._get_monte_carlo_result(target_cost)

        ratios = self._solve_opt_problem(target_cost)

        sample_nums = self._compute_sample_nums_from_ratios(ratios,
                                                            target_cost)
        sample_nums = np.floor(sample_nums)
        ratios = self._compute_ratios_from_sample_nums(sample_nums)

        actual_cost = self._compute_total_cost(sample_nums)
        variance, _ = self._compute_variance_and_grad(ratios, actual_cost)
        allocation = self._make_allocation(sample_nums)

        return OptimizationResult(actual_cost, variance, allocation)

    def _compute_total_cost(self, sample_nums):
        cost = np.dot(sample_nums, self._model_costs)
        return cost

    def _solve_opt_problem(self, target_cost):
        initial_guess = self._model_costs[0] / self._model_costs[1:]
        bounds = [(1, np.inf)] * (self._num_models - 1)
        constraints = self._get_constraints(target_cost)

        slsqp_ratios = self._perform_slsqp_optim(initial_guess, bounds,
                                                 constraints, target_cost)

        nm_ratios = self._perform_nelder_mead_optim(slsqp_ratios, bounds,
                                                    constraints, target_cost)

        return nm_ratios

    def _perform_slsqp_optim(self, initial_guess, bounds, constraints,
                             target_cost):
        options = {"disp": False, "ftol": 1e-10}
        opt_result = scipy_optimize.minimize(
                self._compute_variance_and_grad,
                initial_guess, (target_cost,),
                constraints=constraints,
                bounds=bounds, jac=True,
                method='SLSQP',
                options=options)
        return opt_result.x

    def _perform_nelder_mead_optim(self, initial_guess, bounds, constraints,
                                   target_cost):
        options = {"disp": False, "xatol": 1e-12, "fatol": 1e-12,
                   "maxfev": 500 * len(initial_guess)}
        opt_result = scipy_optimize.minimize(
                self._compute_variance_using_penalties,
                initial_guess,
                args=(target_cost, bounds, constraints),
                method='Nelder-Mead',
                options=options)
        return opt_result.x

    def _get_constraints(self, target_cost):
        def n_constraint(ratios):
            N = target_cost / np.dot(self._model_costs, [1] + list(ratios))
            return N - 1

        def constraint_func(ratios, ind):
            N = target_cost / np.dot(self._model_costs, [1] + list(ratios))
            return N * (ratios[ind] - 1) - 1

        constraints = [{"type": "ineq", "fun": n_constraint, "args": tuple()}]
        for ind in range(self._num_models - 1):
            constraints.append({"type": "ineq",
                                "fun": constraint_func,
                                "args": (ind, )})
        return constraints

    def _compute_variance_and_grad(self, ratios, target_cost):
        ratios_tensor = torch.tensor(ratios, requires_grad=True,
                                     dtype=TORCHDTYPE)
        full_ratios = torch.ones(len(ratios_tensor) + 1, dtype=TORCHDTYPE)
        full_ratios[1:] = ratios_tensor
        covariance = torch.tensor(self._covariance, dtype=TORCHDTYPE)
        model_costs = torch.tensor(self._model_costs, dtype=TORCHDTYPE)
        N = target_cost / (torch.dot(model_costs, full_ratios))
        variance = self._compute_acv_estimator_variance(covariance,
                                                        ratios_tensor, N)
        variance.backward()
        result = (variance.detach().numpy(), ratios_tensor.grad.detach().numpy())
        return result

    def _compute_variance_using_penalties(self, ratios, target_cost,
                                          bounds, constraints):
        ratios_tensor = torch.tensor(ratios, dtype=TORCHDTYPE)
        full_ratios = torch.ones(len(ratios_tensor) + 1, dtype=TORCHDTYPE)
        full_ratios[1:] = ratios_tensor
        covariance = torch.tensor(self._covariance, dtype=TORCHDTYPE)
        model_costs = torch.tensor(self._model_costs, dtype=TORCHDTYPE)
        N = target_cost / (torch.dot(model_costs, full_ratios))
        variance = self._compute_acv_estimator_variance(covariance,
                                                        ratios_tensor, N)
        var = variance.detach().numpy()
        penalty = self._calaculate_penalty(bounds, constraints, ratios)

        return var + penalty

    @staticmethod
    def _calaculate_penalty(bounds, constraints, input_ratios):
        penalty_weight = 1e6
        penalty = 0
        for constr in constraints:
            c_val = constr["fun"](input_ratios, *constr["args"])
            if c_val < 0:
                penalty -= c_val * penalty_weight
        for r, (lb, ub) in zip(input_ratios, bounds):
            lb_val = r - lb
            ub_val = ub - r
            if lb_val < 0:
                penalty -= lb_val * penalty_weight
            if ub_val < 0:
                penalty -= ub_val * penalty_weight
        return penalty

    def _compute_acv_estimator_variance(self, covariance, ratios, N):

        big_C = covariance[1:, 1:]
        c_bar = covariance[0, 1:] / torch.sqrt(covariance[0, 0])

        F, F0 = self._compute_acv_F_and_F0(ratios)
        a = (F0 * c_bar).reshape((-1, 1))
        alpha, _ = torch.solve(a, big_C * F)
        R_squared = torch.dot(a.flatten(), alpha.flatten())
        variance = covariance[0, 0] / N * (1 - R_squared)
        return variance

    @staticmethod
    def _compute_ratios_from_sample_nums(sample_nums):
        ratios = sample_nums[1:] / sample_nums[0]
        return ratios

    def _compute_sample_nums_from_ratios(self, ratios, target_cost):
        N = target_cost / np.dot(self._model_costs, [1] + list(ratios))
        sample_nums = N * np.array([1] + list(ratios))
        return sample_nums

    @abstractmethod
    def _compute_acv_F_and_F0(self, ratios):
        raise NotImplementedError

    @abstractmethod
    def _make_allocation(self, sample_nums):
        raise NotImplementedError
