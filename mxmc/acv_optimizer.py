from abc import abstractmethod

import numpy as np
import torch
from scipy import optimize as scipy_optimize

from .optimizer_base import OptimizerBase, OptimizationResult

TORCHDTYPE = torch.double


class ACVOptimizer(OptimizerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt_path = np.empty((0, 3), dtype=float)

    def optimize(self, target_cost):
        if target_cost < np.sum(self._model_costs):
            return self.get_invalid_result()
        if self._num_models == 1:
            return self._get_monte_carlo_opt_result(target_cost)

        sample_nums = self._solve_opt_problem_ratios(target_cost)
        sample_nums = np.floor(sample_nums)
        allocation = self._make_allocation(sample_nums)
        variance, _ = self._compute_objective_function(sample_nums,
                                                       target_cost)
        cost = self._get_total_cost(sample_nums)

        return OptimizationResult(cost, variance, allocation)  # :, self.opt_path

    def _get_monte_carlo_opt_result(self, target_cost):

        sample_nums = np.floor(np.array([target_cost / self._model_costs[0]]))
        variance = self._covariance[0, 0] / sample_nums[0]
        cost = self._get_total_cost(sample_nums)
        allocation = self._make_allocation(sample_nums)
        return OptimizationResult(cost, variance, allocation)

    def _get_total_cost(self, sample_nums):

        total_sample_nums = np.zeros(len(sample_nums))
        total_sample_nums[0] = sample_nums[0]
        total_sample_nums[1:] = sample_nums[0] + sample_nums[1:]
        cost = np.dot(total_sample_nums, self._model_costs)
        return cost

###############################################################################
        # Ratios

    def _solve_opt_problem_ratios(self, target_cost):

        initial_guess = self._model_costs[0] / self._model_costs[1:]
        # initial_guess = np.array([4.20000000e+01,  4.13636364e+01])
        bounds = [(1, np.inf)] * (self._num_models - 1)

        self._constraints = self._get_ratio_constraints_SLSQP(target_cost)
        options = {"disp": True, "ftol": 1e-10}

        # constraints = self._get_ratio_constraints_trust(target_cost)
        # options = {"disp": True, "maxiter": 5000}
        opt_result = scipy_optimize.minimize(
                lambda x, y: self._compute_objective_function_ratio(x,y),
                                             initial_guess, (target_cost,),
                                             constraints=self._constraints,
                                             bounds=bounds, jac=True,
                                             # hess="3-point",
                                             hess=self._compute_hessian_ratio,
                                             method='SLSQP',
                                             options=options)

        initial_guess = opt_result.x
        options = {"disp": True, "xatol": 1e-12, "fatol": 1e-12, "maxfev":500*len(initial_guess)}
        opt_result = scipy_optimize.minimize(
                lambda x, y: self._compute_objective_function_ratio_penalty(x, y)[0],
                initial_guess, (target_cost,),
                method='Nelder-Mead',
                options=options)
        #
        # options = {"disp": True, "gtol": 1e-12}
        # opt_result = scipy_optimize.minimize(self._compute_objective_function_ratio,
        #                                      initial_guess, (target_cost,),
        #                                      jac=True,
        #                                      hess=self._compute_hessian_ratio,
        #                                      method='CG',
        #                                      options=options)

        N = target_cost / np.dot(self._model_costs, [1] + list(opt_result.x))
        return N*np.array([1] + list(opt_result.x))

    def _get_ratio_constraints_SLSQP(self, target_cost):
        def n_constraint(ratios):
            N = target_cost / np.dot(self._model_costs, [1] + list(ratios))
            return N - 1

        def constraint_func(ratios, ind):
            N = target_cost / np.dot(self._model_costs, [1] + list(ratios))
            return N * (ratios[ind] - 1) - 1

        constraints = [{"type": "ineq", "fun": n_constraint}]
        for ind in range(self._num_models - 1):
            constraints.append({"type": "ineq",
                                "fun": lambda r: constraint_func(r, ind)})
        return constraints

    def _get_ratio_constraints_trust(self, target_cost):
        def n_constraint(ratios):
            N = target_cost / np.dot(self._model_costs, [1] + list(ratios))
            return N - 1

        def constraint_func(ratios, ind):
            N = target_cost / np.dot(self._model_costs, [1] + list(ratios))
            return N * (ratios[ind] - 1) - 1

        constraints = [scipy_optimize.NonlinearConstraint(n_constraint, 0, np.inf)]
        for ind in range(self._num_models - 1):
            constraints.append(scipy_optimize.NonlinearConstraint(
                    lambda r: constraint_func(r, ind), 0, np.inf))
        return constraints

    def _compute_objective_function_ratio(self, ratios, target_cost):
        input_ratios = list(ratios)
        ratios = torch.tensor(ratios, requires_grad=True, dtype=TORCHDTYPE)
        full_ratios = torch.ones(len(ratios) + 1, dtype=TORCHDTYPE)
        full_ratios[1:] = ratios
        covariance = torch.tensor(self._covariance, dtype=TORCHDTYPE)
        model_costs = torch.tensor(self._model_costs, dtype=TORCHDTYPE)
        N = target_cost / (torch.dot(model_costs, full_ratios))
        sample_nums = N * full_ratios
        variance = self._compute_acv_estimator_variance(covariance,
                                                        sample_nums)
        variance.backward()
        result = (variance.detach().numpy(), ratios.grad.detach().numpy())
        # print(input_ratios + [result[0].flatten()[0]], ",")
        # print(result[0], input_ratios)
        return result

    def _compute_objective_function_ratio_penalty(self, ratios, target_cost):
        input_ratios = list(ratios)
        ratios = torch.tensor(ratios, requires_grad=True, dtype=TORCHDTYPE)
        full_ratios = torch.ones(len(ratios) + 1, dtype=TORCHDTYPE)
        full_ratios[1:] = ratios
        covariance = torch.tensor(self._covariance, dtype=TORCHDTYPE)
        model_costs = torch.tensor(self._model_costs, dtype=TORCHDTYPE)
        N = target_cost / (torch.dot(model_costs, full_ratios))
        sample_nums = N * full_ratios
        variance = self._compute_acv_estimator_variance(covariance,
                                                        sample_nums)
        variance.backward()
        var, grad = (variance.detach().numpy(), ratios.grad.detach().numpy())
        # print(input_ratios + [result[0].flatten()[0]], ",")
        penalty_weight = 1e6
        for c in self._constraints:
            val = c["fun"](input_ratios)
            if val < 0:
                var -= val*penalty_weight
        for r in input_ratios:
            val = r - 1
            if val < 0:
                var -= val*penalty_weight
        # print(var, input_ratios,
        #       [c["fun"](input_ratios) for c in self._constraints])

        # self.opt_path = np.append(self.opt_path,
        #                           np.array([input_ratios + [result[0]]]),
        #                           axis=0)
        return var, grad

    def _compute_hessian_ratio(self, ratios, target_cost):
        ratios = torch.tensor(ratios, requires_grad=True, dtype=TORCHDTYPE)
        full_ratios = torch.ones(len(ratios) + 1, dtype=TORCHDTYPE)
        full_ratios[1:] = ratios
        covariance = torch.tensor(self._covariance, dtype=TORCHDTYPE)
        model_costs = torch.tensor(self._model_costs, dtype=TORCHDTYPE)
        N = target_cost / (torch.dot(model_costs, full_ratios))
        sample_nums = N * full_ratios
        variance = self._compute_acv_estimator_variance(covariance,
                                                        sample_nums)
        grad = torch.autograd.grad(variance, ratios, create_graph=True)[0]
        hess_list = [torch.autograd.grad(gi, ratios, retain_graph=True)[0]
                     for gi in grad]
        hess = torch.stack(hess_list)
        return hess.detach().numpy()

###############################################################################
# cost-weighted Ratios

    def _solve_opt_problem_wratios(self, target_cost):

        initial_guess = self._model_costs[0] * np.ones(self._num_models - 1)

        constraints = self._get_wratio_constraints_SLSQP(target_cost)
        bounds = [(1, np.inf)] * (self._num_models - 1)
        options = {"disp": True, "ftol": 1e-10}
        opt_result = scipy_optimize.minimize(self._compute_objective_function_wratio,
                                             initial_guess, (target_cost,),
                                             constraints=constraints,
                                             bounds=bounds, jac=True,
                                             method='SLSQP',
                                             options=options)

        N = target_cost / (self._model_costs[0] + sum(opt_result.x))
        return N*np.array([1] + list(opt_result.x / self._model_costs[1:]))

    def _get_wratio_constraints_SLSQP(self, target_cost):
        def n_constraint(wratios):
            N = target_cost / (self._model_costs[0] + sum(wratios))
            return N - 1

        def constraint_func(wratios, ind):
            N = target_cost / (self._model_costs[0] + sum(wratios))
            return N * (wratios[ind]/self._model_costs[ind + 1] - 1) - 1

        constraints = [{"type": "ineq", "fun": n_constraint}]
        for ind in range(self._num_models - 1):
            constraints.append({"type": "ineq",
                                "fun": lambda r: constraint_func(r, ind)})
        return constraints

    def _compute_objective_function_wratio(self, wratios, target_cost):
        input_ratios = list(wratios / self._model_costs[1:])
        wratios = torch.tensor(wratios, requires_grad=True, dtype=TORCHDTYPE)
        model_costs = torch.tensor(self._model_costs, dtype=TORCHDTYPE)
        ratios = wratios / model_costs[1:]
        full_ratios = torch.ones(len(ratios) + 1, dtype=TORCHDTYPE)
        full_ratios[1:] = ratios
        covariance = torch.tensor(self._covariance, dtype=TORCHDTYPE)
        model_costs = torch.tensor(self._model_costs, dtype=TORCHDTYPE)
        N = target_cost / (torch.dot(model_costs, full_ratios))
        sample_nums = N * full_ratios
        variance = self._compute_acv_estimator_variance(covariance,
                                                        sample_nums)
        variance.backward()
        result = (variance.detach().numpy(), wratios.grad.detach().numpy())
        # print(list(ratios.flatten()) + [result[0].flatten()[0]], ",")
        # print(input_ratios, result)
        return result

###############################################################################
# sample_nums

    def _solve_opt_problem(self, target_cost):

        # initial_guess = np.ones(self._num_models)

        initial_guess = target_cost / self.get_num_models() / self._model_costs
        initial_guess[1:] -= initial_guess[0]

        constraints = [self._get_cost_constraint(target_cost)]
        bounds = [(1, np.inf) for i in range(self._num_models)]
        options = {"disp": True, "ftol": 1e-10}
        opt_result = scipy_optimize.minimize(self._compute_objective_function,
                                             initial_guess, (target_cost,),
                                             constraints=constraints,
                                             bounds=bounds, jac=True,
                                             method='SLSQP',
                                             options=options)
        return opt_result.x

    def _get_cost_constraint(self, target_cost):

        def constraint_func(sample_nums):
            N = sample_nums[0]
            cost = N * self._model_costs[0]
            for i in range(self._num_models - 2):
                cost += self._model_costs[i + 1] * (N + sample_nums[i + 1])
            return target_cost - cost

        constraint_dict = {"type": "ineq", "fun": constraint_func}
        return constraint_dict

    def _compute_objective_function(self, sample_nums, target_cost):
        input_sample_num = list(sample_nums)
        sample_nums = torch.tensor(sample_nums, requires_grad=True,
                                   dtype=TORCHDTYPE)
        covariance = torch.tensor(self._covariance, dtype=TORCHDTYPE)
        model_costs = torch.tensor(self._model_costs, dtype=TORCHDTYPE)
        variance = self._compute_acv_estimator_variance(covariance,
                                                        sample_nums)
        variance.backward()
        result = (variance.detach().numpy(), sample_nums.grad.detach().numpy())
        # print(input_sample_num + [result[0].flatten()[0]], ",")
        # print(result[1])
        return result

###############################################################################

    def _compute_acv_estimator_variance(self, covariance, sample_nums):

        big_C = covariance[1:, 1:]
        c_bar = covariance[0, 1:] / torch.sqrt(covariance[0, 0])
        ratios = self._compute_ratios_from_sample_nums(sample_nums)
        F = self._compute_acv_F_matrix(ratios)
        a = (torch.diag(F) * c_bar).reshape((-1, 1))
        alpha, _ = torch.solve(a, big_C * F)
        R_squared = torch.dot(a.flatten(), alpha.flatten())
        variance = covariance[0, 0] / sample_nums[0] * (1 - R_squared)
        return variance

    def _compute_ratios_from_sample_nums(self, sample_nums):
        ratios = torch.zeros(len(sample_nums) - 1, dtype=torch.double)
        N = sample_nums[0]
        for i in range(self._num_models - 1):
            ratios[i] = 1. + sample_nums[i + 1] / N
        return ratios

    @abstractmethod
    def _compute_acv_F_matrix(self, ratios):
        raise NotImplementedError

    @abstractmethod
    def _make_allocation(self, sample_nums):
        raise NotImplementedError
