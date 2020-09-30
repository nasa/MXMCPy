import numpy as np
import pytest

from mxmc import Optimizer, Estimator
ALGORITHMS = Optimizer.get_algorithm_names()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_vector_optim_balances_individual_variances(algorithm):
    model_costs = np.array([1, 0.1, 0.01])
    covariance = np.empty((3, 3, 2))
    covariance[:, :, 0] = np.array([[1.0, 0.9, 0.8],
                                    [0.9, 1.6, 0.7],
                                    [0.8, 0.7, 2.5]])
    covariance[:, :, 1] = np.array([[1.0, 0.9, 0.85],
                                    [0.9, 1.6, 0.65],
                                    [0.85, 0.65, 2.5]])
    target_cost = 10000
    approx_vars = []
    for cov_ind in [0, [0, 1], 1]:
        vars = _compute_optimal_var_from_vec_qoi(covariance, model_costs,
                                                 target_cost, algorithm,
                                                 cov_ind)
        approx_vars.append(vars)
    assert approx_vars[0][0] <= approx_vars[1][0] <= approx_vars[2][0]
    assert approx_vars[0][1] >= approx_vars[1][1] >= approx_vars[2][1]


def _compute_optimal_var_from_vec_qoi(covariance, model_costs, target_cost,
                                      algorithm, cov_ind):
    opt_cov = np.copy(covariance[:, :, cov_ind])
    optimizer = Optimizer(model_costs,
                          covariance=opt_cov)
    result = optimizer.optimize(algorithm=algorithm, target_cost=target_cost,
                                auto_model_selection=True)
    vars = []
    for i in range(2):
        estimator = Estimator(result.allocation, covariance[:, :, i])
        vars.append(estimator.approximate_variance)
    return vars