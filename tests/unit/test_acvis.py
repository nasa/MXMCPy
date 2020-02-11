import numpy as np
import pytest

from mxmc.optimizer import Optimizer, ALGORITHM_MAP
from mxmc.util.testing import assert_opt_result_equal


@pytest.mark.parametrize("cost_factor", [1, 4])
@pytest.mark.parametrize("covariance_factor", [1, 4])
def test_acvmf_three_models_known_solution(cost_factor,
                                           covariance_factor, mocker):
    covariance = np.array([[1, 0.75, 0.25],
                           [0.75, 1., 0.5],
                           [0.25, 0.5, 1.]]) * covariance_factor
    model_costs = np.array([3, 2, 1])
    optimizer = Optimizer(model_costs, covariance)

    ratios_for_opt = np.array([1, 2])
    mocker.patch.object(ALGORITHM_MAP['acvis'],
                        '_solve_opt_problem',
                        return_value=ratios_for_opt)

    cost_ref = 10. * cost_factor
    var_ref = (63. / 88.) * covariance_factor / cost_factor
    allocation_ref = np.array([[1 * cost_factor, 1, 1, 1, 1, 1],
                               [1 * cost_factor, 0, 0, 1, 0, 0],
                               [2 * cost_factor, 0, 0, 0, 0, 1]], dtype=int)

    target_cost = 10 * cost_factor
    opt_result = optimizer.optimize(algorithm="acvis", target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_ref, var_ref, allocation_ref)


@pytest.mark.parametrize("cost_factor", [1, 4])
@pytest.mark.parametrize("covariance_factor", [1, 4])
def test_acvmf_three_models_unordered(cost_factor, covariance_factor, mocker):
    covariance = np.array([[1, 0.25, 0.75],
                           [0.25, 1., 0.5],
                           [0.75, 0.5, 1.]]) * covariance_factor
    model_costs = np.array([3, 1, 2])
    optimizer = Optimizer(model_costs, covariance)

    ratios_for_opt = np.array([2, 1])
    mocker.patch.object(ALGORITHM_MAP['acvis'],
                        '_solve_opt_problem',
                        return_value=ratios_for_opt)

    cost_ref = 10. * cost_factor
    var_ref = (63. / 88.) * covariance_factor / cost_factor
    allocation_ref = np.array([[1 * cost_factor, 1, 1, 1, 1, 1],
                               [2 * cost_factor, 0, 0, 1, 0, 0],
                               [1 * cost_factor, 0, 0, 0, 0, 1]], dtype=int)

    target_cost = 10 * cost_factor
    opt_result = optimizer.optimize(algorithm="acvis", target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_ref, var_ref, allocation_ref)
