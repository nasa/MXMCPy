import numpy as np
import pytest

from mxmc.optimizer import Optimizer, ALGORITHM_MAP
from mxmc.util.testing import assert_opt_result_equal


@pytest.mark.parametrize("algorithm", ["acvmf", "acvmfu"])
@pytest.mark.parametrize("cost_factor", [1, 4])
@pytest.mark.parametrize("covariance_factor", [1, 4])
def test_acvmf_three_models_known_solution(algorithm, cost_factor,
                                           covariance_factor, mocker):
    covariance = np.array([[1, 0.75, 0.25],
                           [0.75, 1., 0.5],
                           [0.25, 0.5, 1.]]) * covariance_factor
    model_costs = np.array([3, 2, 1])
    optimizer = Optimizer(model_costs, covariance)

    ratios_for_opt = np.array([2, 3])
    mocker.patch.object(ALGORITHM_MAP[algorithm],
                        '_solve_opt_problem',
                        return_value=ratios_for_opt)

    cost_ref = 10. * cost_factor
    var_ref = 0.7179487179487178 * covariance_factor / cost_factor
    allocation_ref = np.array([[1 * cost_factor, 1, 1, 1, 1, 1],
                               [1 * cost_factor, 0, 0, 1, 0, 1],
                               [1 * cost_factor, 0, 0, 0, 0, 1]], dtype=int)

    target_cost = 10 * cost_factor
    opt_result = optimizer.optimize(algorithm=algorithm,
                                    target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_ref, var_ref, allocation_ref)


@pytest.mark.parametrize("cost_factor", [1, 4])
@pytest.mark.parametrize("covariance_factor", [1, 4])
def test_acvmf_three_models_unordered(cost_factor, covariance_factor, mocker):
    covariance = np.array([[1, 0.25, 0.75],
                           [0.25, 1., 0.5],
                           [0.75, 0.5, 1.]]) * covariance_factor
    model_costs = np.array([3, 1, 2])
    optimizer = Optimizer(model_costs, covariance)

    ratios_for_opt = np.array([3, 2])
    mocker.patch.object(ALGORITHM_MAP['acvmfu'],
                        '_solve_opt_problem',
                        return_value=ratios_for_opt)

    cost_ref = 10. * cost_factor
    var_ref = 0.7179487179487178 * covariance_factor / cost_factor
    allocation_ref = np.array([[1 * cost_factor, 1, 1, 1, 1, 1],
                               [1 * cost_factor, 0, 0, 1, 0, 1],
                               [1 * cost_factor, 0, 0, 1, 0, 0]], dtype=int)

    target_cost = 10 * cost_factor
    opt_result = optimizer.optimize(algorithm="acvmfu", target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_ref, var_ref, allocation_ref)


@pytest.mark.parametrize("seed", [1])
def test_acvmf_optimizer_satisfies_constraints(seed):
    num_models = 3
    rand_matrix = np.random.random((num_models, num_models))
    covariance = np.dot(rand_matrix.T, rand_matrix)
    model_costs = np.array([5, 4, 3])

    target_cost = 100
    optimizer = Optimizer(model_costs, covariance)
    opt_result = optimizer.optimize(algorithm="acvmf",
                                    target_cost=target_cost)

    sample_nums = np.cumsum(opt_result.allocation.compressed_allocation[:, 0])
    assert sample_nums[0] >= 1
    for sample in sample_nums[1:]:
        assert sample > sample_nums[0]
