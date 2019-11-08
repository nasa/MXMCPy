import pytest
from pytest_mock import mocker 
import mock
import numpy as np

from mxmc.optimizer import *
from mxmc import acvmf

def assert_opt_result_equal(opt_result, cost_ref, var_ref, sample_array_ref):
    assert np.isclose(opt_result.cost, cost_ref)
    assert np.isclose(opt_result.variance, var_ref)
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         sample_array_ref)

@pytest.mark.parametrize("cost_factor", [1, 4])
@pytest.mark.parametrize("covariance_factor", [1, 4 ])
def test_acvmf_three_models_known_solution(cost_factor,
                                           covariance_factor, mocker):

    covariance = np.array([[1, 0.75, 0.25], 
                           [0.75, 1., 0.5], 
                           [0.25, 0.5, 1.]])*covariance_factor
    model_costs = np.array([3, 2, 1])
    optimizer = Optimizer(model_costs, covariance)

    num_samples_for_opt = np.array([1,1,2])*cost_factor
    mocker.patch.object(ALGORITHM_MAP['acvmf'],
                        '_solve_opt_problem', 
                        return_value=num_samples_for_opt)

    cost_ref = 10.*cost_factor
    var_ref = 0.7179487179487178*covariance_factor/cost_factor
    allocation_ref = np.array([[1*cost_factor, 1, 1, 1, 1, 1],
                               [1*cost_factor, 0, 0, 1, 0, 1], 
                               [1*cost_factor, 0, 0, 0, 0, 1]], dtype=int)

    target_cost = 10*cost_factor
    opt_result = optimizer.optimize(algorithm="acvmf", target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_ref, var_ref, allocation_ref)

@pytest.mark.parametrize("cost_factor", [1,4])
@pytest.mark.parametrize("covariance_factor", [1,4])
def test_acvmf_three_models_unordered(cost_factor, covariance_factor, mocker):

    covariance = np.array([[1, 0.25, 0.75], 
                           [0.25, 1., 0.5], 
                           [0.75, 0.5, 1.]])*covariance_factor
    model_costs = np.array([3, 1, 2])
    optimizer = Optimizer(model_costs, covariance)

    num_samples_for_opt = np.array([1,2,1])*cost_factor
    mocker.patch.object(ALGORITHM_MAP['acvmf'],
                        '_solve_opt_problem', 
                        return_value=num_samples_for_opt)

    cost_ref = 10.*cost_factor
    var_ref = 0.7179487179487178*covariance_factor/cost_factor
    allocation_ref = np.array([[1*cost_factor, 1, 1, 1, 1, 1],
                               [1*cost_factor, 0, 0, 1, 0, 1], 
                               [1*cost_factor, 0, 0, 1, 0, 0]], dtype=int)

    target_cost = 10*cost_factor
    opt_result = optimizer.optimize(algorithm="acvmf", target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_ref, var_ref, allocation_ref)

@pytest.mark.parametrize("seed", [1])
def test_acvmf_optimizer_satisfies_constraints(seed):

    num_models = 3
    rand_matrix = np.random.random((num_models, num_models))
    covariance = np.dot(rand_matrix.T, rand_matrix)
    model_costs = [5, 4, 3]
    
    target_cost = 100
    optimizer = Optimizer(model_costs, covariance)
    opt_result = optimizer.optimize(algorithm="acvmf",
                                    target_cost=target_cost)

    sample_nums = np.cumsum(opt_result.sample_array[:,0])
    assert sample_nums[0] >= 1
    for i, sample in enumerate(sample_nums[1:]):
        assert sample > sample_nums[i]
    
