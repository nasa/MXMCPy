
import numpy as np
import pytest

from mxmc.mlmc import MLMC
from mxmc.optimizer import Optimizer

dummy_var = 999


def assert_opt_result_equal(opt_result, cost_ref, var_ref, sample_array_ref):
    assert np.isclose(opt_result.cost, cost_ref)
    assert np.isclose(opt_result.variance, var_ref)
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         sample_array_ref)


@pytest.fixture
def optimizer_two_model():
    model_costs = np.array([3, 1])
    vardiff_matrix = np.array([[dummy_var, 1], [1, 4]])
    return Optimizer(model_costs=model_costs, vardiff_matrix=vardiff_matrix)


@pytest.fixture
def optimizer_three_model():
    model_costs = np.array([5, 3, 1])
    vardiff_matrix = np.array([[dummy_var, 0.5, dummy_var],
                               [0.5, dummy_var, 1], [dummy_var, 1, 4]])
    return Optimizer(model_costs=model_costs, vardiff_matrix=vardiff_matrix)


@pytest.fixture
def optimizer_four_model():
    model_costs = np.array([11, 5, 3, 1])
    vardiff_matrix = np.array([[dummy_var, 0.25, dummy_var, dummy_var],
                               [0.25, dummy_var, 0.5, dummy_var],
                               [dummy_var, 0.5, dummy_var, 1],
                               [dummy_var, dummy_var, 1, 4]])
    return Optimizer(model_costs=model_costs, vardiff_matrix=vardiff_matrix)


@pytest.mark.parametrize("target_cost, factor", [(8, 1), (16, 2)])
def test_optimize_works_for_simple_two_model_ex(optimizer_two_model,
                                                target_cost, factor):

    sample_array_expected = np.array([[1*factor, 1, 1, 0],
                                      [4*factor, 0, 0, 1]])
    variance_expected = 2/float(factor)
    cost_expected = target_cost

    opt_result = optimizer_two_model.optimize(algorithm="mlmc",
                                              target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_expected, variance_expected,
                            sample_array_expected)


@pytest.mark.parametrize("target_cost, factor", [(24, 1), (48,2)])
def test_optimize_works_for_simple_three_model_ex(optimizer_three_model,
                                                  target_cost, factor):

    sample_array_expected = np.array([[1*factor, 1, 1, 0, 0, 0],
                                      [2*factor, 0, 0, 1, 1, 0],
                                      [8*factor, 0, 0, 0, 0, 1]])
    var_expected = 1.5/float(factor)
    cost_expected = target_cost

    opt_result = optimizer_three_model.optimize(algorithm="mlmc",
                                                target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_expected, var_expected,
                            sample_array_expected)


@pytest.mark.parametrize("target_cost, factor", [(64, 1), (128, 2)])
def test_optimize_works_for_simple_four_model_ex(optimizer_four_model,
                                                 target_cost, factor):

    sample_array_expected = np.array([[1*factor, 1, 1, 0, 0, 0, 0, 0],
                                      [2*factor, 0, 0, 1, 1, 0, 0, 0],
                                      [4*factor, 0, 0, 0, 0, 1, 1, 0],
                                      [16*factor, 0, 0, 0, 0, 0, 0, 1]])
    var_expected = 1/float(factor)
    cost_expected = target_cost

    opt_result = optimizer_four_model.optimize(algorithm="mlmc",
                                               target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_expected, var_expected,
                            sample_array_expected)


def test_three_models_out_of_order():

    target_cost = 24
    model_costs = np.array([5, 1, 3])
    vardiff_matrix = np.array([[dummy_var, dummy_var, 0.5],
                               [dummy_var, 4, 1],
                               [0.5, 1, dummy_var]])
    optimizer = Optimizer(model_costs=model_costs,
                          vardiff_matrix=vardiff_matrix)

    sample_array_expected = np.array([[1, 1, 0, 0, 1, 0],
                                      [2, 0, 1, 0, 0, 1],
                                      [8, 0, 0, 1, 0, 0]])
    var_expected = 1.5
    cost_expected = target_cost

    opt_result = optimizer.optimize(algorithm="mlmc",
                                    target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_expected, var_expected,
                            sample_array_expected)


def test_four_models_out_of_order():

    target_cost = 64
    model_costs = np.array([11, 1, 5, 3])
    vardiff_matrix = np.array([[dummy_var, dummy_var, 0.25, dummy_var],
                               [dummy_var, 4., dummy_var, 1.],
                               [0.25, dummy_var, dummy_var, 0.5],
                               [dummy_var, 1., 0.5, dummy_var]])
    
    optimizer = Optimizer(model_costs=model_costs,
                          vardiff_matrix=vardiff_matrix)

    sample_array_expected = np.array([[1, 1, 0, 0, 1, 0, 0, 0],
                                      [2, 0, 0, 0, 0, 1, 1, 0],
                                      [4, 0, 1, 0, 0, 0, 0, 1],
                                      [16, 0, 0, 1, 0, 0, 0, 0]])
    var_expected = 1.
    cost_expected = target_cost

    opt_result = optimizer.optimize(algorithm="mlmc",
                                    target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_expected, var_expected,
                            sample_array_expected)


def test_raises_error_if_first_model_is_not_highest_cost():
    '''
    MXMC assumes first model is high fidelity. For MLMC, the high fidelity
    model is finest discretization and therefore is the one with highest cost
    '''

    model_costs = np.array([11, 1, 12])
    vardiff_matrix = np.ones([3, 3])
    
    with pytest.raises(ValueError):
        optimizer = Optimizer(model_costs=model_costs,
                              vardiff_matrix=vardiff_matrix)
        _ = optimizer.optimize(algorithm="mlmc", target_cost=50)


def test_optimize_for_noninteger_sample_nums(optimizer_three_model):

    sample_array_expected = np.array([[1, 1, 1, 0, 0, 0],
                                      [3, 0, 0, 1, 1, 0],
                                      [12, 0, 0, 0, 0, 1]])
    target_cost = 36
    var_expected = (1/2. + 1/3. + 1/3.)
    cost_expected = 32

    opt_result = optimizer_three_model.optimize(algorithm="mlmc",
                                                target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_expected, var_expected,
                            sample_array_expected)


def test_mismatched_cost_and_vardiff_raises_error():
    covariance = np.array([[1, 0.9], [0.9, 1]])
    model_costs = np.array([1, 2])
    vardiff_matrix = np.array([[dummy_var, 1, dummy_var],
                               [1, dummy_var, 1],
                               [dummy_var, 1, 2]])

    with pytest.raises(ValueError):
        optimizer = Optimizer(model_costs, covariance, vardiff_matrix)
        _ = optimizer.optimize(algorithm="mlmc", target_cost=30)


def test_mlmc_with_model_selection():
    model_costs = np.array([3, 2, 1])
    vardiff_matrix = np.array([[8, 2, 1],
                               [2, 6, 3],
                               [1, 3, 4]])
    optimizer = Optimizer(model_costs, vardiff_matrix=vardiff_matrix)

    target_cost = 8

    sample_array_expected = np.array([[1, 1, 0, 0, 1, 0],
                                      [4, 0, 0, 0, 0, 1]])
    variance_expected = 2
    cost_expected = target_cost

    opt_result = optimizer.optimize(algorithm="mlmc", target_cost=target_cost,
                                    auto_model_selection=True)
    print(opt_result)
    assert_opt_result_equal(opt_result, cost_expected, variance_expected,
                            sample_array_expected)