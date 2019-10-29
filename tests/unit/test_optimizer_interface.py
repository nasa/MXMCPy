import numpy as np
import pytest

from mxmc.optimizer import Optimizer

ALGORITHMS = ["mfmc", "mlmc"]
DUMMY_VAR = 999


def assert_opt_result_equal(opt_result, cost_ref, var_ref, sample_array_ref):
    assert np.isclose(opt_result.cost, cost_ref)
    assert np.isclose(opt_result.variance, var_ref)
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         sample_array_ref)


@pytest.fixture
def ui_optimizer():
    model_costs = np.array([100, 1])
    covariance = np.array([[1, 0.9], [0.9, 1]])
    vardiff_matrix = np.array([[DUMMY_VAR, 1], [1, 4]])
    return Optimizer(model_costs, covariance=covariance,
                     vardiff_matrix=vardiff_matrix)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("target_cost", [-1, 0.5, 0])
def test_target_cost_too_low_to_run_models(ui_optimizer, algorithm,
                                           target_cost):

    opt_result = ui_optimizer.optimize(algorithm=algorithm,
                                       target_cost=target_cost)
    assert_opt_result_equal(opt_result, 0, np.inf, np.array([[0, 1, 1, 1]]))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_optimizer_returns_tuple_with(ui_optimizer, algorithm):
    opt_result = ui_optimizer.optimize(algorithm="mfmc", target_cost=10)
    for member in ["cost", "variance", "sample_array"]:
        assert member in dir(opt_result)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("mismatched_input", ["costs", "covariance",
                                              "vardiff"])
def test_mismatched_inputs(algorithm, mismatched_input):
    covariance = np.array([[1, 0.9], [0.9, 1]])
    model_costs = np.array([1, 2])
    vardiff_matrix = np.array([[DUMMY_VAR, 1], [1, 2]])
    if mismatched_input == "covariance":
        covariance = np.array([[1, 0.9, 0.8], [0.9, 1, 0.9], [0.8, 0.9, 1]])
    elif mismatched_input == "vardiff":
        vardiff_matrix = np.array([[DUMMY_VAR, 1, DUMMY_VAR],
                                   [1, DUMMY_VAR, 1],
                                   [DUMMY_VAR, 1, 2]])
    elif mismatched_input == "costs":
        model_costs = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        optimizer = Optimizer(model_costs, covariance, vardiff_matrix)
        _ = optimizer.optimize(algorithm=algorithm, target_cost=30)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("asymmetric_input", ["covariance",  "vardiff"])
def test_input_asymmetry(algorithm, asymmetric_input):
    covariance = np.array([[0, 2], [1, 0]])
    model_costs = np.array([1, 1, 1])
    vardiff_matrix = np.array([[DUMMY_VAR, 1], [1, 2]])
    if asymmetric_input == "covariance":
        covariance[0, 1] -= 1
    elif asymmetric_input == "vardiff":
        vardiff_matrix[0, 1] -= 1

    with pytest.raises(ValueError):
        optimizer = Optimizer(model_costs, covariance, vardiff_matrix)
        _ = optimizer.optimize(algorithm=algorithm, target_cost=30)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("num_models", range(1, 4))
def test_optimize_results_are_correct_sizes(algorithm, num_models):
    covariance = np.eye(num_models)
    covariance[0] = np.linspace(1.0, 0.6, num_models)
    covariance[:, 0] = np.linspace(1.0, 0.6, num_models)
    model_costs = np.arange(num_models, 0, -1)
    vardiff_matrix = np.ones([num_models, num_models])

    optimizer = Optimizer(model_costs, covariance, vardiff_matrix)
    opt_result = optimizer.optimize(algorithm=algorithm, target_cost=10)

    if algorithm in ALGORITHMS:
        assert opt_result.sample_array.shape[0] == num_models
    assert opt_result.sample_array.shape[1] == num_models*2


def test_optimizer_can_initialize_with_extra_inputs():
    covariance = np.array([[1, 0.5], [0.5, 1]])
    model_costs = np.array([4800, 4])
    vardiff_matrix = np.ones([2, 2])
    _ = Optimizer(model_costs, covariance, vardiff_matrix, 0, abc=0)





# MODEL SELECTION.............................


@pytest.mark.parametrize("num_models", range(1, 4))
def test_opt_results_are_correct_sizes(num_models):
    covariance = np.eye(num_models)
    model_costs = np.ones(num_models)
    optimizer = Optimizer(model_costs, covariance)
    opt_result = optimizer.optimize(algorithm="mfmc", target_cost=10,
                                    auto_model_selection=True)
    assert opt_result.sample_array.shape[1] == num_models*2