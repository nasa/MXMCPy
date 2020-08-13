import numpy as np
import pytest

from mxmc.optimizer import Optimizer
from mxmc.util.testing import assert_opt_result_equal

ALGORITHMS = Optimizer.get_algorithm_names()
DUMMY_VAR = 999


@pytest.fixture
def ui_optimizer():
    model_costs = np.array([100, 1])
    covariance = np.array([[1, 0.9], [0.9, 1]])
    return Optimizer(model_costs, covariance=covariance)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("target_cost", [-1, 0.5, 0])
def test_target_cost_too_low_to_run_models(ui_optimizer, algorithm,
                                           target_cost):
    opt_result = ui_optimizer.optimize(algorithm=algorithm,
                                       target_cost=target_cost)
    assert_opt_result_equal(opt_result, 0, np.inf, np.array([[1, 1, 0, 0]]))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_optimizer_returns_tuple_with(ui_optimizer, algorithm):
    opt_result = ui_optimizer.optimize(algorithm="mfmc", target_cost=10)
    for member in ["cost", "variance", "allocation"]:
        assert member in dir(opt_result)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_mismatched_cost_and_variance_raises_error(algorithm):
    covariance = np.array([[1, 0.9], [0.9, 1]])
    model_costs = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        optimizer = Optimizer(model_costs, covariance)
        _ = optimizer.optimize(algorithm=algorithm, target_cost=30)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_input_asymmetry(algorithm):
    covariance = np.array([[0, 2], [1, 0]])
    model_costs = np.array([1, 1, 1])
    covariance[0, 1] -= 1

    with pytest.raises(ValueError):
        optimizer = Optimizer(model_costs, covariance)
        _ = optimizer.optimize(algorithm=algorithm, target_cost=30)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("num_models", range(1, 4))
def test_optimize_results_are_correct_sizes(algorithm, num_models):
    covariance = np.eye(num_models)
    covariance[0] = np.linspace(1.0, 0.6, num_models)
    covariance[:, 0] = np.linspace(1.0, 0.6, num_models)
    model_costs = np.arange(num_models, 0, -1)

    optimizer = Optimizer(model_costs, covariance)
    opt_result = optimizer.optimize(algorithm=algorithm, target_cost=20)

    opt_sample_array = opt_result.allocation.compressed_allocation
    if algorithm in ["mfmc", "mlmc", "acvis"]:
        assert opt_sample_array.shape[0] == num_models
    assert opt_sample_array.shape[1] == num_models * 2


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("num_models", range(1, 4))
def test_opt_results_are_correct_sizes_using_model_selection(num_models,
                                                             algorithm):
    covariance = np.eye(num_models)
    covariance[0] = np.linspace(1.0, 0.6, num_models)
    covariance[:, 0] = np.linspace(1.0, 0.6, num_models)
    model_costs = np.arange(num_models, 0, -1)
    optimizer = Optimizer(model_costs, covariance)
    opt_result = optimizer.optimize(algorithm=algorithm, target_cost=20,
                                    auto_model_selection=True)
    opt_sample_array = opt_result.allocation.compressed_allocation
    assert opt_sample_array.shape[1] == num_models * 2


def test_optimizer_can_initialize_with_extra_inputs():
    covariance = np.array([[1, 0.5], [0.5, 1]])
    model_costs = np.array([4800, 4])
    _ = Optimizer(model_costs, covariance, 0, abc=0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_optimizer_returns_monte_carlo_result_for_one_model(algorithm):
    covariance = np.array([[12.]])
    model_costs = np.array([2.])
    target_cost = 8.

    optimizer = Optimizer(model_costs, covariance)

    opt_result = optimizer.optimize(algorithm=algorithm,
                                    target_cost=target_cost)

    N_ref = target_cost / model_costs[0]
    variance_ref = covariance[0] / N_ref
    cost_ref = target_cost

    opt_sample_array = opt_result.allocation.compressed_allocation
    assert np.isclose(variance_ref, opt_result.variance)
    assert np.isclose(cost_ref, opt_result.cost)
    assert N_ref == opt_sample_array[0, 0]
