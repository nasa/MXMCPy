import pytest
import numpy as np

from mxmc.mfmc import MFMC


@pytest.fixture
def mfmc_optimizer():
    covariance = np.array([[1, 0.9], [0.9, 1]])
    model_costs = np.array([1, 2])
    return MFMC(model_costs, covariance)


def test_optimizer_returns_tuple_with(mfmc_optimizer):
    opt_result = mfmc_optimizer.optimize(target_cost=10)
    for member in ["cost", "variance", "sample_array"]:
        assert member in dir(opt_result)


def test_non_matching_covariance_and_costs_lengths():
    covariance = np.random.random((3, 3))
    model_costs = np.array([1, 1])
    with pytest.raises(ValueError):
        MFMC(model_costs, covariance)


def test_covariance_is_symmetric():
    covariance = np.array([[0, 2], [1, 0]])
    model_costs = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        MFMC(model_costs, covariance)


@pytest.mark.parametrize("target_cost", [-1., 0, 0.5])
def test_optimize_with_small_target_cost(mfmc_optimizer, target_cost):
    opt_result = mfmc_optimizer.optimize(target_cost)
    assert opt_result.cost == 0
    assert np.isinf(opt_result.variance)
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         np.array([[0, 1, 1, 1]], dtype=int))


# TODO: What do we want to happen here?
def test_optimize_with_hifi_fastest():
    covariance = np.array([[1, 0.0], [0.0, 1]])
    model_costs = np.array([1, 2])
    mfmc = MFMC(model_costs, covariance)
    opt_result = mfmc.optimize(target_cost=30)
    assert opt_result.cost == 30
    assert opt_result.variance == pytest.approx(1/30)
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         np.array([[30, 1, 0, 0]], dtype=int))


@pytest.mark.parametrize("target_cost_multiplier", [1, 4])
@pytest.mark.parametrize("covariance_multiplier", [1, 4])
def test_case_mfmc(target_cost_multiplier, covariance_multiplier):
    covariance = np.array([[1, 0.5], [0.5, 1]])*covariance_multiplier
    model_costs = np.array([4800, 4])
    mfmc = MFMC(model_costs, covariance)
    target_cost = 14640 * target_cost_multiplier
    opt_result = mfmc.optimize(target_cost)

    print(opt_result.sample_array)
    assert opt_result.cost == pytest.approx(14640 * target_cost_multiplier)
    assert opt_result.variance == pytest.approx(61/240 * covariance_multiplier
                                                / target_cost_multiplier)
    np.testing.assert_array_almost_equal(
            opt_result.sample_array,
            np.array([[3 * target_cost_multiplier, 1, 1, 1],
                      [57 * target_cost_multiplier, 0, 0, 1]], dtype=int))


@pytest.mark.parametrize("num_models", range(1, 4))
def test_opt_results_are_correct_sizes(num_models):
    covariance = np.eye(num_models)
    model_costs = np.ones(num_models)
    mfmc = MFMC(model_costs, covariance)
    opt_result = mfmc.optimize(10)
    assert opt_result.sample_array.shape[1] == num_models*2


@pytest.mark.parametrize("num_models", range(1, 4))
def test_mfmc_opt_results_are_correct_sizes(num_models):
    covariance = np.eye(num_models)
    covariance[0] = np.linspace(1.0, 0.6, num_models)
    covariance[:, 0] = np.linspace(1.0, 0.6, num_models)
    model_costs = np.arange(num_models, 0, -1)
    mfmc = MFMC(model_costs, covariance)
    opt_result = mfmc.optimize(10)
    assert opt_result.sample_array.shape[1] == num_models*2


def test_mfmc_can_initialize_with_extra_inputs():
    covariance = np.array([[1, 0.5], [0.5, 1]])
    model_costs = np.array([4800, 4])
    _ = MFMC(model_costs, covariance, 0, abc=0)
