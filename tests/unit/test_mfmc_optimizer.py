import pytest
import numpy as np

from mxmc.mfmc import MFMC


@pytest.fixture
def mfmc_optimizer():
    covariance = np.array([[1, 0.9], [0.9, 1]])
    model_costs = np.array([1, 2])
    return MFMC(covariance, model_costs)


def test_optimizer_returns_tuple_with(mfmc_optimizer):
    opt_result = mfmc_optimizer.optimize(target_cost=10)
    for member in ["cost", "variance", "sample_array"]:
        assert member in dir(opt_result)


def test_non_matching_covariance_and_costs_lengths():
    covariance = np.random.random((3, 3))
    model_costs = np.array([1, 1])
    with pytest.raises(ValueError):
        MFMC(covariance, model_costs)


def test_covariance_is_symmetric():
    covariance = np.array([[0, 2], [1, 0]])
    model_costs = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        MFMC(covariance, model_costs)


@pytest.mark.parametrize("target_cost", [-1., 0, 2.5])
def test_optimize_with_small_target_cost(mfmc_optimizer, target_cost):
    opt_result = mfmc_optimizer.optimize(target_cost)
    assert opt_result.cost == 0
    assert np.isinf(opt_result.variance)
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         np.array([[0, 1, 1, 1]], dtype=int))


def test_optimize_with_hifi_fastest():
    covariance = np.array([[1, 0.0], [0.0, 1]])
    model_costs = np.array([1, 2])
    mfmc = MFMC(covariance, model_costs)
    opt_result = mfmc.optimize(target_cost=30)
    assert opt_result.cost == 30
    assert opt_result.variance == pytest.approx(0.1)
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         np.array([[10, 1, 1, 1],
                                                   [0, 1, 1, 1]], dtype=int))


def test_case_mfmc():
    covariance = np.array([[1, 0.5], [0.5, 1]])
    model_costs = np.array([48, 4])
    mfmc = MFMC(covariance, model_costs)
    opt_result = mfmc.optimize(target_cost=168)

    assert opt_result.cost == pytest.approx(168)
    assert opt_result.variance == pytest.approx(7/24)
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         np.array([[3, 1, 1, 1],
                                                   [3, 0, 0, 1]], dtype=int))


@pytest.mark.parametrize("num_models", range(1, 4))
def test_opt_results_are_correct_sizes(num_models):
    covariance = np.eye(num_models)
    model_costs = np.ones(num_models)
    mfmc = MFMC(covariance, model_costs)
    opt_result = mfmc.optimize(10)
    assert opt_result.sample_array.shape[1] == num_models*2
