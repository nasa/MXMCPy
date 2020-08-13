import numpy as np
import pytest

from mxmc.optimizer import Optimizer
from mxmc.util.testing import assert_opt_result_equal
from mxmc.sample_allocations.acv_sample_allocation import ACVSampleAllocation


@pytest.mark.parametrize("target_cost_multiplier", [1, 4])
@pytest.mark.parametrize("covariance_multiplier", [1, 4])
def test_case_mfmc(target_cost_multiplier, covariance_multiplier):
    covariance = np.array([[1, 0.5], [0.5, 1]]) * covariance_multiplier
    model_costs = np.array([4800, 4])
    optimizer = Optimizer(model_costs, covariance)
    target_cost = 14640 * target_cost_multiplier
    opt_result = optimizer.optimize(algorithm="mfmc",
                                    target_cost=target_cost)

    expected_cost = 14640 * target_cost_multiplier
    expected_variance = 61 / 240 * covariance_multiplier \
        / target_cost_multiplier
    expected_sample_array = np.array([[3 * target_cost_multiplier, 1, 1, 1],
                                      [57 * target_cost_multiplier, 0, 0, 1]],
                                     dtype=int)
    assert_opt_result_equal(opt_result, expected_cost, expected_variance,
                            expected_sample_array)


def test_mfmc_with_model_selection_hifi_fastest():
    covariance = np.array([[1, 0.0], [0.0, 1]])
    model_costs = np.array([1, 2])
    optimizer = Optimizer(model_costs, covariance)
    opt_result = optimizer.optimize(algorithm="mfmc", target_cost=30,
                                    auto_model_selection=True)

    expected_cost = 30
    expected_variance = 1 / 30
    expected_sample_array = np.array([[30, 1, 0, 0]], dtype=int)

    assert isinstance(opt_result.allocation, ACVSampleAllocation)
    assert_opt_result_equal(opt_result, expected_cost, expected_variance,
                            expected_sample_array)


def test_mfmc_with_model_selection_no_best_result(mocker):
    mocker.patch('mxmc.optimizer.AutoModelSelection.' +
                 '_get_subsets_of_model_indices', return_value=[])

    covariance = np.array([[1, 0.0], [0.0, 1]])
    model_costs = np.array([1, 2])
    optimizer = Optimizer(model_costs, covariance)
    opt_result = optimizer.optimize(algorithm="mfmc", target_cost=30,
                                    auto_model_selection=True)
    expected_cost = 0
    expected_variance = np.inf
    expected_sample_array = np.array([[1, 1, 0, 0]], dtype=int)
    assert_opt_result_equal(opt_result, expected_cost, expected_variance,
                            expected_sample_array)
