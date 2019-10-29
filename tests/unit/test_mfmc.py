import pytest
import numpy as np

from mxmc.optimizer import Optimizer


@pytest.mark.parametrize("target_cost_multiplier", [1, 4])
@pytest.mark.parametrize("covariance_multiplier", [1, 4])
def test_case_mfmc(target_cost_multiplier, covariance_multiplier):
    covariance = np.array([[1, 0.5], [0.5, 1]])*covariance_multiplier
    model_costs = np.array([4800, 4])
    optimizer = Optimizer(model_costs, covariance)
    target_cost = 14640 * target_cost_multiplier
    opt_result = optimizer.optimize(algorithm="mfmc",
                                    target_cost=target_cost)

    assert opt_result.cost == pytest.approx(14640 * target_cost_multiplier)
    assert opt_result.variance == pytest.approx(61/240 * covariance_multiplier
                                                / target_cost_multiplier)
    np.testing.assert_array_almost_equal(
            opt_result.sample_array,
            np.array([[3 * target_cost_multiplier, 1, 1, 1],
                      [57 * target_cost_multiplier, 0, 0, 1]], dtype=int))


def test_mfmc_with_model_selection_hifi_fastest():
    covariance = np.array([[1, 0.0], [0.0, 1]])
    model_costs = np.array([1, 2])
    optimizer = Optimizer(model_costs, covariance)
    opt_result = optimizer.optimize(algorithm="mfmc", target_cost=30,
                                    auto_model_selection=True)
    assert opt_result.cost == 30
    assert opt_result.variance == pytest.approx(1/30)
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         np.array([[30, 1, 0, 0]], dtype=int))