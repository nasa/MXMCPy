import numpy as np
import pytest

from mxmc.optimizer import Optimizer, ALGORITHM_MAP
from mxmc.util.testing import assert_opt_result_equal


@pytest.mark.parametrize("cost_factor", [1, 4])
@pytest.mark.parametrize("covariance_factor", [1, 4])
def test_acvmfmc_known_solution(cost_factor, covariance_factor, mocker):
    covariance = np.array([[1, 0.5], [0.5, 1]]) * covariance_factor
    model_costs = np.array([4800, 4])
    optimizer = Optimizer(model_costs, covariance)

    ratios_for_opt = np.array([20])
    mocker.patch.object(ALGORITHM_MAP["acvmfmc"],
                        '_solve_opt_problem',
                        return_value=ratios_for_opt)

    cost_ref = 14640 * cost_factor
    var_ref = 61 / 240 * covariance_factor / cost_factor
    allocation_ref = np.array([[3 * cost_factor, 1, 1, 1],
                               [57 * cost_factor, 0, 0, 1]], dtype=int)

    target_cost = 14640 * cost_factor
    opt_result = optimizer.optimize(algorithm="acvmfmc",
                                    target_cost=target_cost)
    assert_opt_result_equal(opt_result, cost_ref, var_ref, allocation_ref)
