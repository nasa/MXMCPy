import numpy as np
import pytest

from mxmc.optimizers.approximate_control_variates.generalized_multifidelity \
    import impl_optimizers
from mxmc.optimizer import Optimizer
from mxmc.optimizers.optimizer_base import OptimizationResult


@pytest.mark.parametrize("num_models, num_combinations", [(2, 1),
                                                          (3, 3),
                                                          (4, 10),
                                                          (5, 29),
                                                          (6, 76)])
def test_sr_enumeration(mocker, num_models, num_combinations):
    covariance = np.random.random((num_models, num_models))
    covariance *= covariance.transpose()
    model_costs = np.arange(num_models, 0, -1)
    optimizer = Optimizer(model_costs, covariance)

    mocked_optimizer = mocker.Mock()
    mocked_optimizer.optimize.return_value = OptimizationResult(10, 0.1, None)
    mocker.patch('mxmc.optimizers.approximate_control_variates.'
                 'generalized_multifidelity.impl_optimizers.GMFUnordered',
                 return_value=mocked_optimizer)

    target_cost = 100
    _ = optimizer.optimize("gmfsr", target_cost)

    assert impl_optimizers.GMFUnordered.call_count == num_combinations


@pytest.mark.parametrize("num_models, num_combinations", [(2, 1),
                                                          (3, 3),
                                                          (4, 16),
                                                          (5, 125),
                                                          (6, 1296)])
def test_mr_enumeration(mocker, num_models, num_combinations):
    covariance = np.random.random((num_models, num_models))
    covariance *= covariance.transpose()
    model_costs = np.arange(num_models, 0, -1)
    optimizer = Optimizer(model_costs, covariance)

    mocked_optimizer = mocker.Mock()
    mocked_optimizer.optimize.return_value = OptimizationResult(10, 0.1, None)
    mocker.patch('mxmc.optimizers.approximate_control_variates.'
                 'generalized_multifidelity.impl_optimizers.GMFUnordered',
                 return_value=mocked_optimizer)

    target_cost = 100
    _ = optimizer.optimize("gmfmr", target_cost)

    assert impl_optimizers.GMFUnordered.call_count == num_combinations

