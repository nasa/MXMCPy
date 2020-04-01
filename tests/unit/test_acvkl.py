import numpy as np
import pytest

from mxmc.optimizers.approximate_control_variates.generalized_multifidelity \
    import impl_optimizers
from mxmc.optimizer import Optimizer
from mxmc.optimizers.optimization_result import OptimizationResult


@pytest.mark.parametrize("num_models, num_combinations", [(2, 1),
                                                          (3, 2),
                                                          (4, 4),
                                                          (5, 7),
                                                          (6, 11)])
def test_kl_enumeration(mocker, num_models, num_combinations):
    covariance = np.random.random((num_models, num_models))
    covariance *= covariance.transpose()
    model_costs = np.arange(num_models, 0, -1)
    optimizer = Optimizer(model_costs, covariance)

    mocked_optimizer = mocker.Mock()
    dummy_samples = np.array([[1, 1] + [0]*(num_models*2-2)], dtype=int)
    mocked_optimizer.optimize.return_value = OptimizationResult(10, 0.1,
                                                                dummy_samples)
    mocker.patch('mxmc.optimizers.approximate_control_variates.'
                 'generalized_multifidelity.impl_optimizers.GMFOrdered',
                 return_value=mocked_optimizer)

    target_cost = 100
    _ = optimizer.optimize("acvkl", target_cost)

    assert impl_optimizers.GMFOrdered.call_count == num_combinations


def test_acv_kl_inner_variance_calc(mocker):
    covariance = np.array([[1, 0.75, 0.25],
                           [0.75, 1., 0.5],
                           [0.25, 0.5, 1.]])
    model_costs = np.array([3, 2, 1])
    optimizer = Optimizer(model_costs, covariance, k_models={1}, l_models={1})

    ratios_for_opt = np.array([2, 3])
    mocker.patch('mxmc.optimizers.approximate_control_variates.'
                 'generalized_multifidelity.impl_optimizers.GMFOrdered.'
                 '_solve_opt_problem',
                 return_value=ratios_for_opt)

    dummy_target_cost = 10
    opt_result = optimizer.optimize("acvkl", dummy_target_cost)

    assert np.isclose(opt_result.variance, 204./288)
