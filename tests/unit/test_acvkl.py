import numpy as np
import pytest

from mxmc import acvkl_enumerator as acvkl_enum_module
from mxmc.acvkl_enumerator import NoMatchingCombosError
from mxmc.optimizer import Optimizer
from mxmc.optimizer_base import OptimizationResult


@pytest.mark.parametrize("k_models, l_models, expected_evals",
                         [({0, 1, 2, 3}, {0, 1, 2, 3}, 3),
                          ({1, 2}, {0, 1, 2}, 3),
                          ({1, 2}, {1, 3}, 2),
                          ({0}, {0, 1, 2}, 0),
                          ({1}, {2}, 0),
                          ])
def test_kl_model_evals_in_enumeration(mocker, k_models, l_models,
                                       expected_evals):
    covariance = np.array([[1, 0.75, 0.25],
                           [0.75, 1., 0.5],
                           [0.25, 0.5, 1.]])
    model_costs = np.array([3, 2, 1])
    optimizer = Optimizer(model_costs, covariance,
                          k_models=k_models, l_models=l_models)

    mocked_optimizer = mocker.Mock()
    mocked_optimizer.optimize.return_value = OptimizationResult(10, 0.1, None)
    mocker.patch('mxmc.acvkl_enumerator.ACVKL')
    acvkl_enum_module.ACVKL.return_value = mocked_optimizer

    target_cost = 10
    if expected_evals > 0:
        _ = optimizer.optimize("acvkl", target_cost)
        assert acvkl_enum_module.ACVKL.call_count == expected_evals

    else:
        with pytest.raises(NoMatchingCombosError):
            _ = optimizer.optimize("acvkl", target_cost)


@pytest.mark.parametrize("num_models, num_combinations", [(2, 1),
                                                          (3, 3),
                                                          (4, 10),
                                                          (5, 29),
                                                          (6, 76)])
def test_full_kl_enumeration(mocker, num_models, num_combinations):
    covariance = np.random.random((num_models, num_models))
    covariance *= covariance.transpose()
    model_costs = np.arange(num_models, 0, -1)
    optimizer = Optimizer(model_costs, covariance)

    mocked_optimizer = mocker.Mock()
    mocked_optimizer.optimize.return_value = OptimizationResult(10, 0.1, None)
    mocker.patch('mxmc.acvkl_enumerator.ACVKL')
    acvkl_enum_module.ACVKL.return_value = mocked_optimizer

    target_cost = 100
    _ = optimizer.optimize("acvkl", target_cost)

    assert acvkl_enum_module.ACVKL.call_count == num_combinations
