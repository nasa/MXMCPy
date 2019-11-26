import numpy as np
import pytest

from mxmc.optimizer import Optimizer


def assert_opt_result_equal(opt_result, cost_ref, var_ref, sample_array_ref):
    assert np.isclose(opt_result.cost, cost_ref)
    assert np.isclose(opt_result.variance, var_ref)
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         sample_array_ref)


def test_ordered_kl_enumeration(mocker):
    covariance = np.array([[1, 0.75, 0.25],
                           [0.75, 1., 0.5],
                           [0.25, 0.5, 1.]])
    model_costs = np.array([3, 2, 1])
    optimizer = Optimizer(model_costs, covariance, k="ordered")

    mocker.patch('mxmc.acvkl.ACVKL')

    target_cost = 10
    _ = optimizer.optimize("acvkl", target_cost)