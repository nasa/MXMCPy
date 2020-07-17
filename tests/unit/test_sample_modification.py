import numpy as np
import pytest
import warnings

from mxmc.optimizer import Optimizer
from mxmc.estimator import Estimator
from mxmc.sample_allocation import SampleAllocation
from mxmc.optimizers.mlmc import MLMC
from mxmc.util.sample_modification import maximize_sample_allocation_variance
from mxmc.util.sample_modification import _generate_test_samplings
from mxmc.util.sample_modification import _get_cost_per_sample_by_group


def test_get_cost_per_sample_by_group():

    compressed_allocation = np.array([[10, 1, 1, 1],
                                      [90, 0, 1, 1]])
    model_costs = [10., 1.]
    group_sample_costs = _get_cost_per_sample_by_group(compressed_allocation,
                                                       model_costs)
    expected_group_sample_costs = np.array([11., 1.])

    assert np.array_equal(group_sample_costs, expected_group_sample_costs)


def test_generate_test_samplings_output():

    compressed_allocation = np.array([[10, 1, 1, 1],
                                      [90, 0, 1, 1]])
    covariance = np.identity(2)
    target_cost = 310.
    model_costs = [10., 1.]

    sampling = _generate_test_samplings(compressed_allocation,
                                        model_costs,
                                        target_cost)
    expected_sampling = [(10, 91)]

    for actual, expected in zip(sampling, expected_sampling):
        assert np.array_equal(actual, expected)


def test_maximize_sample_allocation_variance_returns_sample_allocation():

    compressed_allocation = np.array([[9, 1]])
    covariance = np.identity(1)
    target_cost = 10.
    model_costs = [1.]

    base_allocation = SampleAllocation(compressed_allocation)
    adjusted_allocation = maximize_sample_allocation_variance(base_allocation,
                                                              target_cost,
                                                              model_costs,
                                                              covariance)

    assert isinstance(adjusted_allocation, SampleAllocation)


def test_maximize_sample_allocation_variance_increases_samples():

    compressed_allocation = np.array([[10, 1, 1, 1],
                                      [90, 0, 1, 1]])
    covariance = np.identity(2)
    target_cost = 400.
    model_costs = [10., 1.]

    base_allocation = SampleAllocation(compressed_allocation)
    adjusted_allocation = maximize_sample_allocation_variance(base_allocation,
                                                              target_cost,
                                                              model_costs,
                                                              covariance)

    num_base_samples = np.sum(compressed_allocation[:, 0])
    num_adjusted_samples = np.sum(adjusted_allocation.compressed_allocation[:, 0])

    assert num_base_samples < num_adjusted_samples


def test_maximize_sample_allocation_variance_does_not_exceed_target_cost():

    raise NotImplementedError
    compressed_allocation = np.array([[10, 1, 1, 1],
                                      [90, 0, 1, 1]])
    covariance = np.identity(2)
    target_cost = 10.
    model_costs = [10., 1.]

    base_allocation = SampleAllocation(compressed_allocation)
    adjusted_allocation = maximize_sample_allocation_variance(base_allocation,
                                                              target_cost,
                                                              model_costs,
                                                              covariance)

    assert isinstance(adjusted_allocation, SampleAllocation)


def test_maximize_sample_allocation_variance_increases_variance():

    compressed_allocation = np.array([[9, 1]])
    covariance = np.identity(1)
    target_cost = 10.
    model_costs = [1.]

    base_allocation = SampleAllocation(compressed_allocation)
    base_estimate = Estimator(base_allocation, covariance)
    base_variance = base_estimate.approximate_variance

    adjusted_allocation = maximize_sample_allocation_variance(base_allocation,
                                                              target_cost,
                                                              model_costs,
                                                              covariance)

    adjusted_estimate = Estimator(adjusted_allocation, covariance)
    adjusted_variance = adjusted_estimate.approximate_variance

    assert adjusted_variance > base_variance
