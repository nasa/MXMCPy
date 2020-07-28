import numpy as np
import pytest

from mxmc.estimator import Estimator
from mxmc.sample_allocation import SampleAllocation
from mxmc.util.sample_modification import maximize_sample_allocation_variance
from mxmc.util.sample_modification import _generate_test_samplings
from mxmc.util.sample_modification import _get_cost_per_sample_by_group
from mxmc.util.sample_modification import _get_total_sampling_cost


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
    covariance = np.array([[.5, 0.],
                           [0., .8]])
    target_cost = 500.
    model_costs = [10., 1.]

    base_allocation = SampleAllocation(compressed_allocation)
    adjusted_allocation = maximize_sample_allocation_variance(base_allocation,
                                                              target_cost,
                                                              model_costs,
                                                              covariance)

    num_base_samples = np.sum(compressed_allocation[:, 0])
    num_adjusted_samples = \
        np.sum(adjusted_allocation.compressed_allocation[:, 0])

    assert num_base_samples < num_adjusted_samples


def test_maximize_sample_allocation_variance_does_not_exceed_target_cost():

    compressed_allocation = np.array([[10, 1, 1, 1],
                                      [90, 0, 1, 1]])
    covariance = np.identity(2)
    target_cost = 500.
    model_costs = [10., 1.]

    base_allocation = SampleAllocation(compressed_allocation)
    adjusted_allocation = maximize_sample_allocation_variance(base_allocation,
                                                              target_cost,
                                                              model_costs,
                                                              covariance)
    adjusted_cost = \
        _get_total_sampling_cost(adjusted_allocation.compressed_allocation,
                                 model_costs)

    assert adjusted_cost <= target_cost


def test_maximize_sample_allocation_variance_decreases_variance():

    compressed_allocation = np.array([[10, 1, 1, 1],
                                      [90, 0, 1, 1]])
    covariance = np.identity(2)
    target_cost = 500.
    model_costs = [10., 1.]

    base_allocation = SampleAllocation(compressed_allocation)
    base_estimate = Estimator(base_allocation, covariance)
    base_variance = base_estimate.approximate_variance
    adjusted_allocation = maximize_sample_allocation_variance(base_allocation,
                                                              target_cost,
                                                              model_costs,
                                                              covariance)

    adjusted_estimate = Estimator(adjusted_allocation, covariance)
    adjusted_variance = adjusted_estimate.approximate_variance

    assert adjusted_variance < base_variance


def test_generate_test_samplings_expected_output():

    compressed_allocation = np.array([[10, 1, 1, 1],
                                      [90, 0, 1, 1]])
    target_cost = 310.
    model_costs = [10., 1.]

    sampling = _generate_test_samplings(compressed_allocation,
                                        model_costs,
                                        target_cost)
    expected_sampling = [(10, 91)]

    for actual, expected in zip(sampling, expected_sampling):
        assert np.array_equal(actual, expected)


def test_get_cost_per_sample_by_group():

    compressed_allocation = np.array([[10, 1, 1, 1],
                                      [90, 0, 1, 1]])
    model_costs = [10., 1.]
    group_sample_costs = _get_cost_per_sample_by_group(compressed_allocation,
                                                       model_costs)
    expected_group_sample_costs = np.array([11., 1.])

    assert np.array_equal(group_sample_costs, expected_group_sample_costs)

@pytest.mark.parametrize("initial_num_samples", [1, 2, 3, 4]) 
def test_maximize_sample_allocation_for_monte_carlo(initial_num_samples):

    covariance = np.array([[4.]])
    model_costs = np.array([1.])
    target_cost = 4

    base_allocation = SampleAllocation(np.array([[initial_num_samples, 1]]))
    adjusted_allocation = maximize_sample_allocation_variance(base_allocation,
                                                              target_cost,
                                                              model_costs,
                                                              covariance)
    N = int(target_cost / model_costs[0])
    compressed_allocation_expected = np.array([[N, 1]])

    assert np.array_equal(adjusted_allocation.compressed_allocation, 
                          compressed_allocation_expected)
    


def test_maximize_sample_allocation_mocked_generate_test_samplings(mocker):

    DUMMY = 1.0    
    base_allocation_compressed = np.array([[1, 1]])
    target_cost = DUMMY
    model_costs = np.array([DUMMY])
    covariance = np.array([[DUMMY]])

    #forcing algorithm to return # samples = 4 as lowest variance
    mock_test_samplings = [(2,), (3,), (4,)]
    mock_variances = [5 , 4., 3., 2.]

    mocker.patch("mxmc.util.sample_modification._generate_test_samplings",
                 return_value=mock_test_samplings)
    mocker.patch("mxmc.util.sample_modification._get_estimator_variance",
                 side_effect=mock_variances)

    base_allocation = SampleAllocation(base_allocation_compressed)
    adjusted_allocation = maximize_sample_allocation_variance(base_allocation,
                                                              target_cost,
                                                              model_costs,
                                                              covariance)

    compressed_allocation_expected = np.array([[4, 1]])
    assert np.array_equal(adjusted_allocation.compressed_allocation,
                          compressed_allocation_expected)

