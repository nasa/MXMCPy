import numpy as np
import pytest

from mxmc.estimator import Estimator
from mxmc.sample_allocations.mlmc_sample_allocation import MLMCSampleAllocation
from mxmc.util.sample_modification import adjust_sample_allocation_to_cost
from mxmc.util.sample_modification import _generate_test_samplings
from mxmc.util.sample_modification import _get_cost_per_sample_by_group
from mxmc.util.sample_modification import _get_total_sampling_cost


# The one model allocation and cost has a total cost of 10.
@pytest.fixture
def one_model_compressed_allocation():

    return np.array([[10, 1]])


@pytest.fixture
def one_model_cost():
    return [1.]


@pytest.fixture
def one_model_sample_allocation(one_model_compressed_allocation):

    return MLMCSampleAllocation(one_model_compressed_allocation)


# The two model allocation and model costs have a total cost of 200.
@pytest.fixture
def two_model_compressed_allocation():

    return np.array([[10, 1, 1, 1],
                     [90, 0, 1, 1]])


@pytest.fixture
def two_model_costs():
    return [10., 1.]


@pytest.fixture
def two_model_sample_allocation(two_model_compressed_allocation):

    return MLMCSampleAllocation(two_model_compressed_allocation)


def test_returns_sample_allocation(one_model_compressed_allocation,
                                   one_model_cost):

    covariance = np.identity(1)
    target_cost = 11.

    base_allocation = MLMCSampleAllocation(one_model_compressed_allocation)
    adjusted_allocation = adjust_sample_allocation_to_cost(base_allocation,
                                                           target_cost,
                                                           one_model_cost,
                                                           covariance)

    assert isinstance(adjusted_allocation, MLMCSampleAllocation)


def test_increases_samples(two_model_compressed_allocation,
                           two_model_costs):

    covariance = np.array([[1., 0.3],
                           [0.3, 1.]])
    target_cost = 215

    base_allocation = MLMCSampleAllocation(two_model_compressed_allocation)
    adjusted_allocation = adjust_sample_allocation_to_cost(base_allocation,
                                                           target_cost,
                                                           two_model_costs,
                                                           covariance)
    adjusted_sampling = adjusted_allocation.compressed_allocation[:, 0]

    num_base_samples = np.sum(two_model_compressed_allocation[:, 0])
    num_adjusted_samples = np.sum(adjusted_sampling)

    assert num_base_samples < num_adjusted_samples


def test_does_not_exceed_target_cost(two_model_compressed_allocation,
                                     two_model_costs):

    covariance = np.array([[1., 0.3],
                           [0.3, 1.]])
    target_cost = 215

    base_allocation = MLMCSampleAllocation(two_model_compressed_allocation)
    adjusted_allocation = adjust_sample_allocation_to_cost(base_allocation,
                                                           target_cost,
                                                           two_model_costs,
                                                           covariance)
    adjusted_cost = \
        _get_total_sampling_cost(adjusted_allocation.compressed_allocation,
                                 two_model_costs)

    assert adjusted_cost <= target_cost


def test_decreases_variance(two_model_compressed_allocation,
                            two_model_costs):

    # This allocation and model costs have a total cost of 200.
    covariance = np.array([[1., 0.3],
                           [0.3, 1.]])
    target_cost = 215

    base_allocation = MLMCSampleAllocation(two_model_compressed_allocation)
    base_estimate = Estimator(base_allocation, covariance)
    base_variance = base_estimate._get_approximate_variance()
    adjusted_allocation = adjust_sample_allocation_to_cost(base_allocation,
                                                           target_cost,
                                                           two_model_costs,
                                                           covariance)

    adjusted_estimate = Estimator(adjusted_allocation, covariance)
    adjusted_variance = adjusted_estimate._get_approximate_variance()

    assert adjusted_variance < base_variance


@pytest.mark.parametrize("initial_num_samples", [1, 2, 3, 4, 5])
def test_gen_test_samplings_output_monte_carlo(initial_num_samples):

    model_costs = np.array([1.])
    target_cost = 4

    compressed_allocation = np.array([[initial_num_samples, 1]])

    actual_sampling = _generate_test_samplings(compressed_allocation,
                                               model_costs,
                                               target_cost)
    expected_sampling = set()
    if initial_num_samples < target_cost:
        expected_sampling.add((4,))

    assert len(actual_sampling) == len(expected_sampling)
    for actual, expected in zip(actual_sampling, expected_sampling):
        assert actual == expected


def test_gen_test_samplings_output(two_model_compressed_allocation,
                                   two_model_costs):
    target_cost = 201.

    sampling = _generate_test_samplings(two_model_compressed_allocation,
                                        two_model_costs,
                                        target_cost)
    expected_sampling = [(10, 91)]

    assert len(sampling) > 0
    for actual, expected in zip(sampling, expected_sampling):
        assert np.array_equal(actual, expected)


def test_gen_test_samplings_meet_target_cost(two_model_compressed_allocation,
                                             two_model_costs):

    target_cost = 220.
    samplings = _generate_test_samplings(two_model_compressed_allocation,
                                         two_model_costs,
                                         target_cost)

    assert len(samplings) > 0
    for sampling in samplings:

        altered_allocation = np.copy(two_model_compressed_allocation)
        altered_allocation[:, 0] = sampling
        cost = _get_total_sampling_cost(altered_allocation, two_model_costs)

        assert cost <= target_cost


def test_gen_test_samplings_recursion_error_catch(one_model_compressed_allocation,
                                                  one_model_cost,
                                                  capsys):

    target_cost = 5000.
    with pytest.raises(RecursionError):
        _generate_test_samplings(one_model_compressed_allocation,
                                 one_model_cost,
                                 target_cost)

    # Ensure we print a useful message to the user when this happens.
    stdout = capsys.readouterr().out
    assert "Maximum recursion depth exceeded" in stdout


def test_get_cost_per_sample_by_group(two_model_compressed_allocation,
                                      two_model_costs):

    group_sample_costs = \
        _get_cost_per_sample_by_group(two_model_compressed_allocation,
                                      two_model_costs)
    expected_group_sample_costs = np.array([11., 1.])

    assert np.array_equal(group_sample_costs, expected_group_sample_costs)


@pytest.mark.parametrize("initial_num_samples", [1, 2, 3, 4])
def test_result_monte_carlo(initial_num_samples):

    covariance = np.array([[4.]])
    model_costs = np.array([1.])
    target_cost = 4

    base_allocation = MLMCSampleAllocation(np.array([[initial_num_samples, 1]]))
    adjusted_allocation = adjust_sample_allocation_to_cost(base_allocation,
                                                           target_cost,
                                                           model_costs,
                                                           covariance)
    N = int(target_cost / model_costs[0])
    compressed_allocation_expected = np.array([[N, 1]])

    assert np.array_equal(adjusted_allocation.compressed_allocation,
                          compressed_allocation_expected)


def test_result_mocked_generate_test_samplings(mocker):

    DUMMY = 1.0
    base_allocation_compressed = np.array([[1, 1]])
    target_cost = DUMMY
    model_costs = np.array([DUMMY])
    covariance = np.array([[DUMMY]])

    # Forcing algorithm to return # samples = 4 as lowest variance.
    mock_test_samplings = [(2,), (3,), (4,)]
    mock_variances = [5., 4., 3., 2.]

    mocker.patch("mxmc.util.sample_modification._generate_test_samplings",
                 return_value=mock_test_samplings)
    mocker.patch("mxmc.util.sample_modification._get_estimator_variance",
                 side_effect=mock_variances)

    base_allocation = MLMCSampleAllocation(base_allocation_compressed)
    adjusted_allocation = adjust_sample_allocation_to_cost(base_allocation,
                                                           target_cost,
                                                           model_costs,
                                                           covariance)

    compressed_allocation_expected = np.array([[4, 1]])
    assert np.array_equal(adjusted_allocation.compressed_allocation,
                          compressed_allocation_expected)
