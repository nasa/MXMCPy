import numpy as np

from mxmc.estimator import Estimator
from mxmc.sample_allocation import SampleAllocation


def maximize_sample_allocation_variance(sample_allocation, target_cost,
                                        model_costs, covariance):
    '''
    Tests all possible increases to sample counts per group and returns
    sample allocation with the lowest variance while limiting
    total cost to be within the specified target_cost.
    '''
    def _test_sampling(sampling):

        test = np.copy(base_compressed_allocation)
        test[:, 0] = sampling

        allocation = SampleAllocation(test, sample_allocation.method)
        variance = _get_estimator_variance(allocation, covariance)

        return variance, allocation

    base_compressed_allocation = sample_allocation.compressed_allocation
    sampling_tests = _generate_test_samplings(base_compressed_allocation,
                                              model_costs,
                                              target_cost)

    best_allocation = sample_allocation
    best_variance = _get_estimator_variance(sample_allocation, covariance)
    for sampling_test in sampling_tests:

        test_variance, test_allocation = _test_sampling(sampling_test)
        if test_variance < best_variance:

            best_variance = test_variance
            best_allocation = test_allocation

    return best_allocation


def _get_estimator_variance(sample_allocation, covariance):

    estimator = Estimator(sample_allocation, covariance)
    return estimator._get_approximate_variance()


# Get cost of running all samples as specified by a compressed allocation.
def _get_total_sampling_cost(compressed_allocation, model_costs):

    cost_per_sample_by_group = \
        _get_cost_per_sample_by_group(compressed_allocation,
                                      model_costs)

    samples_per_group = compressed_allocation[:, 0]
    return np.dot(cost_per_sample_by_group, samples_per_group)


# Get cost of a single sample for each group in a compressed allocation.
def _get_cost_per_sample_by_group(compressed_allocation, model_costs):

    num_groups = compressed_allocation.shape[0]
    group_costs_per_sample = np.zeros(num_groups)

    for group_index in range(num_groups):

        group_cost = compressed_allocation[group_index, 1] * model_costs[0]
        for model_index in range(1, len(model_costs)):

            col_1 = model_index * 2
            col_2 = col_1 + 1
            col_1_num_samples = compressed_allocation[group_index, col_1]
            col_2_num_samples = compressed_allocation[group_index, col_2]
            model_is_run = col_1_num_samples > 0 or col_2_num_samples > 0

            if model_is_run:
                group_cost += model_costs[model_index]

        group_costs_per_sample[group_index] = group_cost

    return group_costs_per_sample


# Produces an set of tuples, each of which is a unique sampling.
def _generate_test_samplings(compressed_allocation, model_costs, target_cost):

    # Recursively add samplings to set sampling_tests until we've exhausted
    # all possibilities within target_cost.
    def add_test_samplings(test_sampling, cost_remaining):

        for g, group_cost in enumerate(sample_cost_by_group):

            if 0 < group_cost <= cost_remaining:

                new_sampling = np.copy(test_sampling)
                new_sampling[g] += 1
                new_cost_remaining = cost_remaining - group_cost
                sampling_tests.add(tuple(new_sampling))

                if new_cost_remaining > 0.:
                    add_test_samplings(new_sampling, new_cost_remaining)

    sample_cost_by_group = \
        _get_cost_per_sample_by_group(compressed_allocation, model_costs)

    sampling_tests = set()
    cost_margin = \
        target_cost - _get_total_sampling_cost(compressed_allocation,
                                               model_costs)

    if cost_margin > 0:
        add_test_samplings(compressed_allocation[:, 0], cost_margin)

    return sampling_tests
