import itertools

import numpy as np

from mxmc.estimator import Estimator
from mxmc.sample_allocation import SampleAllocation


def maximize_sample_allocation_variance(sample_allocation, target_cost,
                                        model_costs, covariance):
    '''
    Tests all possible increase sample counts per group and returns
    sample allocation with the highest variance while maintaining
    total cost within the specified target_cost.
    '''
    base_sampling = sample_allocation.compressed_allocation
    best_variance = _get_estimator_variance(sample_allocation, covariance)

    test_sample_counts = _generate_test_samplings(base_sampling,
                                                  model_costs,
                                                  target_cost)

    best_allocation = sample_allocation
    for test_sample_count in test_sample_counts:

        test_sampling = np.copy(base_sampling)
        test_sampling[:, 0] = test_sample_count
        test_allocation = SampleAllocation(test_sampling,
                                           method=sample_allocation.method)
        test_variance = _get_estimator_variance(test_allocation, covariance)

        if test_variance > best_variance:

            best_variance = test_variance
            best_allocation = test_allocation

    return best_allocation


def _get_estimator_variance(sample_allocation, covariance):

    estimator = Estimator(sample_allocation, covariance)
    return estimator._get_approximate_variance()


# Get cost of running all samples as specified by a compressed allocation.
def _get_total_sampling_cost(sampling, model_costs):

    cost_per_sample_by_group = _get_cost_per_sample_by_group(sampling,
                                                             model_costs)
    samples_per_group = sampling[:, 0]
    return np.dot(cost_per_sample_by_group, samples_per_group)


# Get cost of a single sample for each group in a compressed allocation.
def _get_cost_per_sample_by_group(sampling, model_costs):

    num_groups = sampling.shape[0]
    group_costs_per_sample = np.zeros(num_groups)

    for group_index in range(num_groups):
        group_cost = sampling[group_index, 1] * model_costs[0]
        for model_index in range(1, len(model_costs)):
            col_1 = model_index * 2
            col_2 = col_1 + 1
            model_is_run = sampling[group_index, col_1] + \
                           sampling[group_index, col_2] > 0
            if model_is_run:
                group_cost += model_costs[model_index]

        group_costs_per_sample[group_index] = group_cost

    return group_costs_per_sample


# Produces an array of tuples, each of which is a viable sample
# count for each group.
def _generate_test_samplings(sampling, model_costs, target_cost):

    def add_test_samplings(test_sampling, cost_remaining):

        for g, group_cost in enumerate(sample_cost_by_group):

            cost_of_adding_sample = (test_sampling[g, 0] + 1) * group_cost
            if cost_of_adding_sample <= cost_remaining:

                new_sampling = np.copy(test_sampling)
                new_sampling[g, 0] += 1
                new_cost_margin = cost_remaining - cost_of_adding_sample
                sampling_tests.add(tuple(new_sampling[:, 0]))
                add_test_samplings(new_sampling, new_cost_margin)

    sample_cost_by_group = _get_cost_per_sample_by_group(sampling,
                                                         model_costs)

    sampling_tests = set()
    cost_margin = target_cost - _get_total_sampling_cost(sampling,
                                                         model_costs)
    add_test_samplings(sampling, cost_margin)
    return sampling_tests
