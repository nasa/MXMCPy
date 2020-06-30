import numpy as np

from mxmc.estimator import Estimator


def maximize_sample_allocation_variance(sample_allocation, target_cost,
                                        model_costs, covariance):

    def cost_margin_available(allocation):

        return True

    def get_allocation_cost(allocation):

        return np.dot(allocation[:, 0], model_costs)

    def get_variance(allocation):

        estimator = Estimator(allocation, covariance)
        return estimator._get_approximate_variance()

    def reduce_variance(allocation):

        starting_var = get_variance(allocation)
        return False

    def find_potential_allocations(allocation):

        return [allocation]

    best_allocation = sample_allocation
    best_variance = get_variance(sample_allocation)

    test_allocations = find_potential_allocations(best_allocation)

    for test_allocation in test_allocations:

        test_variance = get_variance(test_allocation)
        if test_variance > best_variance:

            best_variance = test_variance
            best_allocation = test_allocation

    return best_allocation
