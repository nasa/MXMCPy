import numpy as np
import pytest
import warnings

from mxmc.optimizer import Optimizer
from mxmc.estimator import Estimator
from mxmc.sample_allocation import SampleAllocation
from mxmc.optimizers.mlmc import MLMC
from mxmc.util.sample_modification import maximize_sample_allocation_variance


def test_maximize_sample_allocation_variance_returns_sample_allocation():

    compressed_allocation = np.array([[9, 1]])
    covariance = np.identity(1)
    target_cost = 10
    model_costs = [1]

    base_allocation = SampleAllocation(compressed_allocation)
    adjusted_allocation = maximize_sample_allocation_variance(base_allocation,
                                                              target_cost,
                                                              model_costs,
                                                              covariance)

    assert isinstance(adjusted_allocation, SampleAllocation)


def test_maximize_sample_allocation_variance_increases_variance():

    compressed_allocation = np.array([[9, 1]])
    covariance = np.identity(1)
    target_cost = 10
    model_costs = [1]

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
