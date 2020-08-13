import numpy as np
import pytest

from mxmc import Estimator
from mxmc.estimators.acv_estimator import ACVEstimator
from mxmc.estimators.mlmc_estimator import MLMCEstimator
from mxmc.sample_allocations.acv_sample_allocation import ACVSampleAllocation
from mxmc.sample_allocations.mlmc_sample_allocation import MLMCSampleAllocation


@pytest.mark.parametrize("allocation_type, estimator_type",
                         [(MLMCSampleAllocation, MLMCEstimator),
                          (ACVSampleAllocation, ACVEstimator)])
def test_factory_returns_mlmc_estimator_for_mlmc_allocation(allocation_type,
                                                            estimator_type):

    compressed_allocation = np.ones((2, 2))
    allocation = allocation_type(compressed_allocation)
    covariance = np.ones((1, 1))
    estimator = Estimator(allocation, covariance)
    assert isinstance(estimator, estimator_type)
