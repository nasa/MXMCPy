import numpy as np
import pytest
import warnings

from mxmc.estimator import Estimator
from mxmc.sample_allocations.mblue_sample_allocation import MBLUESampleAllocation


@pytest.fixture
def compressed_allocation_2models():
    return np.array([[1, 1, 0],
                     [2, 0, 1],
                     [1, 1, 1]])

@pytest.fixture
def sample_allocation_2models(compressed_allocation_2models):
    warnings.filterwarnings("ignore",
                            message="Allocation Warning",
                            category=UserWarning)
    return MBLUESampleAllocation(compressed_allocation_2models)


def test_approximate_variance_two_models(sample_allocation_2models):

    covariance = np.array([[1., 1/2.], [1/2., 1.]])
    estimator = Estimator(sample_allocation_2models, covariance)

    assert estimator.approximate_variance == pytest.approx(5 / 11)
    
def test_get_estimate_two_models(sample_allocatin_2models):
    
    covariance = np.array([[1., 1/2.], [1/2., 1.]])
    model_outputs = [np.ones(2)*2., np.ones(3)]

    estimator = Estimator(sample_allocation_2models, covariance)

    assert estimator.get_estimate(model_outputs) == pytest.approx(2.)
