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

@pytest.fixture
def compressed_allocation_3models():
    return np.array([[1, 1, 0, 0],
                     [2, 0, 1, 0],
                     [0, 0, 0, 1],
                     [2, 1, 1, 0],
                     [1, 1, 0, 1],
                     [0, 0, 1, 1],
                     [0, 1, 1, 1]])

@pytest.fixture
def sample_allocation_3models(compressed_allocation_3models):
    warnings.filterwarnings("ignore",
                            message="Allocation Warning",
                            category=UserWarning)
    return MBLUESampleAllocation(compressed_allocation_3models)


def test_approximate_variance_two_models(sample_allocation_2models):

    covariance = np.array([[1., 1/2.], [1/2., 1.]])
    estimator = Estimator(sample_allocation_2models, covariance)

    assert estimator.approximate_variance == pytest.approx(5 / 11)
    
def test_get_estimate_two_models(sample_allocation_2models):
    
    covariance = np.array([[1., 1/2.], [1/2., 1.]])
    model_outputs = [np.ones(2)*2., np.ones(3)]

    estimator = Estimator(sample_allocation_2models, covariance)

    assert estimator.get_estimate(model_outputs) == pytest.approx(2.)


def test_approximate_variance_three_models(sample_allocation_3models):
    '''
    Psi inverse = array([[0.23333333, 0.06666667, 0.05833333],
       [0.06666667, 0.23333333, 0.01666667],
       [0.05833333, 0.01666667, 0.95208333]])  - variance is [0,0] entry
    '''

    covariance = np.array([[1., 1/2., 1/4.], 
                           [1/2., 1., 1/2.],
                           [1/4., 1/2., 1.]])
    estimator = Estimator(sample_allocation_3models, covariance)
    expected_variance = 0.23333333

    assert estimator.approximate_variance == pytest.approx(expected_variance)
    

def test_get_estimate_three_models(sample_allocation_3models):
    

    covariance = np.array([[1., 1/2., 1/4.], 
                           [1/2., 1., 1/2.],
                           [1/4., 1/2., 1.]])
    model_outputs = [np.ones(4)*2., np.ones(4), np.ones(1)*2.]
    estimator = Estimator(sample_allocation_3models, covariance)

    assert estimator.get_estimate(model_outputs) == pytest.approx(2.)

