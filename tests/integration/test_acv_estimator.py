import numpy as np
import pytest

from mxmc.estimator import Estimator
from mxmc.sample_allocations.acv_sample_allocation import ACVSampleAllocation


def test_two_model_estimate():
    compressed_allocation = np.array([[1, 1, 1, 1],
                                      [5, 1, 1, 0],
                                      [10, 0, 0, 1]])
    allocation = ACVSampleAllocation(compressed_allocation)
    model_outputs = [np.arange(1, 7), np.arange(1, 17)]
    covariance = np.array([[1, 0.5], [0.5, 1]])

    est = Estimator(allocation, covariance)

    expected_estimate = 5.848484848484849
    assert est.get_estimate(model_outputs) == pytest.approx(expected_estimate)


def test_two_model_approximate_variance():
    compressed_allocation = np.array([[3, 1, 1, 1],
                                      [57, 0, 0, 1]], dtype=int)
    allocation = ACVSampleAllocation(compressed_allocation)
    covariance = np.array([[1, 0.5], [0.5, 1]])

    est = Estimator(allocation, covariance)

    assert est.approximate_variance == pytest.approx(61 / 240)


@pytest.mark.filterwarnings("ignore:Allocation Warning")
def test_three_model_approximate_variance():
    compressed_allocation = np.array([[1, 1, 1, 1, 0, 0],
                                      [5, 0, 1, 1, 1, 1],
                                      [10, 0, 0, 0, 1, 1]])
    sample_allocation = ACVSampleAllocation(compressed_allocation)
    covariance = np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])
    est = Estimator(sample_allocation, covariance)

    assert est.approximate_variance == pytest.approx(1)
