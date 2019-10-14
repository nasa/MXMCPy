import pytest
import numpy as np
from mxmc.output_processor import OutputProcessor

class SampleAllocationStub:

    def __init__(self, model_indices):
        self.model_indices = model_indices

    def get_sample_indices_for_model(self, model):
        return self.model_indices[model]


@pytest.fixture
def output_processor():
    return OutputProcessor()


def test_compute_cov_matrix_return_nan_array_if_no_outputs(output_processor):
    covariance = output_processor.compute_covariance_matrix([])
    assert all(np.isnan(covariance))


def test_compute_covariance_matrix_one_number_is_nan_array(output_processor):
    covariance = output_processor.compute_covariance_matrix([np.array(1)])
    assert all(np.isnan(covariance))


def test_compute_covariance_matrix_two_samples(output_processor):
    model_outputs = [np.array([1, 2])]
    covariance = output_processor.compute_covariance_matrix(model_outputs)
    assert covariance == np.array([0.5])


def test_compute_covariance_matrix_two_model_zero_variance(output_processor):
    model_outputs = [np.array([1, 1]), np.array([1, 1])]
    covariance = output_processor.compute_covariance_matrix(model_outputs)
    np.testing.assert_array_equal(covariance, np.zeros((2, 2)))


def test_compute_covariance_matrix_two_model_different_sizes(output_processor):
    model_outputs = [np.array([1, 1]), np.array([1, 1, 1]), np.array([1, 1])]
    covariance = output_processor.compute_covariance_matrix(model_outputs)
    np.testing.assert_array_equal(covariance, np.zeros((3, 3)))


def test_compute_covariance_matrix_with_sample_allocation(output_processor):
    model_indices = {0: [1, 2, 4], 1: [2, 3, 4], 2: [0, 1, 2, 3]}
    sample_alloc = SampleAllocationStub(model_indices)
    model_outputs = [np.array([1, 2, 2.5]), np.array([1., 2., 3.]),
                     np.array([1., -0.5, 2., 1.5])]

    covariance = output_processor.compute_covariance_matrix(model_outputs,
                                                            sample_alloc)

    expected = np.array([[7/12., .5, 1.25], [.5, 1., -.25], [1.25, -.25, 7/6.]])
    np.testing.assert_array_almost_equal(covariance, expected)
