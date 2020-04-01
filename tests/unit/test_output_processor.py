import numpy as np

from mxmc.output_processor import OutputProcessor


class SampleAllocationStub:

    def __init__(self, model_indices):
        self.model_indices = model_indices

    def get_sample_indices_for_model(self, model):
        return self.model_indices[model]


def test_compute_cov_matrix_return_empty_array_if_no_outputs():
    covariance = OutputProcessor.compute_covariance_matrix([])
    assert covariance.size == 0


def test_compute_covariance_matrix_one_number_is_nan_array():
    covariance = OutputProcessor.compute_covariance_matrix([np.array(1)])
    assert np.isnan(covariance)


def test_compute_covariance_matrix_two_samples():
    model_outputs = [np.array([1, 2])]
    covariance = OutputProcessor.compute_covariance_matrix(model_outputs)
    assert covariance == np.array([0.5])


def test_compute_covariance_matrix_two_model_zero_variance():
    model_outputs = [np.array([1, 1]), np.array([1, 1])]
    covariance = OutputProcessor.compute_covariance_matrix(model_outputs)
    np.testing.assert_array_equal(covariance, np.zeros((2, 2)))


def test_compute_covariance_matrix_two_model_different_sizes():
    model_outputs = [np.array([1, 1]), np.array([1, 1, 1]), np.array([1, 1])]
    covariance = OutputProcessor.compute_covariance_matrix(model_outputs)
    np.testing.assert_array_equal(covariance, np.zeros((3, 3)))


def test_compute_covariance_matrix_with_sample_allocation_1():
    model_indices = {0: [1, 2, 4], 1: [2, 3, 4], 2: [0, 1, 2, 3]}
    sample_alloc = SampleAllocationStub(model_indices)
    model_outputs = [np.array([1, 2, 2.5]), np.array([1., 2., 3.]),
                     np.array([1., -0.5, 2., 1.5])]

    covariance = OutputProcessor.compute_covariance_matrix(model_outputs,
                                                           sample_alloc)

    expected = np.array(
            [[7 / 12., .5, 1.25], [.5, 1., -.25], [1.25, -.25, 7 / 6.]])
    np.testing.assert_array_almost_equal(covariance, expected)


def test_compute_covariance_matrix_with_sample_allocation_2():
    model_indices = {0: [0, 1], 1: [2, 3, 4]}
    sample_alloc = SampleAllocationStub(model_indices)
    model_outputs = [np.array([1, 2]), np.array([1., 2., 3.])]

    covariance = OutputProcessor.compute_covariance_matrix(model_outputs,
                                                           sample_alloc)

    expected = np.array([[0.5, np.nan], [np.nan, 1.]])
    np.testing.assert_array_almost_equal(covariance, expected)


def test_compute_covariance_matrix_with_no_overlap():
    model_outputs = [np.array([1, 2]), np.array([1.])]

    covariance = OutputProcessor.compute_covariance_matrix(model_outputs)

    expected = np.array([[0.5, np.nan], [np.nan] * 2])
    np.testing.assert_array_almost_equal(covariance, expected)
