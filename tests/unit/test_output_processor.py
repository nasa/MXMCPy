import pytest
import numpy as np
from mxmc.output_processor import OutputProcessor


def test_compute_covariance_matrix_returns_nan_array_given_no_outputs():
    covariance = OutputProcessor.compute_covariance_matrix([])
    assert all(np.isnan(covariance))


def test_compute_covariance_matrix_one_number_is_nan_array():
    covariance = OutputProcessor.compute_covariance_matrix([np.array(1)])
    assert all(np.isnan(covariance))


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
