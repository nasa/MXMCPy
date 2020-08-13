import numpy as np
import pytest

from mxmc.estimators.estimator_base import EstimatorBase


@pytest.fixture
def sample_allocation_mock(mocker):
    sample_allocation = mocker.Mock()
    sample_allocation.num_models = 3
    sample_allocation.get_number_of_samples_per_model.return_value = [1, 6, 15]
    return sample_allocation


@pytest.fixture
def EstimatorBaseMock(mocker):
    mocker.patch.object(EstimatorBase, "__abstractmethods__", new_callable=set)
    return EstimatorBase


@pytest.fixture
def sample_model_outputs(sample_allocation_mock):
    num_samples = sample_allocation_mock.get_number_of_samples_per_model()
    return [np.random.random(n) for n in num_samples]


def test_error_for_mismatched_num_models(sample_allocation_mock,
                                         EstimatorBaseMock):
    covariance = np.eye(sample_allocation_mock.num_models - 1)
    with pytest.raises(ValueError):
        EstimatorBaseMock(sample_allocation_mock, covariance)


def test_error_on_non_symmetric_covariance(sample_allocation_mock,
                                           EstimatorBaseMock):
    covariance = np.eye(sample_allocation_mock.num_models)
    covariance[0, 1] = 1
    with pytest.raises(ValueError):
        EstimatorBaseMock(sample_allocation_mock, covariance)


def test_allocation_matches_model_outputs_num_models(sample_allocation_mock,
                                                     sample_model_outputs,
                                                     EstimatorBaseMock):
    covariance = np.eye(sample_allocation_mock.num_models)
    est = EstimatorBaseMock(sample_allocation_mock, covariance)
    with pytest.raises(ValueError):
        est._validate_model_outputs(sample_model_outputs[1:])


def test_allocation_matches_model_outputs_per_model(sample_allocation_mock,
                                                    sample_model_outputs,
                                                    EstimatorBaseMock):
    covariance = np.random.random((sample_allocation_mock.num_models,
                                   sample_allocation_mock.num_models))
    covariance += covariance.transpose()
    est = EstimatorBaseMock(sample_allocation_mock, covariance)
    sample_model_outputs[1] = sample_model_outputs[1][1:]
    with pytest.raises(ValueError):
        est._validate_model_outputs(sample_model_outputs)


def test_get_estimate_raises_error_for_multiple_outputs(sample_allocation_mock,
                                                        EstimatorBaseMock):
    covariance = np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])
    est = EstimatorBaseMock(sample_allocation_mock, covariance)
    num_samples = sample_allocation_mock.get_number_of_samples_per_model()
    model_multi_outputs = [np.random.random((n, 2)) for n in num_samples]

    with pytest.raises(ValueError):
        est._validate_model_outputs(model_multi_outputs)
