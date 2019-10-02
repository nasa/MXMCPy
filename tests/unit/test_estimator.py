import numpy as np
import pytest

from mxmc.SampleAllocation import SampleAllocation
from mxmc.estimator import Estimator


@pytest.fixture
def sample_model_outputs(sample_allocation):
    num_samples = sample_allocation.get_number_of_samples_per_model()
    return [np.random.random(n) for n in num_samples]


def test_error_for_mismatched_num_models(sample_allocation):
    covariance = np.eye(sample_allocation.num_models - 1)
    with pytest.raises(ValueError):
        Estimator(sample_allocation, covariance)


def test_error_on_non_symmetric_covariance(sample_allocation):
    covariance = np.eye(sample_allocation.num_models)
    covariance[0, 1] = 1
    with pytest.raises(ValueError):
        Estimator(sample_allocation, covariance)


def test_allocation_matches_model_outputs_num_models(sample_allocation,
                                                     sample_model_outputs):
    covariance = np.eye(sample_allocation.num_models)
    est = Estimator(sample_allocation, covariance)
    with pytest.raises(ValueError):
        est.get_estimate(sample_model_outputs[1:])


def test_allocation_matches_model_outputs_per_model(sample_allocation,
                                                    sample_model_outputs):
    covariance = np.eye(sample_allocation.num_models)
    est = Estimator(sample_allocation, covariance)
    sample_model_outputs[1] = sample_model_outputs[1][1:]
    with pytest.raises(ValueError):
        est.get_estimate(sample_model_outputs)


@pytest.mark.parametrize("num_models", range(1, 4))
@pytest.mark.parametrize("num_samples", [10, 100])
def test_estimate_for_monte_carlo(num_models, num_samples):
    compressed_allocation = np.zeros((1, num_models*2), dtype=int)
    compressed_allocation[0, 0] = num_samples
    compressed_allocation[0, 1] = 1
    mc_allocation = SampleAllocation(compressed_allocation, "mc")
    mc_outputs = [np.empty(0) for _ in range(num_models)]
    mc_outputs[0] = np.random.random(num_samples)

    covariance = np.eye(num_models)
    est = Estimator(mc_allocation, covariance)

    expected_estimate = np.mean(mc_outputs[0])
    assert est.get_estimate(mc_outputs) == expected_estimate

