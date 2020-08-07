import numpy as np
import pytest

from mxmc.estimator import Estimator
from mxmc.sample_allocations.mlmc_sample_allocation import MLMCSampleAllocation
from mxmc.sample_allocations.acv_sample_allocation import ACVSampleAllocation


@pytest.fixture(params=[1, 2, 3])
def num_models(request):
    return request.param


@pytest.fixture(params=[10, 100])
def num_samples(request):
    return request.param


@pytest.fixture(params=["mlmc", "acv"])
def mc_sample_allocation(num_models, num_samples, request):
    if request.param == "mlmc":
        compressed_allocation = np.zeros((1, num_models * 2), dtype=int)
        compressed_allocation[0, [0, 1]] = [num_samples, 1]
        return MLMCSampleAllocation(compressed_allocation)

    if request.param == "acv":
        compressed_allocation = np.zeros((1, num_models * 2), dtype=int)
        compressed_allocation[0, [0, 1]] = [num_samples, 1]
        return ACVSampleAllocation(compressed_allocation)

    raise NotImplementedError


def test_estimate_for_monte_carlo(num_models, num_samples,
                                  mc_sample_allocation):
    mc_outputs = [np.empty(0) for _ in range(num_models)]
    mc_outputs[0] = np.random.random(num_samples)

    covariance = np.eye(num_models)
    est = Estimator(mc_sample_allocation, covariance)

    expected_estimate = np.mean(mc_outputs[0])
    assert est.get_estimate(mc_outputs) == pytest.approx(expected_estimate)
