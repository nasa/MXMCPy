import pytest
import numpy as np

from mxmc.sample_allocations.mblue_sample_allocation import MBLUESampleAllocation


@pytest.mark.parametrize("M", range(4))
def test_num_models_for_M_models(M):
    DUMMY = 5
    allocation = MBLUESampleAllocation(np.array([[DUMMY] + M*[1]]))
    assert allocation.num_models == M


@pytest.mark.parametrize("N", range(4))
def test_num_total_samples_with_one_model(N):
    allocation = MBLUESampleAllocation(np.array([[N, 1]]))
    assert allocation.num_total_samples == N


@pytest.mark.parametrize("N", range(4))
def test_get_sample_indices_for_model_with_one_model(N):
    allocation = MBLUESampleAllocation(np.array([[N, 1]]))
    np.testing.assert_array_equal(allocation.get_sample_indices_for_model(0),
                                  np.arange(N))

