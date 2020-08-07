import h5py
import numpy as np
import pytest

from mxmc.sample_allocations.sample_allocation_base import SampleAllocationBase
from mxmc.sample_allocations.acv_sample_allocation import ACVSampleAllocation
from mxmc.sample_allocations.mlmc_sample_allocation import MLMCSampleAllocation

from mxmc.util.read_sample_allocation import read_sample_allocation

class DummyH5:

    def __init__(self, allocation_module, *args):
        self.attrs = lambda: None
        self.attrs.get = lambda x: allocation_module

    def __getitem__(self, key):
        return np.ones((5, 5))


@pytest.mark.parametrize('method', [SampleAllocationBase, ACVSampleAllocation,
                                    MLMCSampleAllocation])
def test_sample_allocation_read(method, mocker):
    key = 'Compressed_Allocation/compressed_allocation'
    mocker.patch('mxmc.util.read_sample_allocation.h5py.File', new=DummyH5)

    dummy_filename = method.__module__
    loaded_allocation = read_sample_allocation(dummy_filename)

    assert type(loaded_allocation)
