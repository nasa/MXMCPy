import h5py
import numpy as np

from mxmc.sample_allocations.sample_allocation_base import SampleAllocationBase
from mxmc.sample_allocations.acv_sample_allocation import ACVSampleAllocation
from mxmc.sample_allocations.mlmc_sample_allocation import MLMCSampleAllocation


ALLOC_MAP = {SampleAllocationBase.__module__: SampleAllocationBase,
             ACVSampleAllocation.__module__: ACVSampleAllocation,
             MLMCSampleAllocation.__module__: MLMCSampleAllocation}


def read_sample_allocation(filename):
    '''
    Read sample allocation from file

    :param filename: name of hdf5 sample allocation file
    :type filename: string

    :Returns: appropriate child of the SampleAllocationBase class based on the
        optimization method stored in the hdf5 file.
    '''
    allocation_file = h5py.File(filename, 'r')
    compressed_key = 'Compressed_Allocation/compressed_allocation'
    compressed_allocation = np.array(allocation_file[compressed_key])
    method = allocation_file.attrs.get('Method')

    return ALLOC_MAP[method](compressed_allocation)
