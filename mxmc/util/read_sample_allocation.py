import h5py
import numpy as np


def read_sample_allocation(filename):
    '''
    Read sample allocation from file

    :param filename: name of hdf5 sample allocation file
    :type filename: string

    :Returns: a SampleAllocation object
    '''
    allocation_file = h5py.File(filename, 'r')
    compressed_key = 'Compressed_Allocation/compressed_allocation'
    compressed_allocation = np.array(allocation_file[compressed_key])
    method = allocation_file.attrs['Method']

    return SampleAllocation(compressed_allocation, method)
