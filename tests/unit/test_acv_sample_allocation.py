import os.path
import warnings

import pytest
import numpy as np

from mxmc.sample_allocations.acv_sample_allocation import ACVSampleAllocation
from mxmc.util.read_sample_allocation import read_sample_allocation



@pytest.fixture
def compressed_allocation():
    return np.array([[1, 1, 1, 1, 0, 0],
                     [5, 0, 1, 1, 1, 1],
                     [10, 0, 0, 0, 1, 1]])


@pytest.fixture
def sample_allocation(compressed_allocation):
    warnings.filterwarnings("ignore",
                            message="Allocation Warning",
                            category=UserWarning)
    return ACVSampleAllocation(compressed_allocation)


def test_get_column_names(sample_allocation):
    assert sample_allocation._get_column_names() == ['0', '1_1', '1_2', '2_1',
                                                     '2_2']
