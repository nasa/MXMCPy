import numpy as np
import pytest

from mxmc.SampleAllocation import SampleAllocation


@pytest.fixture
def compressed_allocation():
    return np.array([[1, 1, 1, 1, 0, 0],
                     [5, 0, 1, 1, 1, 1],
                     [10, 0, 0, 0, 1, 1]])


@pytest.fixture
def sample_allocation(compressed_allocation):
    return SampleAllocation(compressed_allocation, 'MFMC')

