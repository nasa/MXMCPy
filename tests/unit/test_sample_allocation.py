import pytest
from mxmc.SampleAllocation import SampleAllocation


def test_sample_allocation_init():
    sample_allocation_object = SampleAllocation([])
    assert sample_allocation_object.compressed_allocation == []