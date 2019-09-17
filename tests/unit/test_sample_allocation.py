import pytest
import pandas as pd
import numpy as np

from mxmc.SampleAllocation import SampleAllocation


# def test_init():
#    sample_allocation_object = SampleAllocation(np.array([[]]))
#    assert sample_allocation_object.compressed_allocation == [[]]


@pytest.fixture
def compressed_allocation():
    return np.array([[1, 1, 1, 1, 0, 0], [5, 0, 1, 1, 1, 1], [10, 0, 0, 0, 1, 1]])


@pytest.fixture
def expanded_allocation():
    return [[1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1]]


@pytest.fixture
def expanded_allocation_dataframe(expanded_allocation):
    return pd.DataFrame(columns=['0', '1_1', '1_2', '2_1', '2_2'], data=expanded_allocation)


@pytest.fixture
def sample_allocation(compressed_allocation):
    return SampleAllocation(compressed_allocation)


def test_compressed_allocation(sample_allocation, compressed_allocation):
    assert sample_allocation.compressed_allocation == compressed_allocation.tolist()


def test_num_models(sample_allocation):
    assert sample_allocation.num_models == 3


def test_get_column_names(sample_allocation):
    assert sample_allocation._get_column_names() == ['0', '1_1', '1_2', '2_1', '2_2']


def test_one_model_num_models():
    test_allocation = SampleAllocation(np.array([[10, 1]]))
    assert test_allocation.num_models == 1


def test_expand_allocation(sample_allocation, expanded_allocation_dataframe):
    assert pd.DataFrame.equals(sample_allocation.expanded_allocation, expanded_allocation_dataframe)


def test_get_number_of_samples_per_model(sample_allocation):
    assert sample_allocation.get_number_of_samples_per_model() == np.array([1, 6, 15])
