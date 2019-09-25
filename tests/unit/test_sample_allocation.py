import pytest
import pandas as pd
import numpy as np
from mxmc.SampleAllocation import SampleAllocation


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
    assert np.array_equal(sample_allocation.get_number_of_samples_per_model(), np.array([1, 6, 15]))


def test_get_sample_indices_for_model_0(sample_allocation):
    assert sample_allocation.get_sample_indices_for_model(0) == [0]


def test_get_sample_indices_for_model_1(sample_allocation):
    assert sample_allocation.get_sample_indices_for_model(1) == [0, 1, 2, 3, 4, 5]


def test_get_sample_indices_for_model_2(sample_allocation):
    assert sample_allocation.get_sample_indices_for_model(2) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def test_get_total_number_of_samples(sample_allocation):
    assert sample_allocation.get_total_number_of_samples() == 16
