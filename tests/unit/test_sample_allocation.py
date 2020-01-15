import os.path
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

from mxmc.input_generator import InputGenerator
from mxmc.sample_allocation import SampleAllocation


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
    return pd.DataFrame(columns=['0', '1_1', '1_2', '2_1', '2_2'],
                        data=expanded_allocation)


def test_error_raised_if_no_method_specified(compressed_allocation):
    with pytest.raises(ValueError):
        SampleAllocation(compressed_allocation)


def test_compressed_allocation(sample_allocation, compressed_allocation):
    assert np.array_equal(sample_allocation.compressed_allocation,
                          compressed_allocation)


def test_num_models(sample_allocation):
    assert sample_allocation.num_models == 3


def test_get_column_names(sample_allocation):
    assert sample_allocation._get_column_names() == ['0', '1_1', '1_2', '2_1',
                                                     '2_2']


def test_one_model_num_models():
    test_allocation = SampleAllocation(np.array([[10, 1]]), 'MFMC')
    assert test_allocation.num_models == 1


def test_expand_allocation(sample_allocation, expanded_allocation_dataframe):
    assert pd.DataFrame.equals(sample_allocation.expanded_allocation,
                               expanded_allocation_dataframe)


def test_get_number_of_samples_per_model(sample_allocation):
    assert np.array_equal(sample_allocation.get_number_of_samples_per_model(),
                          np.array([1, 6, 15]))


def test_get_sample_indices_for_model_0(sample_allocation):
    assert sample_allocation.get_sample_indices_for_model(0) == [0]


def test_get_sample_indices_for_model_1(sample_allocation):
    assert sample_allocation.get_sample_indices_for_model(1) == [0, 1, 2, 3, 4,
                                                                 5]


def test_get_sample_indices_for_model_2(sample_allocation):
    assert sample_allocation.get_sample_indices_for_model(2) == [1, 2, 3, 4, 5,
                                                                 6, 7, 8, 9,
                                                                 10, 11, 12,
                                                                 13, 14, 15]


def test_get_total_number_of_samples(sample_allocation):
    assert sample_allocation.num_total_samples == 16


@pytest.fixture
def input_names():
    return ['input1', 'input2', 'input3']


@pytest.fixture
def input_array():
    return np.array([[1, 2, 3],
                     [2, 3, 4],
                     [3, 4, 5],
                     [6.4, 3, 7],
                     [2.4, 34.5, 54],
                     [12, 3, 4],
                     [2, 13, 4],
                     [2, 38, 14],
                     [2.4, 83, 4.1],
                     [28, 3.4, 4],
                     [24, 3, 4],
                     [2.8, 43, 4],
                     [2, 3.7, 44],
                     [42, 37, 74],
                     [72, 4.3, 4],
                     [27, 3, 9.4]])


@pytest.fixture
def input_generator_with_array(input_names, input_array):
    return InputGenerator(input_names, input_array)


@pytest.fixture
def input_dataframe(input_names, input_array):
    return pd.DataFrame(columns=input_names, data=input_array)


def test_input_generator_init(input_dataframe, input_generator_with_array):
    assert pd.DataFrame.equals(input_dataframe,
                               input_generator_with_array.input_dataframe)


def test_input_generator_generate_samples(input_generator_with_array,
                                          input_dataframe):
    assert pd.DataFrame.equals(input_dataframe,
                               input_generator_with_array.generate_samples(16))


def test_input_generator_generate_subset_of_samples(input_generator_with_array,
                                                    input_dataframe):
    assert pd.DataFrame.equals(input_dataframe[:5],
                               input_generator_with_array.generate_samples(5))


def test_input_generator_generate_too_many_samples_requested(
        input_generator_with_array,
        input_dataframe):
    with pytest.raises(ValueError):
        input_generator_with_array.generate_samples(17)


def test_sample_allocation_generate_samples(input_generator_with_array,
                                            sample_allocation,
                                            input_dataframe):
    sample_allocation.generate_samples(input_generator_with_array)
    assert pd.DataFrame.equals(input_dataframe, sample_allocation.samples)


def test_sample_allocation_get_samples_for_model_0(input_generator_with_array,
                                                   sample_allocation,
                                                   input_dataframe):
    sample_allocation.generate_samples(input_generator_with_array)
    assert pd.DataFrame.equals(input_dataframe.iloc[[0], :],
                               sample_allocation.get_samples_for_model(0))


def test_sample_allocation_get_samples_for_model_1(input_generator_with_array,
                                                   sample_allocation,
                                                   input_dataframe):
    sample_allocation.generate_samples(input_generator_with_array)
    assert pd.DataFrame.equals(input_dataframe.iloc[[0, 1, 2, 3, 4, 5], :],
                               sample_allocation.get_samples_for_model(1))


def test_sample_allocation_get_samples_for_model_2(input_generator_with_array,
                                                   sample_allocation,
                                                   input_dataframe):
    sample_allocation.generate_samples(input_generator_with_array)
    assert pd.DataFrame.equals(input_dataframe.iloc[
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                15], :],
                               sample_allocation.get_samples_for_model(2))


def test_h5_file_exists(sample_allocation):
    sample_allocation.save('test_save.hdf5')
    assert os.path.exists('test_save.hdf5')


def test_h5_keys_exist(sample_allocation):
    file = h5py.File('test_save.hdf5', 'r')
    data = list(file.keys())
    file.close()
    assert data == ['Compressed_Allocation',
                    'Expanded_Allocation',
                    'Input_Names',
                    'Samples',
                    'Samples_Model_0',
                    'Samples_Model_1',
                    'Samples_Model_2']


def test_h5_data_with_no_samples(sample_allocation):
    file = h5py.File('test_save.hdf5', 'r')
    data = list(file['Compressed_Allocation']['compressed_allocation'])
    file.close()
    assert np.array_equal(data,
                          sample_allocation.compressed_allocation)


def test_h5_method_attribute(sample_allocation):
    sample_allocation.save('test_save.hdf5')
    file = h5py.File('test_save.hdf5', 'r')
    method = file.attrs['Method']
    file.close()
    assert method == 'MFMC'


def test_sample_initialization_from_file_with_no_samples(input_array):
    warnings.filterwarnings("ignore",
                            message="Allocation Warning",
                            category=UserWarning)
    sample_allocation = SampleAllocation('test_save.hdf5')
    assert np.array_equal(sample_allocation.samples,
                          pd.DataFrame())


def test_h5_data_with_samples(input_generator_with_array,
                              sample_allocation,
                              input_dataframe):
    sample_allocation.generate_samples(input_generator_with_array)
    sample_allocation.save('test_save.hdf5')
    file = h5py.File('test_save.hdf5', 'r')
    data = list(file['Samples_Model_0']['samples_model_0'])
    file.close()
    assert np.array_equal(data,
                          sample_allocation.get_samples_for_model(0))


def test_compressed_allocation_initialization_from_file(compressed_allocation):
    warnings.filterwarnings("ignore",
                            message="Allocation Warning",
                            category=UserWarning)
    sample_allocation = SampleAllocation('test_save.hdf5')
    assert np.array_equal(sample_allocation.compressed_allocation,
                          compressed_allocation)


def test_expanded_allocation_initialization_from_file(expanded_allocation):
    warnings.filterwarnings("ignore",
                            message="Allocation Warning",
                            category=UserWarning)
    sample_allocation = SampleAllocation('test_save.hdf5')
    assert np.array_equal(sample_allocation.expanded_allocation,
                          expanded_allocation)


def test_sample_initialization_from_file(input_array):
    warnings.filterwarnings("ignore",
                            message="Allocation Warning",
                            category=UserWarning)
    sample_allocation = SampleAllocation('test_save.hdf5')
    assert np.array_equal(sample_allocation.samples,
                          input_array)


def test_method_flag_initialization_from_file(input_array):
    warnings.filterwarnings("ignore",
                            message="Allocation Warning",
                            category=UserWarning)
    sample_allocation = SampleAllocation('test_save.hdf5')
    assert sample_allocation.method == 'MFMC'
