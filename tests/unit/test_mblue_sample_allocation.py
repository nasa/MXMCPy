import pytest
import numpy as np

from mxmc.sample_allocations.mblue_sample_allocation import MBLUESampleAllocation


@pytest.fixture
def compressed_allocation_2models():
    return np.array([[1, 1, 0],
                     [5, 0, 1],
                     [10, 1, 1]])

@pytest.fixture
def compressed_allocation_3models():
    return np.array([[7, 1, 0, 0],
                     [6, 0, 1, 0],
                     [5, 0, 0, 1],
                     [4, 1, 1, 0],
                     [3, 1, 0, 1],
                     [2, 0, 1, 1],
                     [1, 1, 1, 1]])

@pytest.fixture
def compressed_allocation_3models_3groups():
        return np.array([[6, 0, 0, 1],
                         [2, 1, 1, 0],
                         [4, 0, 1, 1]])

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


def test_num_total_samples_with_two_models(compressed_allocation_2models):
    allocation = MBLUESampleAllocation(compressed_allocation_2models)
    assert allocation.num_total_samples == 16


@pytest.mark.parametrize("model_index, expected_indices", 
                         [(0, np.concatenate((np.array([0]), np.arange(6,16)))),
                          (1, np.arange(1,16))])
def test_get_sample_indices_for_model_with_two_models(model_index,
                                                 expected_indices,
                                                 compressed_allocation_2models):

    allocation = MBLUESampleAllocation(compressed_allocation_2models)
    computed_indices = allocation.get_sample_indices_for_model(model_index)
    np.testing.assert_array_equal(computed_indices, expected_indices)


def test_get_number_of_samples_per_model_with_two_models(
                                                compressed_allocation_2models):
    allocation = MBLUESampleAllocation(compressed_allocation_2models)
    assert np.array_equal(allocation.get_number_of_samples_per_model(),
                          np.array([11, 15]))

def test_allocate_samples_to_models_with_two_models(
                                                compressed_allocation_2models):

    input_array = np.random.random((16, 3))

    inputs_0 = input_array[np.concatenate((np.array([0]), np.arange(6,16))), :]
    inputs_1 = input_array[np.arange(1,16), :]
    ref_samples = [inputs_0, inputs_1]

    allocation = MBLUESampleAllocation(compressed_allocation_2models)
    gen_samples = allocation.allocate_samples_to_models(input_array)

    for i, gen_samples_i in enumerate(gen_samples):
        assert np.array_equal(ref_samples[i], gen_samples_i)


@pytest.mark.parametrize("model_index, expected_indices", 
    [(0, np.concatenate((np.arange(7), np.arange(18,25), np.arange(27,28)))),
     (1, np.concatenate((np.arange(7,13), np.arange(18,22), np.arange(25,28)))),
     (2, np.concatenate((np.arange(13,18), np.arange(22,28))))])
def test_get_sample_indices_for_model_with_three_models(model_index,
                                                 expected_indices,
                                                 compressed_allocation_3models):

    allocation = MBLUESampleAllocation(compressed_allocation_3models)
    computed_indices = allocation.get_sample_indices_for_model(model_index)
    np.testing.assert_array_equal(computed_indices, expected_indices)


def test_get_number_of_samples_per_model_with_three_models(
                                                compressed_allocation_3models):
    allocation = MBLUESampleAllocation(compressed_allocation_3models)
    assert np.array_equal(allocation.get_number_of_samples_per_model(),
                          np.array([15, 13, 11]))

@pytest.mark.parametrize("model_index, expected_indices",
    [(0, np.array([6,7])),
     (1, np.arange(6,12)),
     (2, np.concatenate((np.arange(0, 6), np.arange(8, 12))))])
def test_get_sample_indices_for_model_with_three_models_three_groups(
                                         model_index,
                                         expected_indices,
                                         compressed_allocation_3models_3groups):

    allocation = MBLUESampleAllocation(compressed_allocation_3models_3groups)
    computed_indices = allocation.get_sample_indices_for_model(model_index)
    np.testing.assert_array_equal(computed_indices, expected_indices)

def test_get_number_of_samples_per_model_with_three_models_three_groups(
                                    compressed_allocation_3models_3groups):
    allocation = MBLUESampleAllocation(compressed_allocation_3models_3groups)
    assert np.array_equal(allocation.get_number_of_samples_per_model(),
                          np.array([2, 6, 10]))


