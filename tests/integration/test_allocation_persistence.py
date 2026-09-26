"""Exercise allocation persistence with actual HDF5 files."""
import h5py
import numpy as np
import pytest

from mxmc import Estimator
from mxmc.sample_allocations.sample_allocation_base import SampleAllocationBase
from mxmc.sample_allocations.acv_sample_allocation import ACVSampleAllocation
from mxmc.sample_allocations.mlmc_sample_allocation import MLMCSampleAllocation
from mxmc.util.read_sample_allocation import read_sample_allocation


ALLOCATION_TYPES = [SampleAllocationBase, ACVSampleAllocation,
                    MLMCSampleAllocation]
DATASET_PATH = "Compressed_Allocation/compressed_allocation"
ALLOCATIONS = [
    np.array([[6, 1]], dtype=np.int32),
    np.array([[3, 1, 1, 1], [7, 0, 0, 1]], dtype=np.int64),
    np.array([[4, 1, 1, 1, 0, 0], [3, 0, 0, 1, 0, 0]], dtype=np.int32),
    np.array([[0, 1, 1, 1], [5, 1, 1, 0]], dtype=np.int64),
]


@pytest.mark.parametrize("allocation_type", ALLOCATION_TYPES)
@pytest.mark.parametrize("compressed", ALLOCATIONS)
def test_save_round_trip(tmp_path, allocation_type, compressed):
    allocation = allocation_type(compressed)
    filename = tmp_path / "allocation.h5"
    assert allocation.save(filename) is None

    with h5py.File(filename, "r") as saved:
        assert list(saved) == ["Compressed_Allocation"]
        assert list(saved["Compressed_Allocation"]) == [
            "compressed_allocation"]
        np.testing.assert_array_equal(saved[DATASET_PATH][...], compressed)
        assert saved[DATASET_PATH].dtype == compressed.dtype
        assert saved.attrs["Method"] == allocation_type.__module__

    restored = read_sample_allocation(filename)
    assert type(restored) is allocation_type
    np.testing.assert_array_equal(restored.compressed_allocation, compressed)
    assert restored.compressed_allocation.dtype == compressed.dtype
    assert restored.num_models == allocation.num_models
    assert restored.num_total_samples == allocation.num_total_samples
    np.testing.assert_array_equal(
        restored.get_number_of_samples_per_model(),
        allocation.get_number_of_samples_per_model())
    samples = np.arange(allocation.num_total_samples * 2).reshape(-1, 2)
    for original, loaded in zip(allocation.allocate_samples_to_models(samples),
                                restored.allocate_samples_to_models(samples)):
        np.testing.assert_array_equal(loaded, original)

    # Loading materializes an independent array, not a live HDF5 view.
    restored.compressed_allocation[0, 0] += 1
    with h5py.File(filename, "r") as saved:
        np.testing.assert_array_equal(saved[DATASET_PATH][...], compressed)


@pytest.mark.parametrize("allocation_type", ALLOCATION_TYPES)
@pytest.mark.parametrize("dataset_path",
                         [DATASET_PATH, "compressed_allocation"])
def test_read_existing_layouts(tmp_path, allocation_type, dataset_path):
    filename = tmp_path / "existing.h5"
    compressed = ALLOCATIONS[1]
    with h5py.File(filename, "w") as saved:
        saved.create_group("Compressed_Allocation")
        saved.create_dataset(dataset_path, data=compressed)
        saved.attrs["Method"] = allocation_type.__module__
    original_bytes = filename.read_bytes()

    restored = read_sample_allocation(str(filename))
    assert type(restored) is allocation_type
    np.testing.assert_array_equal(restored.compressed_allocation, compressed)
    assert filename.read_bytes() == original_bytes


@pytest.mark.parametrize("allocation_type", ALLOCATION_TYPES)
def test_nested_dataset_takes_precedence(tmp_path, allocation_type):
    filename = tmp_path / "both.h5"
    with h5py.File(filename, "w") as saved:
        saved.create_dataset(DATASET_PATH, data=ALLOCATIONS[1])
        saved.create_dataset("compressed_allocation", data=ALLOCATIONS[0])
        saved.attrs["Method"] = allocation_type.__module__
    restored = read_sample_allocation(filename)
    np.testing.assert_array_equal(
        restored.compressed_allocation, ALLOCATIONS[1])


@pytest.mark.parametrize("allocation_type", ALLOCATION_TYPES)
def test_save_overwrites_previous_allocation(tmp_path, allocation_type):
    filename = tmp_path / "overwrite.h5"
    allocation_type(ALLOCATIONS[0]).save(filename)
    allocation_type(ALLOCATIONS[1]).save(filename)
    restored = read_sample_allocation(filename)
    np.testing.assert_array_equal(
        restored.compressed_allocation, ALLOCATIONS[1])


@pytest.mark.parametrize("allocation_type",
                         [ACVSampleAllocation, MLMCSampleAllocation])
def test_restored_allocation_preserves_estimates(tmp_path, allocation_type):
    allocation = allocation_type(ALLOCATIONS[1])
    covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
    samples = np.arange(allocation.num_total_samples, dtype=float)
    outputs = allocation.allocate_samples_to_models(samples)
    expected = Estimator(allocation, covariance)
    filename = tmp_path / "estimate.h5"
    allocation.save(filename)
    restored = Estimator(read_sample_allocation(filename), covariance)
    assert restored.get_estimate(outputs) == expected.get_estimate(outputs)
    assert restored.approximate_variance == expected.approximate_variance


@pytest.mark.parametrize("missing", ["dataset", "method"])
def test_invalid_file_remains_an_error(tmp_path, missing):
    filename = tmp_path / "invalid.h5"
    with h5py.File(filename, "w") as saved:
        if missing != "dataset":
            saved.create_dataset(DATASET_PATH, data=ALLOCATIONS[0])
        if missing != "method":
            saved.attrs["Method"] = SampleAllocationBase.__module__
    with pytest.raises(KeyError):
        read_sample_allocation(filename)
