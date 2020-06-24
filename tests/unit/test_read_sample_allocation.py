def test_sample_allocation_read(saved_allocation_path, sample_allocation):
    loaded_allocation = read_sample_allocation(saved_allocation_path)
    np.testing.assert_array_equal(loaded_allocation.compressed_allocation,
                                  sample_allocation.compressed_allocation)
    assert loaded_allocation.num_models == sample_allocation.num_models
    assert loaded_allocation.method == "ACV"
    # need to work on new class based identification of method
    assert False
