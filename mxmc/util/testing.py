import numpy as np


def assert_opt_result_equal(opt_result, cost_ref, var_ref, sample_array_ref):
    assert np.isclose(opt_result.cost, cost_ref)
    assert np.isclose(opt_result.variance, var_ref)
    opt_sample_array = opt_result.allocation.compressed_allocation
    np.testing.assert_array_almost_equal(opt_sample_array, sample_array_ref)