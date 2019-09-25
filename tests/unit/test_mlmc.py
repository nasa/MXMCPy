
import numpy as np
import pytest

from mxmc.mlmc import MLMC


@pytest.fixture
def mlmc_optimizer():
    model_costs = np.array([1, 2])
    mlmc_variances = np.array([1, 0.5])
    return MLMC(model_costs=model_costs, mlmc_variances=mlmc_variances)

def test_optimizer_returns_tuple_with(mlmc_optimizer):
    opt_result = mlmc_optimizer.optimize(target_cost=10)
    for member in ["cost", "variance", "sample_array"]:
        assert member in dir(opt_result)

def test_cost_variances_mismatch_raises_error():
    
    costs = np.array([1,2,3])
    variances = np.array([1,2])
    with pytest.raises(ValueError):
        mlmc = MLMC(model_costs=costs, mlmc_variances=variances)

@pytest.mark.parametrize("variances", [1, None, "varz"])
def test_mlmc_variance_is_right_type_or_raises_error(variances):

    with pytest.raises(ValueError):
        mlmc = MLMC(model_costs=np.array([1]), mlmc_variances=variances)


@pytest.mark.parametrize("target_cost", [-1, 0.5, 0])
def test_target_cost_too_low_to_run_models(mlmc_optimizer, target_cost):

    opt_result = mlmc_optimizer.optimize(target_cost)
    assert opt_result.cost == 0
    assert np.isinf(opt_result.variance)
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         np.array([[0, 1, 1, 1]], dtype=int))

