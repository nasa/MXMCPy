
import numpy as np
import pytest

from mxmc.mlmc import MLMC


@pytest.fixture
def mlmc_optimizer():
    model_costs = np.array([3, 1])
    mlmc_variances = np.array([1, 4])
    return MLMC(model_costs=model_costs, mlmc_variances=mlmc_variances)

@pytest.fixture
def mlmc_three_model():
    model_costs = np.array([5, 3, 1])
    mlmc_variances = np.array([0.5, 1, 4])
    return MLMC(model_costs=model_costs, mlmc_variances=mlmc_variances)

@pytest.fixture
def mlmc_four_model():
    model_costs = np.array([11, 5, 3, 1])
    mlmc_variances = np.array([0.25, 0.5, 1, 4])
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
def test_mlmc_variance_is_valid_type_or_raises_error(variances):

    with pytest.raises(ValueError):
        mlmc = MLMC(model_costs=np.array([1]), mlmc_variances=variances)


@pytest.mark.parametrize("target_cost", [-1, 0.5, 0])
def test_target_cost_too_low_to_run_models(mlmc_optimizer, target_cost):

    opt_result = mlmc_optimizer.optimize(target_cost)
    assert opt_result.cost == 0
    assert np.isinf(opt_result.variance)
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         np.array([[0, 1, 1, 1]], dtype=int))

@pytest.mark.parametrize("num_models", range(1, 4))
def test_opt_results_are_correct_sizes(num_models):
    model_costs = np.ones(num_models)
    mlmc_variances = np.ones(num_models)
    mlmc = MLMC(model_costs=model_costs, mlmc_variances=mlmc_variances)
    opt_result = mlmc.optimize(10)
    assert opt_result.sample_array.shape[0] == num_models
    assert opt_result.sample_array.shape[1] == num_models*2

@pytest.mark.parametrize("target_cost, factor", [(8, 1), (16,2)])
def test_optimize_works_for_simple_two_model_ex(mlmc_optimizer, target_cost,
                                                factor):

    sample_array_expected = np.array([[1*factor,1,1,0], [4*factor,0,0,1]])
    variance_expected = 2/float(factor)
    cost_expected = target_cost

    opt_result = mlmc_optimizer.optimize(target_cost)

    assert opt_result.cost == cost_expected
    assert opt_result.variance == variance_expected
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         sample_array_expected)

@pytest.mark.parametrize("target_cost, factor", [(24, 1), (48,2)])
def test_optimize_works_for_simple_three_model_ex(mlmc_three_model, target_cost,
                                                factor):

    sample_array_expected = np.array([[1*factor,1,1,0,0,0], 
                                      [2*factor,0,0,1,1,0],
                                      [8*factor,0,0,0,0,1]])
    var_expected = 1.5/float(factor)
    cost_expected = target_cost

    opt_result = mlmc_three_model.optimize(target_cost)
    assert opt_result.cost == cost_expected
    assert opt_result.variance == var_expected
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         sample_array_expected)

@pytest.mark.parametrize("target_cost, factor", [(64, 1), (128, 2)])
def test_optimize_works_for_simple_four_model_ex(mlmc_four_model, target_cost,
                                                factor):

    sample_array_expected = np.array([[1*factor,1,1,0,0,0,0,0], 
                                      [2*factor,0,0,1,1,0,0,0],
                                      [4*factor,0,0,0,0,1,1,0],
                                      [16*factor,0,0,0,0,0,0,1]])
    var_expected = 1/float(factor)
    cost_expected = target_cost

    opt_result = mlmc_four_model.optimize(target_cost)
    assert opt_result.cost == cost_expected
    assert opt_result.variance == var_expected
    np.testing.assert_array_almost_equal(opt_result.sample_array,
                                         sample_array_expected)

#Test that result is correct when costs/variances are out of order
