
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
