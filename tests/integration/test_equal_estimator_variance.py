import numpy as np
import pytest

from mxmc.estimator import Estimator
from mxmc.optimizer import Optimizer
from mxmc.sample_allocation import SampleAllocation

ALGORITHMS = ["mfmc", "mlmc", "acvmf", "acvis"]


def monomial_model_variances(powers):
    num_models = len(powers)
    cov = np.empty((num_models, num_models))
    vardiff = np.empty((num_models, num_models))
    for i, p_i in enumerate(powers):
        for j, p_j in enumerate(powers):
            cov[i, j] = 1.0 / (p_i + p_j + 1) - 1.0 / ((p_i + 1) * (p_j + 1))
            if i == j:
                vardiff[i, j] = cov[i, j]
            else:
                vardiff[i, j] = 1 / (2 * p_i + 1) - 2 / (p_i + p_j + 1) \
                                + 1 / (2 * p_j + 1) \
                                - (1 / (p_i + 1) - 1 / (p_j + 1)) ** 2
    return cov, vardiff


def monomial_model_costs(powers):
    return np.power(10.0, -np.arange(len(powers)))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_monomial_model(algorithm):
    exponents = [5, 4, 3, 2, 1]
    covariance, vardiff_matrix = monomial_model_variances(exponents)
    model_costs = monomial_model_costs(exponents)
    optimizer = Optimizer(model_costs, covariance=covariance,
                          vardiff_matrix=vardiff_matrix)

    opt_result = optimizer.optimize(algorithm=algorithm, target_cost=10)
    print(opt_result.sample_array)
    sample_allocation = SampleAllocation(opt_result.sample_array,
                                         method=algorithm)
    estimator = Estimator(sample_allocation, covariance)
    estimator_approx_variance = estimator.approximate_variance
    optimizer_approx_variance = opt_result.variance

    assert estimator_approx_variance \
           == pytest.approx(optimizer_approx_variance)
