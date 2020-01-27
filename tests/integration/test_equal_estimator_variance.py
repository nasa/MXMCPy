import numpy as np
import pytest

from mxmc.estimator import Estimator
from mxmc.optimizer import Optimizer
from mxmc.sample_allocation import SampleAllocation

ALGORITHMS = ["mfmc", "mlmc", "acvmf", "acvmfu", "acvis", "acvkl"]


def monomial_model_covariance(powers):
    num_models = len(powers)
    cov = np.empty((num_models, num_models))
    for i, p_i in enumerate(powers):
        for j, p_j in enumerate(powers):
            cov[i, j] = 1.0 / (p_i + p_j + 1) - 1.0 / ((p_i + 1) * (p_j + 1))
    return cov


def monomial_model_costs(powers):
    return np.power(10.0, -np.arange(len(powers)))


def calculate_costs_from_sample_array(sample_array, model_costs):
    alloc = sample_array[:, 1:].transpose()
    evals = [alloc[0]]
    for i in range(1, alloc.shape[0], 2):
        evals.append(np.max(alloc[i:i+2], axis=0))
    evals = np.array(evals)
    cost = np.dot(evals.dot(sample_array[:, 0]), model_costs)
    return cost


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_monomial_model(algorithm):
    exponents = [4, 3, 2, 1]
    covariance = monomial_model_covariance(exponents)
    model_costs = monomial_model_costs(exponents)
    target_cost = 10
    optimizer = Optimizer(model_costs, covariance=covariance)

    opt_result = optimizer.optimize(algorithm, target_cost)
    sample_allocation = SampleAllocation(opt_result.sample_array,
                                         method=algorithm)
    estimator = Estimator(sample_allocation, covariance)
    estimator_approx_variance = estimator.approximate_variance
    optimizer_approx_variance = opt_result.variance

    assert estimator_approx_variance \
        == pytest.approx(optimizer_approx_variance)

    actual_cost = calculate_costs_from_sample_array(opt_result.sample_array,
                                                    model_costs)
    
    assert actual_cost == opt_result.cost
