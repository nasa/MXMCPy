import numpy as np
import pytest

from mxmc.estimator import Estimator
from mxmc.optimizer import Optimizer
from mxmc.sample_allocation import SampleAllocation

ALGORITHMS = ["mfmc", "mlmc", "acvmf", "acvis", "acvkl"]


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
    evals = sample_array[:, 1:].transpose().dot(sample_array[:, 0])
    evals = np.insert(evals, 0, 0).reshape((-1, 2))
    cost = np.max(evals, axis=1).dot(model_costs)
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

    assert actual_cost <= target_cost
