import numpy as np
import pytest

from mxmc.estimator import Estimator
from mxmc.optimizer import Optimizer, ALGORITHM_MAP
from mxmc.optimizers.approximate_control_variates.recursion_enumerator import MREnumerator  # noqa: E501
from mxmc.optimizers.approximate_control_variates.generalized_multifidelity.gmf_unordered import GMFUnordered  # noqa: E501
from mxmc.optimizers.approximate_control_variates.generalized_multifidelity.gmf_ordered import GMFOrdered  # noqa: E501
from mxmc.optimizers.approximate_control_variates.generalized_independent_samples.gis_optimizer import GISOptimizer  # noqa: E501
from mxmc.optimizers.approximate_control_variates.generalized_recursive_difference.grd_optimizer import GRDOptimizer  # noqa: E501

ALGORITHMS = ALGORITHM_MAP.keys()


def _monomial_model_covariance(powers):
    num_models = len(powers)
    cov = np.empty((num_models, num_models))
    for i, p_i in enumerate(powers):
        for j, p_j in enumerate(powers):
            cov[i, j] = 1.0 / (p_i + p_j + 1) - 1.0 / ((p_i + 1) * (p_j + 1))
    return cov


def _monomial_model_costs(powers):
    return np.power(10.0, -np.arange(len(powers)))


def _calculate_costs_from_allocation(allocation, model_costs):
    sample_array = allocation.compressed_allocation
    alloc = sample_array[:, 1:].transpose()
    evals = [alloc[0]]
    for i in range(1, alloc.shape[0], 2):
        evals.append(np.max(alloc[i:i+2], axis=0))
    evals = np.array(evals)
    cost = np.dot(evals.dot(sample_array[:, 0]), model_costs)
    return cost


def _generate_random_ratios_fulfilling_constraints(constraints, model_costs):
    def gen_new_ratios():
        rand = np.random.random(len(model_costs) - 1)
        return rand * model_costs[0] / model_costs[1:]

    ratios = gen_new_ratios()
    while not _constraints_fulfilled(constraints, ratios):
        ratios = gen_new_ratios()
    return ratios


def _constraints_fulfilled(constraints, ratios):
    for constr in constraints:
        if constr["fun"](ratios, *constr["args"]) < 0:
            return False
    return True


def _assert_opt_result_is_consistent(covariance, model_costs, opt_result):
    sample_allocation = opt_result.allocation
    estimator = Estimator(sample_allocation, covariance)
    estimator_approx_variance = estimator.approximate_variance
    optimizer_approx_variance = opt_result.variance
    assert estimator_approx_variance \
        == pytest.approx(optimizer_approx_variance)
    actual_cost = _calculate_costs_from_allocation(opt_result.allocation,
                                                   model_costs)
    assert opt_result.cost == pytest.approx(actual_cost)


class DummyMREnum(MREnumerator):
    def _get_sub_optimizer(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_opt_result_variance_and_cost_match_allocation(algorithm):
    exponents = [4, 3, 2, 1]
    covariance = _monomial_model_covariance(exponents)
    model_costs = _monomial_model_costs(exponents)
    target_cost = 10
    optimizer = Optimizer(model_costs, covariance=covariance)

    opt_result = optimizer.optimize(algorithm, target_cost)
    _assert_opt_result_is_consistent(covariance, model_costs, opt_result)


def test_mfmc_and_acvmfmc_have_about_equal_variance():
    exponents = [4, 3, 2, 1]
    covariance = _monomial_model_covariance(exponents)
    model_costs = _monomial_model_costs(exponents)
    target_cost = 10
    optimizer = Optimizer(model_costs, covariance=covariance)

    analytical_result = optimizer.optimize("mfmc", target_cost)
    numerical_result = optimizer.optimize("acvmfmc", target_cost)

    assert analytical_result.variance \
        == pytest.approx(numerical_result.variance)


@pytest.mark.parametrize("acv_optimizer", ["acvmf", "acvmfu", "acvmfmc",
                                           "acvis", "wrdiff"])
def test_basic_acv_optimizers_give_consistent_output(acv_optimizer, mocker):
    exponents = [4, 3, 2, 1]
    covariance = _monomial_model_covariance(exponents)
    model_costs = _monomial_model_costs(exponents)
    target_cost = 10
    optimizer = ALGORITHM_MAP[acv_optimizer](model_costs,
                                             covariance=covariance)

    constraints = optimizer._get_constraints(target_cost)

    np.random.seed(0)
    for i in range(25):
        valid_ratios = \
            _generate_random_ratios_fulfilling_constraints(constraints,
                                                           model_costs)
        mocker.patch.object(ALGORITHM_MAP[acv_optimizer], '_solve_opt_problem',
                            return_value=valid_ratios)

        opt_result = optimizer.optimize(target_cost)
        _assert_opt_result_is_consistent(covariance, model_costs, opt_result)


@pytest.mark.parametrize("enum_optimizer", [GMFOrdered, GMFUnordered,
                                            GISOptimizer, GRDOptimizer])
def test_enum_acv_optimizers_give_consistent_output(enum_optimizer, mocker):
    exponents = [4, 3, 2, 1]
    covariance = _monomial_model_covariance(exponents)
    model_costs = _monomial_model_costs(exponents)
    target_cost = 10
    enumerator = DummyMREnum(model_costs, covariance=covariance)

    for recursion_refs in enumerator._recursion_iterator():

        optimizer = enum_optimizer(model_costs, covariance=covariance,
                                   recursion_refs=recursion_refs)

        constraints = optimizer._get_constraints(target_cost)

        np.random.seed(0)
        for i in range(10):
            valid_ratios = \
                _generate_random_ratios_fulfilling_constraints(constraints,
                                                               model_costs)
            mocker.patch.object(optimizer, '_solve_opt_problem',
                                return_value=valid_ratios)

            opt_result = optimizer.optimize(target_cost)
            _assert_opt_result_is_consistent(covariance, model_costs,
                                             opt_result)
