import numpy as np
import pytest

from mxmc.acv_constraints import ACVConstraints

DUMMYVALUE = None


class MockedConstrained(ACVConstraints):
    def __init__(self, num_models, n_value):
        self._num_models = num_models
        self._n_value = n_value

    def _calculate_n(self, _ratios, _target_cost):
        return self._n_value


def _count_constraints_violated(constraints, ratios):
    violated = 0
    for constr in constraints:
        if constr["fun"](ratios, *constr["args"]) < 0:
            violated += 1
    return violated


@pytest.mark.parametrize("num_models", range(1, 4))
def test_n_gt_1_constraint_has_1_entry(num_models):
    opt = MockedConstrained(num_models, DUMMYVALUE)
    constraints = opt._constr_n_greater_than_1(target_cost=DUMMYVALUE)
    assert len(constraints) == 1


@pytest.mark.parametrize("n_value, expected_violations",
                         [(0, 1), (0.999, 1), (1, 0), (10, 0)])
def test_n_gt_1_constraint_is_accurate(n_value, expected_violations):
    opt = MockedConstrained(DUMMYVALUE, n_value)
    constraints = opt._constr_n_greater_than_1(target_cost=DUMMYVALUE)
    constr_violated = _count_constraints_violated(constraints,
                                                  ratios=DUMMYVALUE)
    assert constr_violated == expected_violations


@pytest.mark.parametrize("num_models", range(1, 4))
def test_r_1_gt_n_constraints_correct_size(num_models):
    opt = MockedConstrained(num_models, DUMMYVALUE)
    constraints = opt._constr_ratios_result_in_samples_1_greater_than_n(
            target_cost=DUMMYVALUE)
    assert len(constraints) == num_models - 1


@pytest.mark.parametrize("r1_violates", [True, False])
@pytest.mark.parametrize("r2_violates", [True, False])
@pytest.mark.parametrize("r3_violates", [True, False])
def test_r_1_gt_n_constraints_are_accurate(r1_violates, r2_violates,
                                           r3_violates):
    opt = MockedConstrained(num_models=4, n_value=1)
    constraints = opt._constr_ratios_result_in_samples_1_greater_than_n(
            target_cost=DUMMYVALUE)

    ratios = np.array([2., 2., 2.])
    violations = np.array([r1_violates, r2_violates, r3_violates])
    ratios[violations] -= 0.1
    expected_violations = np.count_nonzero(violations)

    constr_violated = _count_constraints_violated(constraints, ratios)
    assert constr_violated == expected_violations


@pytest.mark.parametrize("num_models", range(1, 4))
def test_r_1_gt_prevr_constraints_correct_size(num_models):
    opt = MockedConstrained(num_models, DUMMYVALUE)
    constraints = \
        opt._constr_ratios_result_in_samples_1_greater_than_prev_ratio(
            target_cost=DUMMYVALUE)
    assert len(constraints) == num_models - 1


@pytest.mark.parametrize("r1_violates", [True, False])
@pytest.mark.parametrize("r2_violates", [True, False])
@pytest.mark.parametrize("r3_violates", [True, False])
def test_r_1_gt_prevr_constraints_are_accurate(r1_violates, r2_violates,
                                               r3_violates):
    opt = MockedConstrained(num_models=4, n_value=1)
    constraints = \
        opt._constr_ratios_result_in_samples_1_greater_than_prev_ratio(
            target_cost=DUMMYVALUE)

    ratios = np.array([2., 3., 4.])
    expected_violations = 0
    if r1_violates:
        ratios -= 0.1
        expected_violations += 1
    if r2_violates:
        ratios[1:] -= 0.1
        expected_violations += 1
    if r3_violates:
        ratios[2:] -= 0.1
        expected_violations += 1

    constr_violated = _count_constraints_violated(constraints, ratios)
    assert constr_violated == expected_violations
