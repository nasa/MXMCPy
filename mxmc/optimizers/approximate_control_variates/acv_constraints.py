from abc import abstractmethod


def satisfies_constraints(x, constraints):
    for constr in constraints:
        c_val = constr["fun"](x, *constr["args"])
        if c_val < 0:
            print(constr["fun"])
            return False
    return True


class ACVConstraints:

    def _constr_n_greater_than_1(self, target_cost):
        def n_constraint(ratios):
            N = self._calculate_n(ratios, target_cost)
            return N - 1
        return [{"type": "ineq", "fun": n_constraint, "args": tuple()}]

    def _constr_ratios_result_in_samples_1_greater_than_n(self, target_cost):
        def n_ratio_constraint(ratios, ind):
            N = self._calculate_n(ratios, target_cost)
            return N * (ratios[ind] - 1) - 1

        nr_constraints = []
        for ind in range(self._num_models - 1):
            nr_constraints.append({"type": "ineq",
                                   "fun": n_ratio_constraint,
                                   "args": (ind, )})
        return nr_constraints

    def _constr_ratios_result_in_samples_greater_than_1(self, target_cost):
        def ratio_1_constraint(ratios, ind):
            N = self._calculate_n(ratios, target_cost)
            return N * (ratios[ind]) - 1

        r1_constraints = []
        for ind in range(self._num_models - 1):
            r1_constraints.append({"type": "ineq",
                                   "fun": ratio_1_constraint,
                                   "args": (ind, )})
        return r1_constraints

    def _constr_ratios_result_in_samples_1_greater_than_prev_ratio(
            self, target_cost):
        def n_ratio_constraint(ratios, ind):
            N = self._calculate_n(ratios, target_cost)
            return N * (ratios[ind] - 1) - 1

        def r_ratio_constraint(ratios, ind):
            N = self._calculate_n(ratios, target_cost)
            return N * (ratios[ind] - ratios[ind-1]) - 1

        nr_constraints = []
        if self._num_models > 1:
            nr_constraints.append({"type": "ineq", "fun": n_ratio_constraint,
                                   "args": (0, )})
        for ind in range(1, self._num_models - 1):
            nr_constraints.append({"type": "ineq",
                                   "fun": r_ratio_constraint,
                                   "args": (ind, )})
        return nr_constraints

    def _constr_ratios_result_in_samples_1_different_than_ref(self,
                                                              target_cost):
        def n_ratio_constraint(ratios, ind):
            N = self._calculate_n(ratios, target_cost)
            return N * (ratios[ind] - 1) - 1

        def ratio_ref_constraint(ratios, ind, ref):
            N = self._calculate_n(ratios, target_cost)
            return N * abs(ratios[ind] - ratios[ref - 1]) - 1

        rl_constraints = []
        for ind, ref in enumerate(self._recursion_refs):
            if ref == 0:
                rl_constraints.append({"type": "ineq",
                                       "fun": n_ratio_constraint,
                                       "args": (ind,)})
            else:
                rl_constraints.append({"type": "ineq",
                                       "fun": ratio_ref_constraint,
                                       "args": (ind, ref)})
        return rl_constraints

    @abstractmethod
    def _calculate_n(self, ratios, target_cost):
        raise NotImplementedError
