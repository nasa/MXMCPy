from abc import abstractmethod


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

    @abstractmethod
    def _calculate_n(self, ratios, target_cost):
        raise NotImplementedError
