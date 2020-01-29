from ..acv_constraints import ACVConstraints
from .gmf_optimizer import GMFOptimizer


class GMFOrdered(GMFOptimizer, ACVConstraints):

    def _get_constraints(self, target_cost):
        constraints = self._constr_n_greater_than_1(target_cost)
        ref_constraints = \
            self._constr_ratios_result_in_samples_1_greater_than_prev_ratio(
                    target_cost)
        constraints.extend(ref_constraints)
        return constraints