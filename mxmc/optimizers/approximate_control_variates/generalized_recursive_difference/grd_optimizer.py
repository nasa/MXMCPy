import numpy as np
import torch

from ..acv_optimizer import ACVOptimizer, TORCHDTYPE
from ..acv_constraints import ACVConstraints


class GRDOptimizer(ACVOptimizer, ACVConstraints):

    def _get_bounds(self):
        return [(0, np.inf)] * (self._num_models - 1)

    def _get_constraints(self, target_cost):
        constraints = self._constr_n_greater_than_1(target_cost)
        ref_constraints = self._constr_ratios_result_in_samples_greater_than_1(
                target_cost)
        constraints.extend(ref_constraints)
        return constraints

    def _compute_acv_F_and_F0(self, ratios):
        ones = torch.ones(len(ratios), dtype=TORCHDTYPE)
        full_ratios = torch.ones(len(ratios) + 1, dtype=TORCHDTYPE)
        full_ratios[1:] = ratios
        ref_ratios = full_ratios[self._recursion_refs]
        ref_tensor = torch.tensor(self._recursion_refs, dtype=TORCHDTYPE)

        modelia = torch.ger(ref_tensor, ones)
        modelib = torch.ger(torch.arange(1, self._num_models,
                                         dtype=TORCHDTYPE), ones)
        modelja = torch.transpose(modelia, 0, 1)
        modeljb = torch.transpose(modelib, 0, 1)

        ria = torch.ger(ref_ratios, ones)
        rib = torch.ger(ratios, ones)
        rja = torch.transpose(ria, 0, 1)
        rjb = torch.transpose(rib, 0, 1)

        F = ria * (modelia == modelja).type(TORCHDTYPE) / (ria * rja) \
            - ria * (modelia == modeljb).type(TORCHDTYPE) / (ria * rjb) \
            - rib * (modelib == modelja).type(TORCHDTYPE) / (rib * rja) \
            + rib * (modelib == modeljb).type(TORCHDTYPE) / (rib * rjb)

        F0 = torch.zeros(len(ratios), dtype=TORCHDTYPE)
        filter = [i == 0 for i in self._recursion_refs]
        F0[filter] = 1

        return F, F0

    def _make_allocation(self, sample_nums):
        allocation = np.zeros([len(sample_nums), self._num_models * 2],
                              dtype=int)
        for i in range(len(sample_nums)):
            allocation[i, i * 2 + 1] = 1
        for i in range(1, self._num_models):
            ref_model = self._recursion_refs[i - 1]
            allocation[:, i * 2] = allocation[:, ref_model * 2 + 1]

        allocation[:, 0] = sample_nums

        return allocation

    def _get_model_eval_ratios(self, ratios):
        full_ratios = np.ones(len(ratios) + 1)
        full_ratios[1:] = ratios
        ref_ratios = np.zeros(len(ratios) + 1)
        ref_ratios[1:] = full_ratios[self._recursion_refs]
        eval_ratios = full_ratios + ref_ratios
        return eval_ratios

    def _get_model_eval_ratios_autodiff(self, ratios_tensor):
        full_ratios = torch.ones(len(ratios_tensor) + 1, dtype=TORCHDTYPE)
        full_ratios[1:] = ratios_tensor
        ref_ratios = torch.zeros(len(ratios_tensor) + 1, dtype=TORCHDTYPE)
        ref_ratios[1:] = full_ratios[self._recursion_refs]
        eval_ratios = full_ratios + ref_ratios
        return eval_ratios
