import numpy as np
import torch

from ..acv_optimizer import ACVOptimizer, TORCHDTYPE


class GMFOptimizer(ACVOptimizer):

    def _get_bounds(self):
        return [(0, np.inf)] * (self._num_models - 1)

    def _compute_acv_F_and_F0(self, ratios):
        ones = torch.ones(len(ratios), dtype=TORCHDTYPE)
        full_ratios = torch.ones(len(ratios) + 1, dtype=TORCHDTYPE)
        full_ratios[1:] = ratios
        ref_ratios = full_ratios[self._recursion_refs]

        ria = torch.ger(ref_ratios, ones)
        rib = torch.ger(ratios, ones)
        rja = torch.transpose(ria, 0, 1)
        rjb = torch.transpose(rib, 0, 1)

        F = torch.min(ria, rja) / (ria * rja) \
            - torch.min(ria, rjb) / (ria * rjb) \
            - torch.min(rib, rja) / (rib * rja) \
            + torch.min(rib, rjb) / (rib * rjb)

        F0 = torch.min(ones, ref_ratios)/ref_ratios \
            - torch.min(ones, ratios)/ratios

        return F, F0

    def _make_allocation(self, sample_nums):
        ordered_sample_nums = np.unique(sample_nums)

        allocation = np.zeros([len(ordered_sample_nums), self._num_models * 2],
                              dtype=int)
        for i, samp in enumerate(sample_nums):
            allocation[samp >= ordered_sample_nums, i * 2 + 1] = 1
        for i in range(1, self._num_models):
            ref_model = self._recursion_refs[i - 1]
            allocation[:, i * 2] = allocation[:, ref_model * 2 + 1]

        allocation[0, 0] = ordered_sample_nums[0]
        allocation[1:, 0] = ordered_sample_nums[1:] - ordered_sample_nums[:-1]

        return allocation

    def _get_model_eval_ratios(self, ratios):
        full_ratios = np.ones(len(ratios) + 1)
        full_ratios[1:] = ratios
        ref_ratios = full_ratios[[0] + self._recursion_refs]
        eval_ratios = np.maximum(full_ratios, ref_ratios)
        return eval_ratios

    def _get_model_eval_ratios_autodiff(self, ratios_tensor):
        full_ratios = torch.ones(len(ratios_tensor) + 1, dtype=TORCHDTYPE)
        full_ratios[1:] = ratios_tensor
        ref_ratios = full_ratios[[0] + self._recursion_refs]
        eval_ratios = torch.max(full_ratios, ref_ratios)
        return eval_ratios
