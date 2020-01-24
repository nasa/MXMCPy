import numpy as np
import torch

from .acv_optimizer import TORCHDTYPE
from .acv_standard import ACVStandard
from .acv_constraints import ACVConstraints


class ACVMFBase(ACVStandard):

    def _compute_acv_F_and_F0(self, ratios):

        F = torch.zeros((self._num_models - 1, self._num_models - 1),
                        dtype=TORCHDTYPE)
        for i in range(self._num_models - 1):
            F[i, i] = (ratios[i] - 1) / ratios[i]
        for i in range(self._num_models - 2):
            for j in range(i + 1, self._num_models - 1):
                min_ratio = torch.min(ratios[[i, j]])
                F[i, j] = (min_ratio - 1) / min_ratio
                F[j, i] = F[i, j]

        F0 = torch.diag(F)
        return F, F0

    def _make_allocation(self, sample_nums):
        unique_nums = list(set(sample_nums[1:]))
        unique_nums.sort()
        allocation = np.zeros((1 + len(unique_nums), 2 * self._num_models),
                              dtype=int)
        allocation[0, 1:] = np.ones(2 * self._num_models - 1, dtype=int)
        allocation[0, 0] = sample_nums[0]
        for i, num in enumerate(unique_nums):
            allocation[i + 1, 0] = num - np.sum(allocation[:i+1, 0])

        for i in range(1, self._num_models):
            allocation[1:, i*2+1][sample_nums[i] >= unique_nums] = 1

        return allocation


class ACVMFU(ACVMFBase, ACVConstraints):

    def _get_constraints(self, target_cost):
        constraints = self._constr_n_greater_than_1(target_cost)
        nr_constraints = \
            self._constr_ratios_result_in_samples_1_greater_than_n(target_cost)
        constraints.extend(nr_constraints)
        return constraints
