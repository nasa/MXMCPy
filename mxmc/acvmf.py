import numpy as np
import torch

from .acv_optimizer import ACVOptimizer, TORCHDTYPE


class ACVMF(ACVOptimizer):

    def _compute_acv_F_matrix(self, ratios):

        F = torch.zeros((self._num_models - 1, self._num_models - 1),
                        dtype=TORCHDTYPE)
        for i in range(self._num_models - 1):
            F[i, i] = (ratios[i] - 1) / ratios[i]
        for i in range(self._num_models - 2):
            for j in range(i + 1, self._num_models - 1):
                min_ratio = torch.min(ratios[[i, j]])
                F[i, j] = (min_ratio - 1) / min_ratio
                F[j, i] = F[i, j]
        return F

    def _make_allocation(self, sample_nums):
        total_sample_nums = np.copy(sample_nums)
        total_sample_nums[1:] += sample_nums[0]
        unique_nums = list(set(total_sample_nums[1:]))
        unique_nums.sort()
        allocation = np.zeros((1 + len(unique_nums), 2 * self._num_models),
                              dtype=int)
        allocation[0, 1:] = np.ones(2 * self._num_models - 1, dtype=int)
        allocation[0, 0] = sample_nums[0]
        for i, num in enumerate(unique_nums):
            allocation[i + 1, 0] = num - np.sum(allocation[:i+1, 0])

        for i in range(1, self._num_models):
            allocation[1:, i*2+1][total_sample_nums[i] >= unique_nums] = 1

        return allocation
