import numpy as np
from scipy import optimize as scipy_optimize
import torch

from .acv_optimizer import ACVOptimizer

class ACVIS(ACVOptimizer):

    def __init__(self, model_costs, covariance, *args, **kwargs):
        super().__init__(model_costs, covariance, *args, **kwargs)

    def _compute_acv_F_matrix(self, ratios):

        F = torch.zeros((self._num_models-1, self._num_models-1))
        for i in range(self._num_models-1):
            F[i, i] = (ratios[i]-1)/ratios[i]
        for i in range(self._num_models-2):
            for j in range(i+1, self._num_models-1):
                F[i, j] = ((ratios[i]-1)/ratios[i])*((ratios[j]-1)/ratios[j])
                F[j, i] = F[i, j]
        return F

    def _make_allocation(self, sample_nums):

        allocation = np.zeros((self._num_models, 2 * self._num_models),
                              dtype=int)
        allocation[0,1:] = np.ones(2*self._num_models-1, dtype=int)
        allocation[0, 0] = sample_nums[0]

        deltaNs_unused = list(sample_nums[1:])
        deltaNs_used = []

        for i in range(self._num_models-1):
            min_ind = deltaNs_unused.index(min(deltaNs_unused))
            true_ind = np.argwhere(sample_nums[1:] == min(deltaNs_unused))
            min_deltaN = deltaNs_unused[min_ind]
            allocation[i+1, 0] = min_deltaN
            allocation[i+1, 3+2*true_ind] = 1
            deltaNs_used.append(deltaNs_unused.pop(min_ind))

        return allocation
