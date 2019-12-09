import numpy as np
import torch

from .acv_optimizer import ACVOptimizer, TORCHDTYPE


class ACVKL(ACVOptimizer):

    def __init__(self, model_costs, covariance, *args, **kwargs):
        super().__init__(model_costs, covariance, *args, **kwargs)
        self._k_models = kwargs['k']
        self._l_model = kwargs['l']

    def _compute_acv_F_matrix(self, ratios):
        F = torch.zeros((self._num_models - 1, self._num_models - 1),
                        dtype=TORCHDTYPE)

        rL = None if self._l_model is None else ratios[self._l_model-1]

        for i in range(self._num_models - 1):
            model_i = i + 1
            ri = ratios[i]

            for j in range(self._num_models - 1):
                model_j = j + 1
                rj = ratios[j]

                if model_i in self._k_models and model_j in self._k_models:
                    F[i, j] = 1 - 1/rj - 1/ri
                elif model_i in self._k_models:
                    F[i, j] = 1/rL - 1/rj - torch.min(ri, rL)/(ri * rL)
                elif model_j in self._k_models:
                    F[i, j] = 1/rL - torch.min(rj, rL)/(rj * rL) - 1/ri
                else:
                    F[i, j] = 1/rL - 1/rj - 1/ri

                F[i, j] += torch.min(ri, rj)/(ri * rj)

        return F

    def _make_allocation(self, sample_nums):
        return np.zeros([1, self._num_models * 2])
