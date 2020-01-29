import numpy as np
import torch

from .acv_optimizer import ACVOptimizer, TORCHDTYPE


class ACVStandard(ACVOptimizer):

    def _get_model_eval_ratios(self, ratios):
        full_ratios = np.ones(len(ratios) + 1)
        full_ratios[1:] = ratios
        return full_ratios

    def _get_model_eval_ratios_autodiff(self, ratios_tensor):
        full_ratios = torch.ones(len(ratios_tensor) + 1, dtype=TORCHDTYPE)
        full_ratios[1:] = ratios_tensor
        return full_ratios
