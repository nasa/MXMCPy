import numpy as np

from .optimizer_base import OptimizerBase


class ACVKLEnumerator(OptimizerBase):

    def optimize(self, target_cost):
        if target_cost < np.sum(self._model_costs):
            return self._get_invalid_result()
        if self._num_models == 1:
            return self._get_monte_carlo_result(target_cost)

        return self._get_monte_carlo_result(target_cost)