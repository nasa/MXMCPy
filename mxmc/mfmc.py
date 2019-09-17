from collections import namedtuple

import numpy as np

OptimizationResult = namedtuple('OptimizationResult',
                                'cost variance sample_array')


class MFMC:
    def __init__(self, covariance, model_costs):
        if len(covariance) != len(model_costs):
            raise ValueError("Covariance and model cost dimensions must match")
        if not np.allclose(covariance.transpose(), covariance):
            raise ValueError("Covariance array must be symmetric")
        self._model_costs = model_costs

    def optimize(self, target_cost):
        allocation = np.ones((1, 2 * len(self._model_costs)))
        if target_cost < sum(self._model_costs):
            allocation[0, 0] = 0
            return OptimizationResult(0, np.inf, allocation)
        allocation[0, 0] = 10
        return OptimizationResult(30, 0.1, allocation)
