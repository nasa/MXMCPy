from collections import namedtuple
from abc import ABCMeta, abstractmethod

import numpy as np

OptimizationResult = namedtuple('OptimizationResult',
                                'cost variance sample_array')


class Optimizer(metaclass=ABCMeta):
    def __init__(self, model_costs, covariance=None):
        self._model_costs = model_costs
        self._num_models = len(self._model_costs)
        if covariance is not None:
            self._validation(covariance)

    def _validation(self, covariance):
        if len(covariance) != self._num_models:
            raise ValueError("Covariance and model cost dimensions must match")
        if not np.allclose(covariance.transpose(), covariance):
            raise ValueError("Covariance array must be symmetric")

    @abstractmethod
    def optimize(self, target_cost):
        raise NotImplementedError
