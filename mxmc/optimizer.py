from collections import namedtuple
from abc import ABCMeta, abstractmethod

import numpy as np

OptimizationResult = namedtuple('OptimizationResult',
                                'cost variance sample_array')


class Optimizer(metaclass=ABCMeta):
    def __init__(self, model_costs, covariance=None, mlmc_variances=None):
        self._model_costs = model_costs
        self._num_models = len(self._model_costs)
        if covariance is not None:
            self._validate_covariance(covariance)
        if mlmc_variances is not None:
            self._validate_mlmc_variances(mlmc_variances)

    def _validate_covariance(self, covariance):
        if len(covariance) != self._num_models:
            raise ValueError("Covariance and model cost dimensions must match")
        if not np.allclose(covariance.transpose(), covariance):
            raise ValueError("Covariance array must be symmetric")

    def _validate_mlmc_variances(self, mlmc_variances): 
        if type(mlmc_variances) != np.ndarray:
            raise ValueError("MLMC variances must be numpy array")
        if len(self._model_costs) != len(mlmc_variances):
            raise ValueError("Model costs & variances must have same length!")

    @abstractmethod
    def optimize(self, target_cost):
        raise NotImplementedError

    def _make_invalid_result(self):
        allocation = np.ones((1, 2 * self._num_models))
        allocation[0, 0] = 0
        return OptimizationResult(0, np.inf, allocation)
