from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np

OptimizationResult = namedtuple('OptimizationResult',
                                'cost variance sample_array')


class InconsistentModelError(Exception):
    pass


class OptimizerBase(metaclass=ABCMeta):
    def __init__(self, model_costs, covariance=None, *_, **__):
        self._model_costs = model_costs
        self._num_models = len(self._model_costs)
        self._covariance = covariance

        if covariance is not None:
            self._validate_covariance_matrix(covariance)

    def _validate_covariance_matrix(self, matrix):
        if len(matrix) != self._num_models:
            error_msg = "Covariance matrix and model cost dimensions must match"
            raise ValueError(error_msg)
        if not np.allclose(matrix.transpose(), matrix):
            error_msg = "Covariance matrix array must be symmetric"
            raise ValueError(error_msg)

    @abstractmethod
    def optimize(self, target_cost):
        raise NotImplementedError

    def subset(self, model_indices):
        subset_costs = np.copy(self._model_costs[model_indices])
        subset_covariance = self._get_subset_of_matrix(self._covariance,
                                                       model_indices)
        return self.__class__(subset_costs, subset_covariance)

    @staticmethod
    def _get_subset_of_matrix(matrix, model_indices):
        if matrix is None:
            return None
        return np.copy(matrix[model_indices][:, model_indices])

    def get_num_models(self):
        return self._num_models

    def get_invalid_result(self):
        allocation = np.ones((1, 2 * self._num_models))
        allocation[0, 0] = 0
        return OptimizationResult(0, np.inf, allocation)
