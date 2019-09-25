
import numpy as np

from .optimizer import OptimizationResult, Optimizer


class MLMC(Optimizer):
    def __init__(self, model_costs, covariance=None, mlmc_variances=None):

        if type(mlmc_variances) != np.ndarray:
            raise ValueError("MLMC variances must be numpy array")
        if len(model_costs) != len(mlmc_variances):
            raise ValueError("Model costs & variances must have same length!")


    def optimize(self, target_cost):

        actual_cost = 0
        estimator_variance=0
        allocation = np.array([[0]])
        result = OptimizationResult(actual_cost, estimator_variance, allocation)
        return result
    
    
