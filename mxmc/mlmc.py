
import numpy as np

from .optimizer import OptimizationResult, Optimizer


class MLMC(Optimizer):
    def __init__(self, model_costs, covariance=None, mlmc_variances=None):
        return


    def optimize(self, target_cost):

        actual_cost = 0
        estimator_variance=0
        allocation = np.array([[0]])
        result = OptimizationResult(actual_cost, estimator_variance, allocation)
        return result
    
