"""
This example compares the performance of several sample allocation
optimization algorithms on a randomly generated model scenario. In this context,
a model scenario is parameterized by the model covariance matrix, model costs,
and the target cost.

The LKJ Cholesky Correlation Prior [LEWANDOWSKI2009]_ is used to randomly
sample a correlation matrix from the uniform distribution over all possible
correlation matrices. This sampling is implemented in PyMC3, which is a
required dependency to run this example (pip install pymc3). Model costs and
variances are then sampled from user-defined distributions to fully define
the model scenario.

.. [LEWANDOWSKI2009] Lewandowski, Daniel, Dorota Kurowicka, and Harry Joe.
   "Generating random correlation matrices based on vines and extended onion
   method." Journal of Multivariate Analysis 100.9 (2009): 1989-2001.

"""

import numpy as np
import pymc3 as pm

from mxmc import Optimizer
from mxmc.optimizers.optimizer_base import InconsistentModelError

def gen_random_corr(num_models, eta=1):
    model = pm.Model()
    with model:
        packed = pm.LKJCorr('packed_L', n=num_models, eta=eta, transform=None)
        packed_array = packed.random(1)
    return unpack_upper_triangle(packed_array, num_models)


def unpack_upper_triangle(packed, num_models):
    '''
    Format of packed upper triangle
    [[- 0 1 2 3]
     [- - 4 5 6]
     [- - - 7 8]
     [- - - - 9]
     [- - - - -]]
    '''
    if isinstance(packed, float):
        return np.array([[1, packed], [packed, 1]])

    corr_matrix = np.eye(num_models)
    packed_iter = iter(packed[0])

    for i in range(num_models - 1):
        for j in range(i + 1, num_models):
            corr_matrix[i, j] = packed_iter.__next__()

    return corr_matrix + np.triu(corr_matrix, 1).T


def build_cov_matrix(corr_matrix, var_ratios, hifi_variance):
    cov_matrix = np.zeros(corr_matrix.shape)
    for i, vr_i in enumerate(var_ratios):
        for j, vr_j in enumerate(var_ratios):
            std_i = np.sqrt(vr_i * hifi_variance)
            std_j = np.sqrt(vr_j * hifi_variance)
            cov_matrix[i, j] = corr_matrix[i, j] * std_i * std_j
    return cov_matrix


if __name__ == '__main__':

    num_models = 4
    target_cost = 1
    hifi_variance = 1
    hifi_cost = target_cost / 100.
    min_cost_ratio = -6
    var_ratio_bnds = (0.1, 1.5)
    eta = 1 # uniform sampling over correlation matrix space

    algorithms_to_compare = ["mlmc", "wrdiff",  "grdmr", "acvis", "gismr",
                             "mfmc", "acvmf", "acvkl", "acvmfu", "gmfmr"]

    cost_ratios = 10 ** np.random.uniform(-6, 0, num_models)
    cost_ratios[0] = 1
    var_ratios = np.random.uniform(var_ratio_bnds[0], var_ratio_bnds[1],
                 num_models)
    var_ratios[0] = 1

    costs = cost_ratios * hifi_cost

    corr_matrix = gen_random_corr(num_models, eta)
    covariance = build_cov_matrix(corr_matrix, var_ratios, hifi_variance)

    optimizer = Optimizer(costs, covariance)

    print("----------------------------------------")
    print(" Algorithm   Variance   Variance w/ AMS")
    print("----------------------------------------")
    template = "{:^11s} {:^10.2e} {:^17.2e}"
    for algorithm in algorithms_to_compare:
        variance = []
        for ams in [True, False]:
            try:
                var = optimizer.optimize(algorithm, target_cost, ams).variance

            except InconsistentModelError:
                var = np.nan

            variance.append(var)

        print(template.format(algorithm, variance[0], variance[1]))
