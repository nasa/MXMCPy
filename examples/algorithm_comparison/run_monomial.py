"""
This example compares the performance of several sample allocation
optimization algorithms on two model scenarios.  The model scenarios, as
originally detailed in [GORODETSKY2019]_, consist of 5 algorithm_comparison models:

+-------------+------------------------+
|  Model      |      Model Costs       |
|             +------------------------+
|             | No Cost Gap | Cost Gap |
+=============+=============+==========+
| M0(x) = x^5 |  10^0       |  10^0    |
| M1(x) = x^4 |  10^-1      |  10^-2   |
| M2(x) = x^3 |  10^-2      |  10^-3   |
| M3(x) = x^2 |  10^-3      |  10^-4   |
| M4(x) = x^3 |  10^-4      |  10^-5   |
+-------------+-------------+----------+

A target cost of 20 is used.

.. [GORODETSKY2019] Gorodetsky, Alex A., et al. "A generalized approximate
   control variate framework for multifidelity uncertainty quantification."
   Journal of Computational Physics 408 (2020): 109257.

"""

import numpy as np
from mxmc import Optimizer


def monomial_covariance(exponents):
    num_models = len(exponents)
    cov = np.empty((num_models, num_models))
    for i, p_i in enumerate(exponents):
        for j, p_j in enumerate(exponents):
            cov[i, j] = 1.0 / (p_i + p_j + 1) - 1.0 / ((p_i + 1) * (p_j + 1))
    return cov


covariance = monomial_covariance(exponents=[5, 4, 3, 2, 1])
target_cost = 20
model_costs_without_gap = np.power(10.0, [0, -1, -2, -3, -4])
model_costs_with_gap = np.power(10.0, [0, -2, -3, -4, -5])


algorithms_to_compare = ["mlmc", "wrdiff",  "grdmr", "acvis", "gismr", "mfmc",
                         "acvmf", "acvkl", "acvmfu", "gmfmr"]


# NO COST GAP SCENARIO
print("-------:: No Cost Gap Scenario ::-------")
print(" Algorithm   Variance   Variance w/ AMS")
print("----------------------------------------")
template = "{:^11s} {:^10.2e} {:^17.2e}"
no_cost_gap_optimizer = Optimizer(model_costs_without_gap, covariance)
for algorithm in algorithms_to_compare:
    opt_result = no_cost_gap_optimizer.optimize(algorithm, target_cost,
                                                auto_model_selection=False)
    opt_result_ams = no_cost_gap_optimizer.optimize(algorithm, target_cost,
                                                    auto_model_selection=True)
    print(template.format(algorithm, opt_result.variance,
                          opt_result_ams.variance))


# COST GAP SCENARIO
print("\n---------:: Cost Gap Scenario ::---------")
print(" Algorithm   Variance   Variance w/ AMS")
print("-----------------------------------------")
cost_gap_optimizer = Optimizer(model_costs_with_gap, covariance)
for algorithm in algorithms_to_compare:
    opt_result = cost_gap_optimizer.optimize(algorithm, target_cost,
                                             auto_model_selection=False)
    opt_result_ams = cost_gap_optimizer.optimize(algorithm, target_cost,
                                                 auto_model_selection=True)
    print(template.format(algorithm, opt_result.variance,
                          opt_result_ams.variance))


