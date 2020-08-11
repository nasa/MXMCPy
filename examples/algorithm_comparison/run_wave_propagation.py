"""
This example compares the performance of several sample allocation
optimization algorithms on three model scenarios.  The model scenarios were
originally detailed in [GORODETSKY2019]_ and consist of 10 wave propagation
models models (5 discretization levels for 2 fidelities). A target cost of 30
is used in all three scenarios.

.. [GORODETSKY2019] Gorodetsky, Alex A., et al. "A generalized approximate
   control variate framework for multifidelity uncertainty quantification."
   Journal of Computational Physics 408 (2020): 109257.

"""

import numpy as np
from mxmc import Optimizer


COVARIANCE = {
    "multi-level":
        np.array([[6.65766181e-03, 6.04188989e-03, 6.08250235e-03, 4.44249691e-03, 2.01783173e-03],
                  [6.04188989e-03, 5.50090784e-03, 5.52036014e-03, 4.03400912e-03, 1.80267218e-03],
                  [6.08250235e-03, 5.52036014e-03, 5.64192736e-03, 4.18272846e-03, 2.01581639e-03],
                  [4.44249691e-03, 4.03400912e-03, 4.18272846e-03, 3.17935213e-03, 1.66504443e-03],
                  [2.01783173e-03, 1.80267218e-03, 2.01581639e-03, 1.66504443e-03, 1.23863308e-03]]),
    "multifidelity":
        np.array([[6.65766181e-03, 4.60502693e-03, 3.97264949e-03, 2.61840669e-03, 1.32530016e-03],
                  [4.60502693e-03, 3.29418203e-03, 2.91843871e-03, 1.98862505e-03, 1.08735695e-03],
                  [3.97264949e-03, 2.91843871e-03, 2.69095674e-03, 1.90608074e-03, 1.16773677e-03],
                  [2.61840669e-03, 1.98862505e-03, 1.90608074e-03, 1.41199346e-03, 9.57954385e-04],
                  [1.32530016e-03, 1.08735695e-03, 1.16773677e-03, 9.57954385e-04, 8.20080820e-04]]),
    "multifidelity-with-gap":
        np.array([[6.65766181e-03, 3.97264949e-03, 2.61840669e-03, 1.32530016e-03],
                  [3.97264949e-03, 2.69095674e-03, 1.90608074e-03, 1.16773677e-03],
                  [2.61840669e-03, 1.90608074e-03, 1.41199346e-03, 9.57954385e-04],
                  [1.32530016e-03, 1.16773677e-03, 9.57954385e-04, 8.20080820e-04]])
}
MODELCOSTS = {"multi-level": np.array([1.000, 0.147, 0.026, 0.009, 0.002]),
              "multifidelity": np.array([1.000, 0.080, 0.013, 0.004, 0.002]),
              "multifidelity-with-gap": np.array([1.000, 0.013, 0.004, 0.002])}


target_cost = 30
algorithms_to_compare = ["mlmc", "wrdiff",  "grdmr", "acvis", "gismr", "mfmc",
                         "acvmf", "acvkl", "acvmfu", "gmfmr"]
model_scenarios = ["multi-level", "multifidelity", "multifidelity-with-gap"]


for scenario in model_scenarios:
    print("\n-------:: {:^22s} ::-------".format(scenario))
    print("  Algorithm   Variance   Variance w/ AMS ")
    print("------------------------------------------")
    template = " {:^11s} {:^10.2e} {:^17.2e} "
    optimizer = Optimizer(MODELCOSTS[scenario], COVARIANCE[scenario])
    for algorithm in algorithms_to_compare:
        opt_result = optimizer.optimize(algorithm, target_cost,
                                        auto_model_selection=False)
        opt_result_ams = optimizer.optimize(algorithm, target_cost,
                                            auto_model_selection=True)
        print(template.format(algorithm, opt_result.variance,
                              opt_result_ams.variance))


