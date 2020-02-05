import numpy as np
import pandas as pd

from common_functions import calculate_variances_for_target_costs, \
                             print_variances_as_table, \
                             plot_variances

# CASE a - ML example:   200 (II)  100 (II)	 50 (II)  25 (II)  10 (II)
# CASE b - MF Example 1: 200 (II)  100 (I)	 50 (I)	  25 (I)   10 (I)
# CASE c - MF Example 2: 200 (II)  50 (I)	 25 (I)	  10 (I)
COVARIANCE = {"a": np.array([[6.65766181e-03, 6.04188989e-03, 6.08250235e-03, 4.44249691e-03, 2.01783173e-03],
                             [6.04188989e-03, 5.50090784e-03, 5.52036014e-03, 4.03400912e-03, 1.80267218e-03],
                             [6.08250235e-03, 5.52036014e-03, 5.64192736e-03, 4.18272846e-03, 2.01581639e-03],
                             [4.44249691e-03, 4.03400912e-03, 4.18272846e-03, 3.17935213e-03, 1.66504443e-03],
                             [2.01783173e-03, 1.80267218e-03, 2.01581639e-03, 1.66504443e-03, 1.23863308e-03]]),
              "b": np.array([[6.65766181e-03, 4.60502693e-03, 3.97264949e-03, 2.61840669e-03, 1.32530016e-03],
                             [4.60502693e-03, 3.29418203e-03, 2.91843871e-03, 1.98862505e-03, 1.08735695e-03],
                             [3.97264949e-03, 2.91843871e-03, 2.69095674e-03, 1.90608074e-03, 1.16773677e-03],
                             [2.61840669e-03, 1.98862505e-03, 1.90608074e-03, 1.41199346e-03, 9.57954385e-04],
                             [1.32530016e-03, 1.08735695e-03, 1.16773677e-03, 9.57954385e-04, 8.20080820e-04]]),
              "c": np.array([[6.65766181e-03, 3.97264949e-03, 2.61840669e-03, 1.32530016e-03],
                             [3.97264949e-03, 2.69095674e-03, 1.90608074e-03, 1.16773677e-03],
                             [2.61840669e-03, 1.90608074e-03, 1.41199346e-03, 9.57954385e-04],
                             [1.32530016e-03, 1.16773677e-03, 9.57954385e-04, 8.20080820e-04]])}
MODELCOSTS = {"a": np.array([1.000, 0.147, 0.026, 0.009, 0.002]),
              "b": np.array([1.000, 0.080, 0.013, 0.004, 0.002]),
              "c": np.array([1.000, 0.013, 0.004, 0.002])}
# cost = 6*cost if using unnormalized costs


if __name__ == "__main__":
    TEST_CASE = 'c'

    # Generate variances using MXMC
    # TARGET_COSTS = [15, 30, 60]
    # VARIANCES = calculate_variances_for_target_costs(COVARIANCE[TEST_CASE],
    #                                                  MODELCOSTS[TEST_CASE],
    #                                                  TARGET_COSTS)
    # VARIANCES.to_csv('results/wave_prop_output_' + TEST_CASE + '.csv')

    # Load variances from previously generated csv
    VARIANCES = pd.read_csv('results/wave_prop_output_' + TEST_CASE + '.csv',
                            index_col=0)

    print_variances_as_table(VARIANCES)
    plot_variances(VARIANCES, plot_type="line",
                   title='Wave Propagation ({})'.format(TEST_CASE))

