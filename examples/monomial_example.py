import numpy as np
import pandas as pd

from common_functions import calculate_variances_for_target_costs, \
                             print_variances_as_table, \
                             plot_variances


def monomial_model_variances(powers):
    num_models = len(powers)
    cov = np.empty((num_models, num_models))
    for i, p_i in enumerate(powers):
        for j, p_j in enumerate(powers):
            cov[i, j] = 1.0 / (p_i + p_j + 1) - 1.0 / ((p_i + 1) * (p_j + 1))
    return cov


def monomial_model_costs(powers):
    return np.power(10.0, -np.arange(len(powers)))


if __name__ == "__main__":
    EXPONENTS = [5, 4, 3, 2, 1]
    EXP_STR = "".join([str(i) for i in EXPONENTS])

    # Generate variances using MXMC
    # COVARIANCE = monomial_model_variances(EXPONENTS)
    # MODELCOSTS = monomial_model_costs(EXPONENTS)
    # TARGET_COSTS = [10, 20, 40, 80]
    # VARIANCES = calculate_variances_for_target_costs(COVARIANCE, MODELCOSTS,
    #                                                  TARGET_COSTS)
    # VARIANCES.to_csv('results/monomial_output_' + EXP_STR + '.csv')

    # Load variances from previously generated csv
    VARIANCES = pd.read_csv('results/monomial_output_' + EXP_STR + '.csv',
                            index_col=0)

    print_variances_as_table(VARIANCES)
    plot_variances(VARIANCES, plot_type="line",
                   title="Monomial Models with Exponents: " + str(EXPONENTS))




