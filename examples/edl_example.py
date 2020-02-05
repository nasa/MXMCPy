import numpy as np
import pandas as pd

from common_functions import calculate_variances_for_target_costs, \
                             print_variances_as_table, \
                             plot_variances

# Models: POST2(dt=0.001), POST2(dt=0.01), POST2(dt=0.1), KNN, SVM
COVARIANCE = {"latitude": np.array([[5.006571668844440361e-03, 4.982831872195880689e-03, 4.990001599278432419e-03, 2.151774373130215922e-03, 2.554720499296729303e-03],
                                    [4.982831872195880689e-03, 5.002219747508702095e-03, 4.986669217729891405e-03, 2.149134420510651934e-03, 2.553527138306761323e-03],
                                    [4.990001599278432419e-03, 4.986669217729891405e-03, 5.014244812405098221e-03, 2.151568433743736644e-03, 2.556029607535900087e-03],
                                    [2.151774373130215922e-03, 2.149134420510651934e-03, 2.151568433743736644e-03, 1.059352606070574103e-03, 1.134114482370184463e-03],
                                    [2.554720499296729303e-03, 2.553527138306761323e-03, 2.556029607535900087e-03, 1.134114482370184463e-03, 1.402897876128381604e-03]]),
              "longitude": np.array([[8.591887371251756209e-03, 8.552093733063356740e-03, 8.553512649352402672e-03, 3.321484435808258386e-03, 5.334927281277434229e-03],
                                     [8.552093733063356740e-03, 8.584454649987324423e-03, 8.551107062423695185e-03, 3.320675282732167034e-03, 5.330640787246037661e-03],
                                     [8.553512649352402672e-03, 8.551107062423695185e-03, 8.588666297123638202e-03, 3.320162258414128403e-03, 5.334636417382157721e-03],
                                     [3.321484435808258386e-03, 3.320675282732167034e-03, 3.320162258414128403e-03, 1.542124328184134134e-03, 2.166089902807024400e-03],
                                     [5.334927281277434229e-03, 5.330640787246037661e-03, 5.334636417382157721e-03, 2.166089902807024400e-03, 3.549313585301805841e-03]])}

MODELCOSTS = np.array([2.270000000000000000e+02, 2.300000000000000000e+01,
                       3.100000000000000089e+00, 2.574933675519108594e-04,
                       2.141122975983993665e-06])
# cost = 6*cost if using unnormalized costs


if __name__ == "__main__":
    TEST_CASE = 'latitude'

    # Generate variances using MXMC
    # TARGET_COSTS = [2500]
    # VARIANCES = calculate_variances_for_target_costs(COVARIANCE[TEST_CASE],
    #                                                  MODELCOSTS,
    #                                                  TARGET_COSTS)
    # VARIANCES.to_csv('results/edl_output_' + TEST_CASE + '.csv')

    # Load variances from previously generated csv
    VARIANCES = pd.read_csv('results/edl_output_' + TEST_CASE + '.csv',
                            index_col=0)

    print_variances_as_table(VARIANCES)
    plot_variances(VARIANCES, plot_type="bar",
                   title='EDL POST2 ' + TEST_CASE)