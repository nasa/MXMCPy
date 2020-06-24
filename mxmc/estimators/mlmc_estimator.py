import numpy as np

from .acv_estimator import ACVEstimator

class MLMCEstimator(ACVEstimator):

    def _calculate_alpha(self):
        return -1.0*np.ones(self._num_models - 1)
