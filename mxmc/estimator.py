import numpy as np

from mxmc.estimators.acv_estimator import ACVEstimator
from mxmc.estimators.mblue_estimator import MBLUEEstimator
from mxmc.estimators.mlmc_estimator import MLMCEstimator
from mxmc.sample_allocations.acv_sample_allocation import ACVSampleAllocation
from mxmc.sample_allocations.mblue_sample_allocation import MBLUESampleAllocation
from mxmc.sample_allocations.mlmc_sample_allocation import MLMCSampleAllocation

ALLOCATION_TO_ESTIMATOR_MAP = {ACVSampleAllocation: ACVEstimator,
                               MBLUESampleAllocation: MBLUEEstimator,
                               MLMCSampleAllocation: MLMCEstimator}

class Estimator:
    '''
    Class to create MXMC estimators given an optimal sample allocation and
    outputs from high & low fidelity models.

    :param allocation: SampleAllocation object defining the optimal sample
            allocation using an MXMX optimizer.
    :type allocation: SampleAllocation object
    :param covariance: Covariance matrix defining covariance among all
            models being used for estimator. Size MxM where M is # models.
    :type covariance: 2D np.array
    '''

    def __new__(cls, allocation, covariance):

        estimator_type = ALLOCATION_TO_ESTIMATOR_MAP[allocation.__class__]
        return estimator_type(allocation, covariance)
        
