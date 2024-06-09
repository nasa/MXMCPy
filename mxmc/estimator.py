from mxmc.estimators.acv_estimator import ACVEstimator
from mxmc.estimators.mlmc_estimator import MLMCEstimator
from mxmc.sample_allocations.acv_sample_allocation import ACVSampleAllocation
from mxmc.sample_allocations.mlmc_sample_allocation import MLMCSampleAllocation
from mxmc.sample_allocations.sample_allocation_base import SampleAllocation  # Assuming this is the base class for all sample allocations
import numpy as np

ALLOCATION_TO_ESTIMATOR_MAP = {
    ACVSampleAllocation: ACVEstimator,
    MLMCSampleAllocation: MLMCEstimator
}

class Estimator:
    """
    Class to create MXMC estimators given an optimal sample allocation and
    outputs from high & low fidelity models.

    :param allocation: SampleAllocation object defining the optimal sample
            allocation using an MXMC optimizer.
    :type allocation: SampleAllocation
    :param covariance: Covariance matrix defining covariance among all
            models being used for estimator. Size MxM where M is # models.
    :type covariance: np.ndarray
    :raises TypeError: If allocation is not an instance of SampleAllocation
    :raises ValueError: If covariance is not a square matrix or its dimensions
                        do not match the number of models.
    """

    def __new__(cls, allocation: SampleAllocation, covariance: np.ndarray):
        if not isinstance(allocation, SampleAllocation):
            raise TypeError("allocation must be an instance of SampleAllocation")
        if not isinstance(covariance, np.ndarray):
            raise TypeError("covariance must be a numpy array")
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be a square matrix")
        if allocation.num_models != covariance.shape[0]:
            raise ValueError("Dimensions of covariance matrix must match the number of models")

        estimator_type = ALLOCATION_TO_ESTIMATOR_MAP.get(allocation.__class__)
        if estimator_type is None:
            raise ValueError(f"No estimator found for allocation type {type(allocation).__name__}")

        return estimator_type(allocation, covariance)
