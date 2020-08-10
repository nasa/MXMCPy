from abc import ABCMeta, abstractmethod
import numpy as np


class EstimatorBase(metaclass=ABCMeta):
    """
    Class to create MXMC estimators given an optimal sample allocation and
    outputs from high & low fidelity models.

    :param allocation: SampleAllocation object defining the optimal sample
            allocation using an MXMX optimizer.
    :type allocation: SampleAllocation object
    :param covariance: Covariance matrix defining covariance among all
            models being used for estimator. Size MxM where M is # models.
    :type covariance: 2D np.array
    """
    def __init__(self, allocation, covariance):

        self._allocation = allocation
        self._covariance = covariance
        self._approximate_variance = None

        if covariance is not None:
            self._validation(covariance)

    @property
    def approximate_variance(self):
        if self._approximate_variance is None:
            self._approximate_variance = self._get_approximate_variance()
        return self._approximate_variance

    def _validation(self, covariance):
        if len(covariance) != self._allocation.num_models:
            raise ValueError("Covariance and allocation dimensions must match")
        if not np.allclose(covariance.transpose(), covariance):
            raise ValueError("Covariance array must be symmetric")

    @abstractmethod
    def get_estimate(self, model_outputs):
        """
        Compute MXMC expected value estimate.

        :param model_outputs: arrays of outputs for each model evaluated at the
            random inputs prescribed by the optimal sample allocation. Note:
            each output array must correspond exactly to the size/order of the
            random inputs given by the optimal SampleAllocation object.
        :type model_outputs: list of np.arrays

        :Returns: the expected value estimator (float)
        """
        raise NotImplementedError

    @abstractmethod
    def _get_approximate_variance(self):
        raise NotImplementedError

    def _validate_model_outputs(self, model_outputs):
        if len(model_outputs) != self._allocation.num_models:
            raise ValueError("Number of models in model output did not match "
                             "the number in sample allocation")
        for outputs, num_samps in zip(model_outputs,
                self._allocation.get_number_of_samples_per_model()):
            if len(outputs) != num_samps:
                raise ValueError("Number of outputs per model does not match "
                                 "the sample allocation")
            if len(outputs.shape) > 1 and outputs.shape[1] != 1:
                raise ValueError("Estimators are not currently implemented "
                                 "for multiple outputs")
