import numpy as np

class MBLUEEstimator:
    '''
    :param allocation: SampleAllocation object defining the optimal sample
            allocation using an MXMX optimizer.
    :type allocation: SampleAllocation object
    :param covariance: Covariance matrix defining covariance among all
            models being used for estimator. Size MxM where M is # models.
    :type covariance: 2D np.array
    '''
    def __init__(self, allocation, covariance):

        self._allocation = allocation
        self._covariance = covariance
        self._num_models = self._allocation.num_models
        self._num_samples_per_model = \
            self._allocation.get_number_of_samples_per_model()
        if covariance is not None:
            self._validation(covariance)

        self._approximate_variance = None

    @property
    def approximate_variance(self):
        if self._approximate_variance is None:
            self._approximate_variance = self._get_approximate_variance()
        return self._approximate_variance

    def _validation(self, covariance):
        if len(covariance) != self._num_models:
            raise ValueError("Covariance and allocation dimensions must match")
        if not np.allclose(covariance.transpose(), covariance):
            raise ValueError("Covariance array must be symmetric")

    def get_estimate(self, model_outputs):
        '''
        '''
        pass

    def _get_approximate_variance(self):
        pass

    def _validate_model_outputs(self, model_outputs):
        if len(model_outputs) != self._num_models:
            raise ValueError("Number of models in model output did not match "
                             "the number in sample allocation")
        for outputs, num_samps in zip(model_outputs,
                                      self._num_samples_per_model):
            if len(outputs) != num_samps:
                raise ValueError("Number of outputs per model does not match "
                                 "the sample allocation")
            if len(outputs.shape) > 1 and outputs.shape[1] != 1:
                raise ValueError("Estimators are not currently implemented "
                                 "for multiple outputs")
