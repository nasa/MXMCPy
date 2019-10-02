import numpy as np


class Estimator:

    def __init__(self, allocation, covariance):
        self._allocation = allocation
        self._covariance = covariance
        if covariance is not None:
            self._validation(covariance)

    def _validation(self, covariance):
        if len(covariance) != self._allocation.num_models:
            raise ValueError("Covariance and allocation dimensions must match")
        if not np.allclose(covariance.transpose(), covariance):
            raise ValueError("Covariance array must be symmetric")

    def get_estimate(self, model_outputs):
        if len(model_outputs) != self._allocation.num_models:
            raise ValueError("Number of models in model output did not match "
                             "the number in sample allocation")
        model_samples = self._allocation.get_number_of_samples_per_model()
        for outputs, num_samps in zip(model_outputs, model_samples):
            if len(outputs) != num_samps:
                raise ValueError("Number of outputs per model does not match "
                                 "the sample allocation")

        return np.mean(model_outputs[0])
