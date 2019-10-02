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

        k0 = self._allocation.get_k0_matrix()
        k = self._allocation.get_k_matrix()
        cov_q_delta = k0 * self._covariance[0, 1:]
        cov_delta_delta = k * self._covariance[1:, 1:]

        alpha = - np.linalg.solve(cov_delta_delta, cov_q_delta)

        Q = np.mean(model_outputs[0])
        for i in range(1, self._allocation.num_models):
            filt_1, filt_2 = self._allocation.get_sample_split_for_model(i)
            Q_i1 = model_outputs[i][filt_1]
            Q_i2 = model_outputs[i][filt_2]
            if len(Q_i1) != 0 or len(Q_i2) != 0:
                Q += alpha[i-1] * (np.mean(Q_i1) - np.mean(Q_i2))

        return Q




