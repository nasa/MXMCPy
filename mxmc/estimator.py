import numpy as np


class Estimator:

    def __init__(self, allocation, covariance):
        self._allocation = allocation
        self._covariance = covariance
        self._num_models = self._allocation.num_models
        self._num_samples_per_model = \
            self._allocation.get_number_of_samples_per_model()
        if covariance is not None:
            self._validation(covariance)

        self._cov_delta_delta, self._cov_q_delta = \
            self._calculate_cov_delta_terms()
        self._alpha = self._calculate_alpha()
        self.approximate_variance = self._get_approximate_variance()

    def _validation(self, covariance):
        if len(covariance) != self._num_models:
            raise ValueError("Covariance and allocation dimensions must match")
        if not np.allclose(covariance.transpose(), covariance):
            raise ValueError("Covariance array must be symmetric")

    def get_estimate(self, model_outputs):
        self._validate_model_outputs(model_outputs)
        q = np.mean(model_outputs[0])
        for i in range(1, self._allocation.num_models):
            filt_1, filt_2 = self._allocation.get_sample_split_for_model(i)
            q_i1 = model_outputs[i][filt_1]
            q_i2 = model_outputs[i][filt_2]
            if len(q_i1) != 0 or len(q_i2) != 0:
                q += self._alpha[i - 1] * (np.mean(q_i1) - np.mean(q_i2))

        return q

    def _calculate_cov_delta_terms(self):
        k_0 = self._allocation.get_k0_matrix()
        k = self._allocation.get_k_matrix()
        cov_q_delta = k_0 * self._covariance[0, 1:]
        cov_delta_delta = k * self._covariance[1:, 1:]
        return cov_delta_delta, cov_q_delta

    def _get_approximate_variance(self):
        n_0 = self._allocation.get_number_of_samples_per_model()[0]
        var_q0 = self._covariance[0, 0]

        variance = var_q0 / n_0 \
                   + self._alpha.dot(self._cov_delta_delta.dot(self._alpha)) \
                   + 2 * self._alpha.dot(self._cov_q_delta)

        return variance

    def _calculate_alpha(self):
        if self._allocation.method != "mlmc":
            alpha = self._calculate_acv_alphas()
        else:
            alpha = - np.ones(self._num_models - 1)
        return alpha

    def _calculate_acv_alphas(self):
        k_indices = [i - 1 for i in self._allocation.utilized_models if i != 0]
        temp_cov_delta_delta = self._cov_delta_delta[k_indices][:, k_indices]
        temp_cov_q_delta = self._cov_q_delta[k_indices]
        alpha = np.zeros(self._num_models - 1)
        alpha[k_indices] = - np.linalg.solve(temp_cov_delta_delta,
                                             temp_cov_q_delta)
        return alpha

    def _validate_model_outputs(self, model_outputs):
        if len(model_outputs) != self._num_models:
            raise ValueError("Number of models in model output did not match "
                             "the number in sample allocation")
        for outputs, num_samps in zip(model_outputs,
                                      self._num_samples_per_model):
            if len(outputs) != num_samps:
                raise ValueError("Number of outputs per model does not match "
                                 "the sample allocation")
