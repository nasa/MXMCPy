import numpy as np

from .estimator_base import EstimatorBase


class ACVEstimator(EstimatorBase):
    """
    Class to create ACV estimators given an optimal sample allocation and
    outputs from high & low fidelity models.

    :param allocation: ACVSampleAllocation object defining the optimal sample
            allocation using an ACV optimizer.
    :type allocation: ACVSampleAllocation object
    :param covariance: Covariance matrix defining covariance among all
            models being used for estimator. Size MxM where M is # models.
    :type covariance: 2D np.array
    """
    def __init__(self, allocation, covariance):
        super().__init__(allocation, covariance)
        self._cov_delta_delta, self._cov_q_delta = \
            self._calculate_cov_delta_terms()
        self._alpha = self._calculate_alpha()

    def get_estimate(self, model_outputs):
        self._validate_model_outputs(model_outputs)
        q = np.mean(model_outputs[0])
        for i in range(1, self._allocation.num_models):
            ranges_1, ranges_2 = self._allocation.get_sample_split_for_model(i)
            n_1 = sum([len(i) for i in ranges_1])
            n_2 = sum([len(i) for i in ranges_2])
            for rng in ranges_1:
                q += self._alpha[i - 1] * np.sum(model_outputs[i][rng]) / n_1
            for rng in ranges_2:
                q -= self._alpha[i - 1] * np.sum(model_outputs[i][rng]) / n_2

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
        k_indices = [i - 1 for i in self._allocation.utilized_models if i != 0]
        temp_cov_delta_delta = self._cov_delta_delta[k_indices][:, k_indices]
        temp_cov_q_delta = self._cov_q_delta[k_indices]
        alpha = np.zeros(self._allocation.num_models - 1)
        alpha[k_indices] = - np.linalg.solve(temp_cov_delta_delta,
                                             temp_cov_q_delta)
        return alpha
