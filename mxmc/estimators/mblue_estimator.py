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
        self._psi_inverse = None

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

        if self._psi_inverse is None:
            self._psi_inverse = self._get_psi_inverse()
        output_vector = self._build_y_vector(model_outputs)
        estimate = np.dot(self._psi_inverse[0, :], output_vector)
        return estimate

    def _get_approximate_variance(self):
        
        if self._psi_inverse is None:
            self._psi_inverse = self._get_psi_inverse()
        return self._psi_inverse[0, 0]

    def _get_psi_inverse(self):
        psi_matrix = self._build_psi_matrix()
        psi_inverse = np.linalg.inv(psi_matrix)
        return psi_inverse

    def _build_psi_matrix(self):
        #equation 2.6 for psi matrix
        psi = np.zeros((self._num_models, self._num_models))
        
        for row in self._allocation.compressed_allocation:
            num_samples = row[0]
            model_indices = row[1:]
            cov_sub_mat = self._get_covariance_sub_matrix(model_indices)
            restrict_mat = self._get_restriction_matrix(model_indices)
            prolong_mat = restrict_mat.T
            P_k_times_C_k_inv = np.dot(prolong_mat, np.linalg.inv(cov_sub_mat))
            psi += num_samples * np.dot(P_k_times_C_k_inv, restrict_mat)
        
        return psi

    def _build_y_vector(self, model_outputs):
        #equation 2.6 for y vector
        y_vector = np.zeros(self._num_models)
        summed_inds = np.zeros(self._num_models, dtype=int)
        for row in self._allocation.compressed_allocation:
            num_samples = row[0]
            if num_samples == 0:
                continue
            model_inds = row[1:]
            cov_sub_mat = self._get_covariance_sub_matrix(model_inds)
            prolong_mat = self._get_restriction_matrix(model_inds).T
            output_sum = self._get_summed_outputs_for_model_group(model_outputs,
                                                                  num_samples,
                                                                  model_inds,
                                                                  summed_inds)
            summed_inds += model_inds*num_samples
            P_k_times_C_k_inv = np.dot(prolong_mat, np.linalg.inv(cov_sub_mat))
            y_vector += np.dot(P_k_times_C_k_inv, output_sum)

        return y_vector

    def _get_summed_outputs_for_model_group(self, model_outputs, num_samples,
                                            model_inds, start_inds):
        end_inds = start_inds + model_inds*num_samples
        models_in_group = np.nonzero(start_inds-end_inds)[0]
        summed_outputs = np.zeros(len(models_in_group))

        for i, model_ind in enumerate(models_in_group):
            start = start_inds[model_ind]
            end = end_inds[model_ind]
            summed_outputs[i] = np.sum(model_outputs[model_ind][start:end])

        return summed_outputs

    def _get_covariance_sub_matrix(self, model_indices):

        one_indices = np.where(model_indices==1)[0]   
        cov_sub_mat = np.zeros((len(one_indices), len(one_indices)))

        #TODO gotta be a slick way to do this but googling is hard right now:
        for i, index_i in enumerate(one_indices):
            for j, index_j in enumerate(one_indices):
                cov_sub_mat[i,j] = self._covariance[index_i, index_j]

        return cov_sub_mat

    def _get_restriction_matrix(self, model_indices):

        one_indices = np.where(model_indices==1)[0]
    
        restrict_mat = np.zeros((len(one_indices), len(model_indices)))
        for i, one_index in enumerate(one_indices):
            restrict_mat[i, one_index] = 1.0

        return restrict_mat

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
