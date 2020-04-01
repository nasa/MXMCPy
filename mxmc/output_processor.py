from functools import reduce

import numpy as np
import pandas as pd


class OutputProcessor:
    '''
    Class to estimate covariance matrix from collection of output samples from
    multiple models. Handles the general case where each models' outputs were
    not computed from the same set of random inputs and pairwise overlaps
    between samples are identified using a SampleAllocation object.
    '''
    def __init__(self):
        pass

    @staticmethod
    def compute_covariance_matrix(model_outputs, sample_allocation=None):
        '''
        Estimate covariance matrix from collection of output samples from
        multiple models. In the simple case, the outputs for each model were
        generated from the same collection of inputs and covariance can be
        straightforwardly computed. In the general case, the outputs were not
        computed from the same inputs and a SampleAllocation object must be
        supplied to identify overlap between samples to compute covariance.

        :param model_outputs: list of arrays of outputs for each model. Each
            array must be the same size unless a sample allocation is provided.
        :type model_outputs: list of np.array
        :param sample_allocation: An MXMC sample allocation object defining the
            indices of samples that each model output was generated for, if
            applicable. Default is None indicating that all supplied model
            output arrays are the same size and were generated for the same
            inputs.
        :type sample_allocation: SampleAllocation object.

        :Returns: covariance matrix among all model outputs (2D np.array with
            size equal to the number of models).
        '''
        output_df = OutputProcessor._build_output_dataframe(model_outputs,
                                                            sample_allocation)

        cov_matrix = OutputProcessor. \
            _initialize_matrix_diagonal_with_variances(output_df)
        cov_matrix = OutputProcessor. \
            _compute_offdiagonal_elements(cov_matrix, output_df)

        return cov_matrix

    @staticmethod
    def _build_output_dataframe(model_outputs, sample_allocation):

        if sample_allocation is None:
            output_df = pd.DataFrame(model_outputs)
        else:
            output_df = \
                OutputProcessor. \
                _build_output_df_from_allocation(model_outputs,
                                                 sample_allocation)
        return output_df

    @staticmethod
    def _initialize_matrix_diagonal_with_variances(output_df):

        return np.diag([np.var(row, ddof=1)
                        for _, row in output_df.iterrows()])

    @staticmethod
    def _build_output_df_from_allocation(model_outputs, sample_alloc):

        sub_dfs = []
        for model_index, outputs in enumerate(model_outputs):

            alloc = sample_alloc.get_sample_indices_for_model(model_index)
            sub_dfs.append(pd.DataFrame({model_index: outputs,
                                         'alloc_indices': alloc}))

        def merge_func(x, y):
            return x.merge(y, how='outer')

        output_df = reduce(merge_func, sub_dfs).set_index('alloc_indices')
        output_df.sort_index(inplace=True)
        return output_df.T

    @staticmethod
    def _compute_offdiagonal_elements(matrix, output_df):

        num_models = output_df.shape[0]
        for i in range(num_models - 1):

            for j in range(i + 1, num_models):

                model_out_pair = output_df.loc[[i, j]].dropna(axis=1)

                model_out_i = model_out_pair.loc[i].values
                model_out_j = model_out_pair.loc[j].values

                if len(model_out_i) <= 1:
                    element_matrix = np.nan
                else:
                    element_matrix = np.cov(model_out_i, model_out_j)[0, 1]

                matrix[i, j] = element_matrix
                matrix[j, i] = element_matrix

        return matrix
