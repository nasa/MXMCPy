from functools import reduce

import numpy as np
import pandas as pd


class OutputProcessor():

    def __init__(self):
        pass

    def compute_covariance_matrix(self, model_outputs, sample_allocation=None):
        '''
        '''
        offdiag_operator = lambda x, y: np.cov(x, y)[0, 1]
        return self._build_matrix(model_outputs, sample_allocation,
                                  offdiag_operator)

    def compute_vardiff_matrix(self, model_outputs, sample_allocation=None):
        offdiag_operator = lambda x, y: np.var(x - y)
        return self._build_matrix(model_outputs, sample_allocation,
                                  offdiag_operator)

    def _build_matrix(self, model_outputs, sample_allocation, operator):
        output_df = self._build_output_dataframe(model_outputs,
                                                 sample_allocation)
        matrix = self._initialize_matrix_diagonal_with_variances(output_df)
        matrix = self._compute_offdiagonal_elements(matrix, output_df,
                                                    operator)
        return matrix

    def _build_output_dataframe(self, model_outputs, sample_allocation):
        if sample_allocation is None:
            output_df = pd.DataFrame(model_outputs)
        else:
            output_df = self._build_output_df_from_allocation(model_outputs,
                                                              sample_allocation)
        return output_df

    @staticmethod
    def _initialize_matrix_diagonal_with_variances(output_df):
        return np.diag(
                [np.var(row, ddof=1) for _, row in output_df.iterrows()])

    @staticmethod
    def _build_output_df_from_allocation(model_outputs, sample_alloc):

        sub_dfs = []
        for model_index, outputs in enumerate(model_outputs):
            alloc = sample_alloc.get_sample_indices_for_model(model_index)
            sub_dfs.append(pd.DataFrame({model_index: outputs,
                                         'alloc_indices': alloc}))

        merge_func = lambda x, y: x.merge(y, how='outer')
        output_df = reduce(merge_func, sub_dfs).set_index('alloc_indices')
        output_df.sort_index(inplace=True)
        return output_df.T

    @staticmethod
    def _compute_offdiagonal_elements(matrix, output_df, operator):
        num_models = output_df.shape[0]
        for i in range(num_models - 1):

            for j in range(i + 1, num_models):

                model_out_pair = output_df.loc[[i, j]].dropna(axis=1)

                model_out_i = model_out_pair.loc[i].values
                model_out_j = model_out_pair.loc[j].values

                if len(model_out_i) <= 1:
                    element_matrix = np.nan
                else:
                    element_matrix = operator(model_out_i, model_out_j)

                matrix[i, j] = element_matrix
                matrix[j, i] = element_matrix

        return matrix
