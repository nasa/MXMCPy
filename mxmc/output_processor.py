from functools import reduce

import numpy as np
import pandas as pd


class OutputProcessor():

    def __init__(self):
        pass

    def compute_covariance_matrix(self, model_outputs, sample_allocation=None):
        output_df = self._build_output_dataframe(model_outputs,
                                                 sample_allocation)
        cov_matrix = self._initialize_matrix_diagonal_with_variances(output_df)
        cov_matrix = self._compute_offdiagonal_elements(cov_matrix, output_df)
        return cov_matrix

    def _build_output_dataframe(self, model_outputs, sample_allocation):
        if sample_allocation is None:
            output_df = pd.DataFrame(model_outputs)
        else:
            output_df = self._build_output_df_from_allocation(model_outputs,
                                                              sample_allocation)
        return output_df

    @staticmethod
    def _initialize_matrix_diagonal_with_variances(output_df):
        return np.diag([np.var(row, ddof=1) for _, row in output_df.iterrows()])

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
