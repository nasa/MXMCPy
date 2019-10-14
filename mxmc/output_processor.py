from functools import reduce
import numpy as np
import pandas as pd

class OutputProcessor():

    def __init__(self):
        pass

    def compute_covariance_matrix(self, model_outputs, sample_allocation=None):
        '''
        '''
        if len(model_outputs) == 1 and model_outputs[0].size == 1:
            return np.array([np.nan])

        if len(model_outputs) == 0:
            return np.array([])

        if sample_allocation is None:
            if len({model_output.size for model_output in model_outputs}) == 1:
                return np.cov(model_outputs)

            output_df = pd.DataFrame(model_outputs)

        else:
            output_df = self._assemble_output_df(model_outputs,
                                                 sample_allocation)

        cov = np.diag([np.var(row, ddof=1) for _, row in output_df.iterrows()])

        num_models = len(model_outputs)
        for i in range(num_models - 1):
            for j in range(i + 1, num_models):
                model_out_pair = output_df.loc[[i, j]].dropna(axis=1)

                model_out_i = model_out_pair.loc[i].values
                model_out_j = model_out_pair.loc[j].values

                element_cov = np.cov(model_out_i, model_out_j)[0, 1]
                cov[i, j] = element_cov
                cov[j, i] = element_cov
        return cov

    @staticmethod
    def _assemble_output_df(model_outputs, sample_alloc):

        sub_dfs = []
        for model_index, outputs in enumerate(model_outputs):
            alloc = sample_alloc.get_sample_indices_for_model(model_index)
            sub_dfs.append(pd.DataFrame({model_index: outputs,
                                         'alloc_indices': alloc}))

        merge_func = lambda x, y: x.merge(y, how='outer')
        output_df = reduce(merge_func, sub_dfs).set_index('alloc_indices')
        output_df.sort_index(inplace=True)
        return output_df.T
