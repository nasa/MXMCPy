import numpy as np
import pandas as pd

class OutputProcessor():

    def compute_covariance_matrix(model_outputs):
        if len(model_outputs) == 1 and model_outputs[0].size == 1:
            return np.array([np.nan])

        if len(model_outputs) == 0:
            return np.array([])

        if len({model_output.size for model_output in model_outputs}) == 1:
            return np.cov(model_outputs)

        output_df = pd.DataFrame(model_outputs)

        cov = np.diag([np.var(row) for _, row in output_df.iterrows()])

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
