import numpy as np

class OutputProcessor():

    def compute_covariance_matrix(model_outputs):
        if len(model_outputs) == 1 and model_outputs[0].size == 1:
            return np.array([np.nan])

        if len(model_outputs) == 0:
            return np.array([])

        if len({model_output.size for model_output in model_outputs}) == 1:
            return np.cov(model_outputs)

        cov = np.diag([np.var(out, ddof=1) for out in model_outputs])
        num_models = len(model_outputs)
        for i in range(num_models - 1):
            for j in range(i + 1, num_models):
                model_out_i = model_outputs[i]
                model_out_j = model_outputs[j]
                min_length = min(len(model_out_i), len(model_out_j))
                element_cov = np.cov([model_out_i[:min_length],
                                      model_out_j[:min_length]])[0, 1]
                cov[i, j] = element_cov
                cov[j, i] = element_cov
        return cov
