import numpy as np
from mxmc import Optimizer


# Model properties
model_costs = np.array([1, 0.1, 0.01])

# Covariance matrices of two QOIs, these could be estimated from pilot samples
covariance_qoi1 = np.array([[1.0, 0.9, 0.8],
                            [0.9, 1.6, 0.7],
                            [0.8, 0.7, 2.5]])
covariance_qoi2 = np.array([[1.0, 0.9, 0.85],
                            [0.9, 1.6, 0.65],
                            [0.85, 0.65, 2.5]])

# Combine covariance matrices into 3d numpy array
combined_covariance = np.empty((3, 3, 2))
combined_covariance[:, :, 0] = covariance_qoi1
combined_covariance[:, :, 1] = covariance_qoi2

# Use combined covariance matrix and proceed with optimization as normal
optimizer = Optimizer(model_costs, combined_covariance)
result = optimizer.optimize(algorithm="acvmf", target_cost=10000)

print("Variance for QOI 1:", result.variance[0])
print("Variance for QOI 2:", result.variance[1])
