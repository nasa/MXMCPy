import numpy as np

from mxmc import Optimizer
from mxmc import OutputProcessor
from mxmc import Estimator

from spring_mass_model import SpringMassModel

np.random.seed(1)


def get_sample_beta_distribution(num_samples):

    shift = 1.0
    scale = 2.5
    alpha = 3.
    beta = 2.
    return shift + scale * np.random.beta(alpha, beta, num_samples)


num_pilot_samples = 10

model_costs = np.array([100, 10, 1])
model_hifi = SpringMassModel(time_step=0.001)
model_medfi = SpringMassModel(time_step=0.01)
model_lofi = SpringMassModel(time_step=1)

#Step 1a) - run pilot samples / get pilot outputs
pilot_inputs = get_sample_beta_distribution(num_pilot_samples)
pilot_outputs_hifi = np.zeros(num_pilot_samples)
pilot_outputs_medfi = np.zeros(num_pilot_samples)
pilot_outputs_lofi = np.zeros(num_pilot_samples)

for i, pilot_input in enumerate(pilot_inputs):
    pilot_outputs_hifi[i] = model_hifi.evaluate([pilot_input]) 
    pilot_outputs_medfi[i] = model_medfi.evaluate([pilot_input])
    pilot_outputs_lofi[i] = model_lofi.evaluate([pilot_input])

#Step 1b) - get covariance matrix
pilot_outputs = [pilot_outputs_hifi, pilot_outputs_medfi, pilot_outputs_lofi]
covariance_matrix = OutputProcessor.compute_covariance_matrix(pilot_outputs)

#Step 2) - perform variance minimization optimization for select algorithms:
algorithms = ["acvmf", "acvkl", "grdmr"]
target_cost = 10000
variance_results = {}
sample_allocation_results = {}

mxmc_optimizer = Optimizer(model_costs, covariance_matrix)

for algorithm in algorithms:
    opt_result = mxmc_optimizer.optimize(algorithm, target_cost)
    variance_results[algorithm] = opt_result.variance
    sample_allocation_results[algorithm] = opt_result.allocation

best_method = min(variance_results, key=variance_results.get)
sample_allocation = sample_allocation_results[best_method]

print("Best method = ", best_method)

#Step 3) Generate input samples for models
all_samples = get_sample_beta_distribution(sample_allocation.num_total_samples)
model_input_samples = sample_allocation.allocate_samples_to_models(all_samples)

#Step 4) Run models with prescribed inputs, store outputs
input_samples_hifi = model_input_samples[0]
outputs_hifi = np.zeros(len(input_samples_hifi))
for i, input_hifi in enumerate(input_samples_hifi):
    outputs_hifi[i] = model_hifi.evaluate([input_hifi])

input_samples_medfi = model_input_samples[1]
outputs_medfi = np.zeros(len(input_samples_medfi))
for i, input_medfi in enumerate(input_samples_medfi):
    outputs_medfi[i] = model_medfi.evaluate([input_medfi])

input_samples_lofi = model_input_samples[2]
outputs_lofi = np.zeros(len(input_samples_lofi))
for i, input_lofi in enumerate(input_samples_lofi):
    outputs_lofi[i] = model_lofi.evaluate([input_lofi])

#Step 5) Make an estimate
model_outputs = [outputs_hifi, outputs_medfi, outputs_lofi]
estimator = Estimator(sample_allocation, covariance_matrix)
estimate = estimator.get_estimate(model_outputs)
print("estimate = ", estimate)



