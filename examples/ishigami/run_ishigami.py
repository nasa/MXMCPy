import numpy as np

from mxmc import Optimizer
from mxmc import OutputProcessor
from mxmc import Estimator

from ishigami_model import IshigamiModel


def get_uniform_sample_distribution(num_samples):

    return np.random.uniform(low=-np.pi,
                             high=np.pi,
                             size=num_samples)


np.random.seed(1)

num_pilot_samples = 10

model_costs = np.array([1, .05, .001])
model_hifi = IshigamiModel(a=5., b=.1)
model_medfi = IshigamiModel(a=4.75, b=.1)
model_lofi = IshigamiModel(a=3., b=.9)

# Step 1a) - run pilot samples / get pilot outputs
pilot_inputs = get_uniform_sample_distribution([num_pilot_samples, 3])

pilot_outputs_hifi = np.zeros(num_pilot_samples)
pilot_outputs_medfi = np.zeros(num_pilot_samples)
pilot_outputs_lofi = np.zeros(num_pilot_samples)

pilot_outputs_hifi = model_hifi.evaluate(pilot_inputs)
pilot_outputs_medfi = model_medfi.evaluate(pilot_inputs)
pilot_outputs_lofi = model_lofi.evaluate(pilot_inputs)

# Step 1b) - get covariance matrix
pilot_outputs = [pilot_outputs_hifi, pilot_outputs_medfi, pilot_outputs_lofi]
covariance_matrix = OutputProcessor.compute_covariance_matrix(pilot_outputs)

# Step 2) - perform variance minimization optimization for select algorithms:
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

# Step 3) Generate input samples for models
all_samples = get_uniform_sample_distribution([sample_allocation.num_total_samples, 3])
model_input_samples = sample_allocation.allocate_samples_to_models(all_samples)

# Step 4) Run models with prescribed inputs, store outputs
input_samples_hifi = model_input_samples[0]
outputs_hifi = model_hifi.evaluate(input_samples_hifi)

input_samples_medfi = model_input_samples[1]
outputs_medfi = model_medfi.evaluate(input_samples_medfi)

input_samples_lofi = model_input_samples[2]
outputs_lofi = model_lofi.evaluate(input_samples_lofi)

# Step 5) Make an estimate
model_outputs = [outputs_hifi, outputs_medfi, outputs_lofi]
estimator = Estimator(sample_allocation, covariance_matrix)
estimate = estimator.get_estimate(model_outputs)
print("estimate = ", estimate)



