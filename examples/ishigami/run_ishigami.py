import numpy as np

from mxmc import Optimizer
from mxmc import OutputProcessor
from mxmc import Estimator

from ishigami_model import IshigamiModel


# Acquires a uniform sample distribution based on Ishigami requirements.
def get_uniform_sample_distribution(num_samples):

    return np.random.uniform(low=-np.pi,
                             high=np.pi,
                             size=[num_samples, 3])


np.random.seed(1)
num_pilot_samples = 10
model_costs = np.array([1, .05, .001])

high_fidelity_model = IshigamiModel(a=5., b=.1)
medium_fidelity_model = IshigamiModel(a=4.75, b=.1)
low_fidelity_model = IshigamiModel(a=3., b=.9)
models = [high_fidelity_model, medium_fidelity_model, low_fidelity_model]

# Step 1: Compute model outputs for pilot samples.
pilot_inputs = get_uniform_sample_distribution(num_pilot_samples)
pilot_outputs = list()
for model in models:
    pilot_outputs.append(model.evaluate(pilot_inputs))

# Get covariance matrix from model outputs.
covariance_matrix = OutputProcessor.compute_covariance_matrix(pilot_outputs)

# Step 2: Perform sample allocation optimization.
target_cost = 10000
variance_results = dict()
sample_allocation_results = dict()

mxmc_optimizer = Optimizer(model_costs, covariance_matrix)

for algorithm in ["acvmf", "acvkl", "grdmr"]:
    opt_result = mxmc_optimizer.optimize(algorithm, target_cost)
    variance_results[algorithm] = opt_result.variance
    sample_allocation_results[algorithm] = opt_result.allocation

best_method = min(variance_results, key=variance_results.get)
sample_allocation = sample_allocation_results[best_method]

print("Best method = ", best_method)

# Step 3: Generate input samples for models.
all_samples = get_uniform_sample_distribution(sample_allocation.num_total_samples)
model_input_samples = sample_allocation.allocate_samples_to_models(all_samples)

# Step 4: Compute model outputs for prescribed inputs.
model_outputs = list()
for input_sample, model in zip(model_input_samples, models):
    model_outputs.append(model.evaluate(input_sample))

# Step 5. Form estimator.
estimator = Estimator(sample_allocation, covariance_matrix)
estimate = estimator.get_estimate(model_outputs)

print("estimate = ", estimate)
