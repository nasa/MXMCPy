import numpy as np

from mxmc import Optimizer
from mxmc import OutputProcessor
from mxmc import Estimator

from ishigami_model import IshigamiModel


# Samples from a uniform distribution based on Ishigami requirements.
def get_uniform_sample_distribution(num_samples):

    return np.random.uniform(low=-np.pi,
                             high=np.pi,
                             size=[num_samples, 3])


np.random.seed(1)
num_pilot_samples = 10
model_costs = np.array([1, .05, .001])

high_fidelity_model = IshigamiModel(a=5., b=.1, c=4.)
medium_fidelity_model = IshigamiModel(a=4.75, b=.1, c=4.)
low_fidelity_model = IshigamiModel(a=3., b=.9, c=2.)
models = [high_fidelity_model, medium_fidelity_model, low_fidelity_model]

# Step 1: Compute model outputs for pilot samples.
pilot_inputs = get_uniform_sample_distribution(num_pilot_samples)
pilot_outputs = list()
for model in models:
    pilot_outputs.append(model.evaluate(pilot_inputs))

# MXMC's OutputProcessor provides a convenient way to compute the
# covariance matrix from the model's pilot outputs.
covariance_matrix = OutputProcessor.compute_covariance_matrix(pilot_outputs)

# Step 2: Perform sample allocation optimization.
target_cost = 10000
variance_results = dict()
sample_allocation_results = dict()

# MXMC's Optimizer computes optimal sample allocation and can be used with
# a variety of algorithms. Here we test every algorithm available in order
# to find the algorithm that produces the lowest variance given the pilot
# samples' covariance matrix.
mxmc_optimizer = Optimizer(model_costs, covariance_matrix)

algorithms = Optimizer.get_algorithm_names()
for algorithm in algorithms:
    opt_result = mxmc_optimizer.optimize(algorithm, target_cost)
    variance_results[algorithm] = opt_result.variance
    sample_allocation_results[algorithm] = opt_result.allocation

    print("{} method variance: {}".format(algorithm, opt_result.variance))

best_method = min(variance_results, key=variance_results.get)
sample_allocation = sample_allocation_results[best_method]

print("Best method: ", best_method)

# Step 3: Generate input samples for models.
num_total_samples = sample_allocation.num_total_samples
all_samples = get_uniform_sample_distribution(num_total_samples)
model_input_samples = \
    sample_allocation.allocate_samples_to_models(all_samples)

print("MXMC prescribed samples per model: ", end="")
for samples in model_input_samples:
    print("{} ".format(samples.shape[0]), end="")

print("\n")

# Step 4: Compute model outputs for prescribed inputs.
model_outputs = list()
for input_sample, model in zip(model_input_samples, models):
    model_outputs.append(model.evaluate(input_sample))

# Step 5. Form estimator.

# MXMC's Estimator can be used to run the models with the
# given sample allocation to produce an estimate of the
# quantity of interest.
estimator = Estimator(sample_allocation, covariance_matrix)
estimate = estimator.get_estimate(model_outputs)

print("Estimate = ", estimate)
