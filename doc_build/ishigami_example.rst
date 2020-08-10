
End-to-End Example 
=============================

This example provides a simple demonstration of ``MXMCPy`` functionality. Here, the high-fidelity model considered is the Ishigami function, which is frequently used to test methods for uncertainty quantification:

.. math::
    f^{(1)}\ =\ sin(z_1)\ +\ 5\ sin^{2}(z_2)\ +\ \frac{1}{10}\ z_3^{4} sin(z_1)
.. math::
    with\ \ z_i \sim \textit{U} (-\pi,\ \pi)

The following two correlated functions [1] are treated as low-fidelity models to facilitate the use of multi-model estimators:

.. math::
    f^{(2)}\ =\ sin(z_1)\ +\ 4.75\ sin^{2}(z_2)\ +\ \frac{1}{10}\ z_3^{4} sin(z_1)
.. math::
    f^{(3)}\ =\ sin(z_1)\ +\ 3\ sin^{2}(z_2)\ +\ \frac{9}{10}\ z_3^{2} sin(z_1)

Furthermore, the costs associated with evaluating the functions :math:`f^{(1)}`,  :math:`f^{(2)}`, and  :math:`f^{(3)}` are assumed to be 1.0, 0.05, 0.001 seconds, respectively. The goal is to estimate :math:`E[f^{(1)}]` using ``MXMCPy`` by leveraging all three models and comparing with the analytical solution, :math:`E[f^{(1)}] = 2.5`.

This example covers each step for utilizing ``MXMCPy``: estimating the covariance of
model outputs from pilot samples, performing the sample allocation optimization,
and forming an estimator from outputs generated from each model according to
the optimal sample allocation.

The full source code for this example can be found in the ``MXMCPy`` repository:

``/examples/ishigami/run_ishigami.py``


Step 1: Compute model outputs for pilot samples
-----------------------------------------------

In order to determine the optimal sample allocation across the available models, ``MXMCPy`` needs the cost of each model as well as the covariance matrix of the outputs from each model. If the covariance matrix is unknown, it can be estimated from pilot samples, demonstrated here.

Starting from the beginning, the necessary Python modules are imported, including ``MXMCPy`` classes and Numpy:

.. code-block:: python

    import numpy as np

    from mxmc import Optimizer
    from mxmc import OutputProcessor
    from mxmc import Estimator

It is assumed that a user-defined class (``IshigamiModel``) implementing the Ishigami models and function (``get_uniform_sample_distribution``) for sampling the uniform random inputs are available. Three IshigamiModels are
then instantiated per the parameters in the equations above and
stored in a list variable named "models":

.. code-block:: python

    from ishigami_model import IshigamiModel

    num_pilot_samples = 10
    model_costs = np.array([1, .05, .001])

    high_fidelity_model =   IshigamiModel(a=5.,   b=.1, c=4.)
    medium_fidelity_model = IshigamiModel(a=4.75, b=.1, c=4.)
    low_fidelity_model =    IshigamiModel(a=3.,   b=.9, c=2.)

    models = [high_fidelity_model, medium_fidelity_model, low_fidelity_model]

Ten pilot samples are computed with each model as follows:

.. code-block:: python

    pilot_inputs = get_uniform_sample_distribution(num_pilot_samples)
    pilot_outputs = list()
    for model in models:
        pilot_outputs.append(model.evaluate(pilot_inputs))


The covariance matrix is then computed using the ``MXMCPy`` ``OutputProcessor`` class:

.. code-block:: python

    covariance_matrix = OutputProcessor.compute_covariance_matrix(pilot_outputs)

At this point the necessary elements for computing optimal sample allocation are
now available: model costs, pilot outputs, and a covariance matrix.

Step 2: Perform sample allocation optimization
--------------------------------------------------------------------------

The sample allocation optimization with ``MXMCPy`` is now performed assuming a computational budget or target cost of 10000 seconds. 

In the below snippet taken from the example code, all available optimization
algorithms are individually tested to find the method that produces the lowest
estimator variance. The Optimizer.optimize() method returns an instance of the ``OptimizationResult`` class with attributes for estimatore variance and optimal sample allocation. The sample allocation with the lowest variance will be used to generate an estimator in the subsequent steps.

.. code-block:: python

    target_cost = 10000
    variance_results = dict()
    sample_allocation_results = dict()

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


The ``Optimizer`` class also provides functionality for determining an optimal
subset of the models via the boolean parameter auto_model_selection of the
Optimizer.optimize() method. By default, all provided models are used. Note that enabling this option could take considerably longer as every
combination of the models will be tested.

.. code-block:: python

    mxmc_optimizer = Optimizer(model_costs,
                               covariance_matrix,
                               auto_model_selection=True)

    opt_result = mxmc_optimizer.optimize(algorithm, target_cost)
    variance_results = opt_result.variance
    sample_allocation_results = opt_result.allocation


Step 3: Generate input samples for models
--------------------------------------------------------------

The ``SampleAllocation`` class provides functionality for determining how many random input samples are needed and how they are to be allocated across the available models. This is demonstrated below for the optimal sample allocation object found in the previous step:

.. code-block:: python

    num_total_samples = sample_allocation.num_total_samples
    all_samples = get_uniform_sample_distribution(num_total_samples) # User code.
    model_input_samples = sample_allocation.allocate_samples_to_models(all_samples)

The num_total_samples property can be referenced for creation of an ndarray of
inputs, which can then be provided to the allocate_samples_to_models() method.
This method will redistribute the ndarray of input samples into a list of
ndarrays, each containing the prescribed number of samples for each model.

Step 4: Compute model outputs for prescribed inputs
---------------------------------------------------------------

Now that the input samples for each model have been generated and allocated, 
outputs are generated by evaluating the Ishigami models as follows:

.. code-block:: python

    model_outputs = list()
    for input_sample, model in zip(model_input_samples, models):
        model_outputs.append(model.evaluate(input_sample))

The outputs are stored in a list of ndarrays corresponding to each model.

Note that for the practical case where evaluating the available models is time consuming and must be done in an `offline` fashion, the ``SampleAllocation`` class has functionality to save and load to/from disc. This way, the optimal sample allocation can be saved after Step 2 above and then loaded to generate an estimator in Step 5 next.

Step 5: Form estimator
--------------------------------------------------

Finally, an estimator for :math:`E[f^{(1)}]`` is computed using the ``Estimator`` class and the model outputs from the previous step:

.. code-block:: python

    estimator = Estimator(sample_allocation, covariance_matrix)
    estimate = estimator.get_estimate(model_outputs)

    print("Estimate = ", estimate)

Note that the sample allocation object from Step 2 and the covariance matrix from Step 1 are required here. The covariance matrix could also be updated at this point using the model outputs generated in the previous step. The ``MXMCPy`` estimator is close to the true value of 2.5.



| [1] "High-dimensional and higher-order multifidelity Monte Carlo estimators" (Quaglino, 2018)

