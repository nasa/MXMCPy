Vector-valued Quantities of Interest
====================================

``MXMCPy`` allows for the consideration of multiple quantities of interest (QOIs) during sample allocation optimization.  In this case, ``MXMCPy`` will minimize the sum of variance of the individual QOI components as seen in [1].  MLMC is an exception to this rule.  When considering multiple QOIs with MLMC, the maximum variance among the QOIs is minimized as in [2].

Performing this type of vector-valued sample allocation optimization is easy in ``MXMCPy``.  Simply, use a 3-dimensional covariance array in the creation of an ``Optimizer``.  The third dimension of the array is used for the different QOIs.  So in the case where there are *N* QOIs and *M* models, the shape of a covariance array should be *N* x *N* x *M*.  The example below uses 3 models with two QOIs:

.. code-block:: python

    covariance = np.empty((3, 3, 2))
    covariance[:, :, 0] = np.array([[1.0, 0.9, 0.8],
                                    [0.9, 1.6, 0.7],
                                    [0.8, 0.7, 2.5]])
    covariance[:, :, 1] = np.array([[1.0, 0.9, 0.85],
                                    [0.9, 1.6, 0.65],
                                    [0.85, 0.65, 2.5]])

    mxmc_optimizer = Optimizer(model_costs, covariance)
    optimization_result = mxmc_optimizer.optimize(algorithm, target_cost)

The optimization result of a vector-valued optimization will contain an ``approximate_variance`` array which indicates the approximate variances for each of the QOIs.


| [1] Quaglino, A., Pezzuto, S., & Krause, R. (2019). High-dimensional and higher-order multifidelity Monte Carlo estimators. Journal of Computational Physics, 388, 300-315.
| [2] Giles, M. B. (2015). Multilevel monte carlo methods. Acta Numerica, 24, 259.
