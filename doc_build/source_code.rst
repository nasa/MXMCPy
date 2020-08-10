.. _sourcecode-section:

Source Code Documentation
=========================

Documentation for the core MXMC classes.

Optimizer Module 
------------------------------

.. automodule:: optimizers.optimizer_base
.. autoclass:: OptimizerBase
	:members:

Sample Allocation Module
-------------------------------

.. automodule:: sample_allocations.sample_allocation_base

.. autoclass:: SampleAllocationBase
	:members:

.. automodule:: sample_allocations.acv_sample_allocation
.. autoclass:: ACVSampleAllocation
	:members:

.. automodule:: sample_allocations.mlmc_sample_allocation
.. autoclass:: MLMCSampleAllocation
	:members:

Estimator Module
------------------------------

.. automodule:: estimator
.. autoclass:: Estimator
	:members:

Output Processor Module 
-------------------------------------

.. automodule:: output_processor
.. autoclass:: OutputProcessor
	:members:


Utility Module
------------------------------

.. automodule:: util.generic_numerical_optimization
.. autofunction:: perform_slsqp_then_nelder_mead
.. autofunction:: perform_slsqp
.. autofunction:: perform_nelder_mead

.. automodule:: util.read_sample_allocation
.. autofunction:: read_sample_allocation

.. automodule:: util.testing
.. autofunction:: assert_opt_result_equal
