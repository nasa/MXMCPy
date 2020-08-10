.. _sourcecode-section:

Source Code Documentation
=========================

Documentation for the core MXMC classes.

Optimizer Module 
------------------------------

.. automodule:: optimizers.optimizer_base
.. autoclass:: OptimizerBase
	:members:

.. automodule:: optimizers.mfmc
.. autoclass:: MFMC
	:members:

.. automodule:: optimizers.mlmc
.. autoclass:: MLMC
	:members:

.. automodule:: optimizers.model_selection
.. autoclass:: AutoModelSelection
	:members:

.. automodule:: optimizers.approximate_control_variates.acv_constraints
.. autoclass:: ACVConstraints
	:members:

.. automodule:: optimizers.approximate_control_variates.acv_optimizer
.. autoclass:: ACVOptimizer
	:members:

.. automodule:: optimizers.approximate_control_variates.recursion_enumerator
.. autoclass:: RecursionEnumerator
	:members:

.. automodule:: optimizers.approximate_control_variates.generalized_independent_samples.gis_optimizer
.. autoclass:: GISOptimizer
	:members:

.. automodule:: optimizers.approximate_control_variates.generalized_independent_samples.impl_optimizers
.. autoclass:: ACVIS
	:members:

.. autoclass:: GISSR
	:members:

.. autoclass:: GISMR
	:members:

.. automodule:: optimizers.approximate_control_variates.generalized_multifidelity.gmf_optimizer
.. autoclass:: GMFOptimizer
	:members:

.. automodule:: optimizers.approximate_control_variates.generalized_multifidelity.gmf_ordered
.. autoclass:: GMFOrdered
	:members:

.. automodule:: optimizers.approximate_control_variates.generalized_multifidelity.gmf_unordered
.. autoclass:: GMFUnordered
	:members:

.. automodule:: optimizers.approximate_control_variates.generalized_multifidelity.impl_optimizers
.. autoclass:: ACVKL
	:members:

.. autoclass:: ACVMFMC
	:members:

.. autoclass:: ACVMF
	:members:

.. autoclass:: ACVMFU
	:members:

.. autoclass:: GMFSR
	:members:

.. autoclass:: GMFMR
	:members:

.. automodule:: optimizers.approximate_control_variates.generalized_recursive_difference.grd_optimizer
.. autoclass:: GRDOptimizer
	:members:

.. automodule:: optimizers.approximate_control_variates.generalized_recursive_difference.impl_optimizers
.. autoclass:: GRDMR
	:members:

.. autoclass:: GRDSR
	:members:

.. autoclass:: WRDiff
	:members:

Sample Allocation 
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
