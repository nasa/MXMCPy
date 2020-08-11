.. _sourcecode-section:

Source Code Documentation
=========================

Documentation for the core MXMC classes.

Optimizer Module 
------------------------------

.. automodule:: optimizer
.. autoclass:: Optimizer
	:members:

Sample Allocation Module
-------------------------------

.. module:: sample_allocations

.. class:: SampleAllocation

.. module:: sample_allocations.sample_allocation_base
.. automethod:: SampleAllocationBase.__init__
.. automethod:: SampleAllocationBase.get_number_of_samples_per_model
.. automethod:: SampleAllocationBase.get_sample_indices_for_model
.. automethod:: SampleAllocationBase.allocate_samples_to_models

Output Processor Module 
-------------------------------------

.. automodule:: output_processor
.. autoclass:: OutputProcessor
	:members:

Estimator Module
------------------------------

.. automodule:: estimator
.. autoclass:: Estimator
	:members:

Utilities Module
------------------------------

.. module:: util.generic_numerical_optimization
.. method:: perform_slsqp_then_nelder_mead
.. method:: perform_slsqp
.. method:: perform_nelder_mead

.. module:: util.read_sample_allocation
.. method:: read_sample_allocation

.. module:: util.sample_modification
.. method:: adjust_sample_allocation_to_cost
