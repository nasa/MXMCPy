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

	Base class for managing the allocations of random input samples (model
	evaluations) across available models. Provides a user with number of
	model evaluations required to generate an estimator and how to partition
	input samples to do so, after the sample allocation optimization problem
	is solved.

	:param compressed_allocation: a two dimensional array completely
		describing a MXMC sample allocation; returned by each optimizer
		class as the result of sample allocation optimization. See docs
		for a description and example of the format.
	:type compressed_allocation: 2D np.array

	:ivar num_total_samples: number of total input samples needed across all
		available models
	:ivar _utilized_models: list of indices corresponding to models with
		samples allocated to them
	:ivar num_models: total number of available models

	.. method:: get_number_of_samples_per_model

		Returns the total number of samples allocated to each available model
		:type: (list of integers)

	.. method:: get_sample_indices_for_model

		:param model_index: index of model to return indices for (from 0 to
			#models-1)
		:type model_index: int

		Returns binary array with indices of samples required by specified model

		:type: (np.array with length of num_total_samples)

	.. method:: allocate_samples_to_models

		Allocates a given array of all input samples across all available
		models according to the sample allocation determined by an MXMX
		optimizer.

		:param all_samples: array of user-generated random input samples with
			length equal to num_total_samples
		:type all_samples: 2D np.array

		Returns individual arrays of input samples for all available models
		:type: (list of np.arrays with length equal to num_models)

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
