import warnings

import h5py
import numpy as np


class SampleAllocationBase:
    '''
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
    :ivar _utilized_models: list of indices corresponding to models with samples
        allocated to them
    :ivar num_models: total number of available models

    '''
    def __init__(self, compressed_allocation):

        self.compressed_allocation = np.array(compressed_allocation)
        self.num_models = self._calculate_num_models()
        self.num_total_samples = np.sum(self.compressed_allocation[:, 0])
        self._utilized_models = None

    @property
    def utilized_models(self):
        if self._utilized_models is None:
            self._utilized_models = self._find_utilized_models()
        return self._utilized_models

    def get_number_of_samples_per_model(self):
        '''
        :Returns: The total number of samples allocated to each available model
            (list of integers)
        '''
        samples_per_model = np.empty(self.num_models, dtype=int)
        for model_index in range(self.num_models):
            samples_per_model[model_index] = \
                len(self.get_sample_indices_for_model(model_index))

        return samples_per_model

    def get_sample_indices_for_model(self, model_index):
        '''
        :param model_index: index of model to return indices for (from 0 to
            #models-1)
        :type model_index: int

        :Returns: binary array with indices of samples required by the
            specified model (np.array with length of num_total_samples)
        '''
        if model_index == 0:
            ranges = self._get_ranges_from_samples_and_bool(
                self.compressed_allocation[:, 0],
                self.compressed_allocation[:, 1])
        else:
            col_1 = model_index * 2
            col_2 = col_1 + 1
            model_used = np.max(self.compressed_allocation[:, [col_1, col_2]],
                                axis=1)
            ranges = self._get_ranges_from_samples_and_bool(
                self.compressed_allocation[:, 0], model_used)
        if ranges:
            ranges = np.hstack(ranges)

        return ranges

    def allocate_samples_to_models(self, all_samples):
        '''
        Allocates a given array of all input samples across all available
        models according to the sample allocation determined by an MXMX
        optimizer.

        :param all_samples: array of user-generated random input samples with
            length equal to num_total_samples
        :type all_samples: 2D np.array

        :Returns: individual arrays of input samples for all available models
            (list of np.arrays with length equal to num_models)
        '''
        if len(all_samples) < self.num_total_samples:
            raise ValueError("Too few inputs samples to allocate to models!")

        model_samples = []
        for model_index in range(self.num_models):
            sample_indices = self.get_sample_indices_for_model(model_index)
            model_samples.append(all_samples[sample_indices])

        return model_samples

    def _get_num_samples_per_column(self):

        samples_per_group = self.compressed_allocation[:, 0]
        samples_per_col = np.zeros(2*self.num_models-1, dtype=int)

        for i, col_inds in enumerate(self.compressed_allocation[:, 1:].T):
            samples_per_col[i] = np.sum(samples_per_group[col_inds == 1])

        return samples_per_col

    def save(self, file_path):

        h5_file = h5py.File(file_path, 'w')
        self.write_file_data_set(h5_file, "Compressed_Allocation",
                                 self.compressed_allocation)
        h5_file.attrs['Method'] = "ACV"
        h5_file.close()

    @staticmethod
    def write_file_data_set(h5file, group_name, dataset):
        h5file.create_group(group_name)
        h5file[group_name].create_dataset(name=group_name.lower(),
                                          data=dataset)

    @staticmethod
    def _get_ranges_from_samples_and_bool(n_samples, used_by_samples):
        ranges = []
        range_start = 0
        range_end = 0
        for n, is_used in zip(n_samples, used_by_samples):
            if is_used:
                range_end += n
            else:
                if range_start != range_end:
                    ranges.append(range(range_start, range_end))
                range_end += n
                range_start = range_end
        if range_start != range_end:
            ranges.append(np.arange(range_start, range_end))
        return ranges

    def _calculate_num_models(self):
        return int(1 + (np.shape(self.compressed_allocation)[1] - 2) / 2)

    def _find_utilized_models(self):

        utilized_models = list()

        samples_per_col = self._get_num_samples_per_column()

        if samples_per_col[0] > 0:
            utilized_models.append(0)
        else:
            warnings.warn("Allocation Warning: Model 0 is not evaluated,")

        for i in range(1, self.num_models):

            i_1 = i * 2
            i_2 = i_1 + 1

            i_1_allocation = self.compressed_allocation[:, i_1]
            i_2_allocation = self.compressed_allocation[:, i_2]

            if not np.array_equal(i_1_allocation, i_2_allocation):
                utilized_models.append(i)
            else:
                num_evals = samples_per_col[i_1-1]
                if num_evals > 0:
                    warnings.warn("Allocation Warning: Model %d is " % (i + 1)
                                  + "evaluated %d times but does " % num_evals
                                  + "not contribute to reduction in variance.")

        return utilized_models
