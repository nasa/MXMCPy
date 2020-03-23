import warnings

import h5py
import numpy as np

def read_allocation(filename):
    '''
    Read sample allocation from file

    :param filename: name of hdf5 sample allocation file
    :type filename: string

    '''
    allocation_file = h5py.File(filename, 'r')
    compressed_key = 'Compressed_Allocation/compressed_allocation'
    compressed_allocation = np.array(allocation_file[compressed_key])
    method = allocation_file.attrs['Method']

    return SampleAllocation(compressed_allocation, method)


class SampleAllocation:

    def __init__(self, compressed_allocation, method=None):
        self._init_from_data(compressed_allocation, method)
        self.num_total_samples = np.sum(self.compressed_allocation[:, 0])

        self._expanded_allocation = None
        self._num_shared_samples = None
        self._utilized_models = None

    def _init_from_data(self, compressed_allocation_data, method):

        self.compressed_allocation = np.array(compressed_allocation_data)
        self.num_models = self._calculate_num_models()
        self.method = method

    @property
    def num_shared_samples(self):
        if self._num_shared_samples is None:
            self._num_shared_samples = self._calculate_sample_sharing_matrix()
        return self._num_shared_samples

    @property
    def utilized_models(self):
        if self._utilized_models is None:
            self._utilized_models = self._find_utilized_models()
        return self._utilized_models

    def get_number_of_samples_per_model(self):

        samples_per_model = np.empty(self.num_models, dtype=int)
        for model_index in range(self.num_models):
            samples_per_model[model_index] = \
                len(self.get_sample_indices_for_model(model_index))

        return samples_per_model

    def get_sample_indices_for_model(self, model_index):

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

        if len(all_samples) < self.num_total_samples:
            raise ValueError("Too few inputs samples to allocate to models!")

        model_samples = []
        for model_index in range(self.num_models):
            sample_indices = self.get_sample_indices_for_model(model_index)
            model_samples.append(all_samples[sample_indices])

        return model_samples

    def get_k0_matrix(self):

        k_indices = [i - 1 for i in self.utilized_models if i != 0]
        k_0 = np.empty(self.num_models - 1)
        n = self._get_num_samples_per_column()

        for i in k_indices:

            i_1 = i * 2 + 1
            i_2 = i_1 + 1
            k_0[i] = self.num_shared_samples[0, i_1] / n[0] / n[i_1] \
                - self.num_shared_samples[0, i_2] / n[0] / n[i_2]

        return k_0

    def _get_num_samples_per_column(self):

        samples_per_group = self.compressed_allocation[:, 0]
        samples_per_col = np.zeros(2*self.num_models-1, dtype=int)

        for i, col_inds in enumerate(self.compressed_allocation[:, 1:].T):
            samples_per_col[i] = np.sum(samples_per_group[col_inds == 1])

        return samples_per_col

    def get_k_matrix(self):

        k_size = self.num_models - 1
        k = np.zeros((k_size, k_size))
        n = self._get_num_samples_per_column()
        k_indices = [i - 1 for i in self.utilized_models if i != 0]

        for i in k_indices:

            i_1 = i * 2 + 1
            i_2 = i_1 + 1

            for j in k_indices:

                j_1 = j * 2 + 1
                j_2 = j_1 + 1
                k[i, j] = \
                    self.num_shared_samples[i_1, j_1] / n[i_1] / n[j_1] \
                    - self.num_shared_samples[i_1, j_2] / n[i_1] / n[j_2] \
                    - self.num_shared_samples[i_2, j_1] / n[i_2] / n[j_1] \
                    + self.num_shared_samples[i_2, j_2] / n[i_2] / n[j_2]

        return k

    def _calculate_sample_sharing_matrix(self):

        pseudo_expanded = self.compressed_allocation[:, 0].reshape((-1, 1)) * \
                          self.compressed_allocation[:, 1:]
        num_cols = 2*self.num_models - 1
        sample_sharing = np.empty((num_cols, num_cols))

        for i in range(num_cols):
            shared_with_key = self.compressed_allocation[:, i+1] == 1
            sample_sharing[i] = np.sum(pseudo_expanded[shared_with_key], axis=0)

        return sample_sharing

    def save(self, file_path):

        h5_file = h5py.File(file_path, 'w')
        self.write_file_data_set(h5_file, "Compressed_Allocation",
                                 self.compressed_allocation)
        if not self.method:
            h5_file.attrs['Method'] = "None"
        else:
            h5_file.attrs['Method'] = self.method
        h5_file.close()

    @staticmethod
    def write_file_data_set(h5file, group_name, dataset):
        h5file.create_group(group_name)
        h5file[group_name].create_dataset(name=group_name.lower(), data=dataset)

    def get_sample_split_for_model(self, model_index):
        col_1 = model_index * 2
        col_2 = col_1 + 1
        model_filter = np.logical_or(self.compressed_allocation[:, col_1] == 1,
                                     self.compressed_allocation[:, col_2] == 1)
        model_alloc = self.compressed_allocation[model_filter]
        ranges_1 = self._get_ranges_from_samples_and_bool(model_alloc[:, 0],
                                                          model_alloc[:, col_1])
        ranges_2 = self._get_ranges_from_samples_and_bool(model_alloc[:, 0],
                                                          model_alloc[:, col_2])
        return ranges_1, ranges_2

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

    def _get_column_names(self):

        column_names = ["0"]
        for i in range(1, self.num_models):

            column_names.append(str(i) + '_1')
            column_names.append(str(i) + '_2')

        return column_names

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
