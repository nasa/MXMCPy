import warnings

import h5py
import numpy as np
import pandas as pd

def read_allocation(filename):
    allocation_file = h5py.File(filename, 'r')
    compressed_key = 'Compressed_Allocation/compressed_allocation'
    compressed_allocation = np.array(allocation_file[compressed_key])
    method = allocation_file.attrs['Method']

    return SampleAllocation(compressed_allocation, method)



class SampleAllocation:

    def __init__(self, compressed_allocation, method=None):
        self.samples = None
        self._init_from_data(compressed_allocation, method)

        self._expanded_allocation = None
        self._num_shared_samples = None
        self._utilized_models = None

        self.num_total_samples = np.sum(self.compressed_allocation[:, 0])

    def _init_from_data(self, compressed_allocation_data, method):

        self.compressed_allocation = np.array(compressed_allocation_data)
        self.num_models = self._calculate_num_models()
        self.samples = pd.DataFrame()
        self.method = method

    @property
    def expanded_allocation(self):
        if self._expanded_allocation is None:
            self._expanded_allocation = self._expand_allocation()
        return self._expanded_allocation

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

        model_0_allocation = self.expanded_allocation[['0']]
        samples_per_model[0] = model_0_allocation.sum(axis=0).values[0]

        for model_index in range(1, self.num_models):

            allocation_sums = self._convert_2_to_1(model_index)
            samples_per_model[model_index] = np.sum(allocation_sums)

        return samples_per_model

    def get_sample_indices_for_model(self, model_index):

        if model_index == 0:
            return list(self.expanded_allocation['0'].to_numpy().nonzero()[0])

        allocation_sums = self._convert_2_to_1(model_index)
        return list(allocation_sums.nonzero()[0])

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
        n = self.expanded_allocation.sum(axis=0).values

        for i in k_indices:

            i_1 = i * 2 + 1
            i_2 = i_1 + 1
            k_0[i] = self.num_shared_samples[0, i_1] / n[0] / n[i_1] \
                - self.num_shared_samples[0, i_2] / n[0] / n[i_2]

        return k_0

    def get_k_matrix(self):

        k_size = self.num_models - 1
        k = np.zeros((k_size, k_size))
        n = self.expanded_allocation.sum(axis=0).values
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

        keys = self._get_column_names()
        sample_sharing = np.empty((len(keys), len(keys)))

        for i, key in enumerate(keys):

            shared_with_key = self.expanded_allocation[key] == 1
            sample_sharing[i] = \
                self.expanded_allocation[shared_with_key].sum(axis=0).values

        return sample_sharing

    def save(self, file_path):

        h5_file = h5py.File(file_path, 'w')

        self.create_file_structure(h5_file)
        self.write_sample_allocation_data_to_file(h5_file)

        h5_file.close()

    def create_file_structure(self, file):

        file.create_group('Compressed_Allocation')
        file.create_group('Expanded_Allocation')
        file.create_group('Samples')
        file.create_group('Input_Names')

        for model in range(self.num_models):

            group_name = 'Samples_Model_' + str(model)
            file.create_group(group_name)

    def write_sample_allocation_data_to_file(self, file):

        if not self.method:
            file.attrs['Method'] = "None"
        else:
            file.attrs['Method'] = self.method

        if not self.samples.empty:

            for model_index in range(self.num_models):

                group_name = 'Samples_Model_' + str(model_index)
                data = self.get_samples_for_model(model_index)

                self.write_file_data_set(file, group_name, data)

        self.write_file_data_set(file, "Compressed_Allocation",
                                 self.compressed_allocation)

        self.write_file_data_set(file, "Expanded_Allocation",
                                 self.expanded_allocation)

        self.write_file_data_set(file, "Samples",
                                 self.samples)

    @staticmethod
    def write_file_data_set(file, group_name, data_set):

        file[group_name].create_dataset(name=group_name.lower(), data=data_set)

    def get_sample_split_for_model(self, model_index):

        col_1 = '%d_1' % model_index
        col_2 = '%d_2' % model_index
        model_filter = np.logical_or(self.expanded_allocation[col_1] == 1,
                                     self.expanded_allocation[col_2] == 1)
        filter_1 = self.expanded_allocation[col_1][model_filter] == 1
        filter_2 = self.expanded_allocation[col_2][model_filter] == 1

        return filter_1, filter_2

    def _expand_allocation(self):

        expanded_allocation_data_frames = list()
        columns = self._get_column_names()

        for row in self.compressed_allocation:

            sample_group_size = row[0]
            row_data = []
            if sample_group_size > 0:
                row_data = [row[1:]] * sample_group_size

            data_frame = pd.DataFrame(columns=columns, data=row_data)
            expanded_allocation_data_frames.append(data_frame)

        expanded_dataframe = pd.concat(expanded_allocation_data_frames,
                                       ignore_index=True)
        return expanded_dataframe

    def _calculate_num_models(self):
        return int(1 + (np.shape(self.compressed_allocation)[1] - 2) / 2)

    def _get_column_names(self):

        column_names = ["0"]
        for i in range(1, self.num_models):

            column_names.append(str(i) + '_1')
            column_names.append(str(i) + '_2')

        return column_names

    def _convert_2_to_1(self, model_index):

        column_names = [str(model_index) + '_1', str(model_index) + '_2']
        temp_sums = self.expanded_allocation[column_names].sum(axis=1).values
        temp_sums[temp_sums == 2] = 1

        return temp_sums

    def _find_utilized_models(self):

        utilized_models = list()

        if self.expanded_allocation.iloc[:, 0].sum() > 0:
            utilized_models.append(0)
        else:
            warnings.warn("Allocation Warning: Model 0 is not evaluated,")

        for i in range(1, self.num_models):

            i_1 = i * 2 - 1
            i_2 = i_1 + 1

            i_1_allocation = self.expanded_allocation.iloc[:, i_1]
            i_2_allocation = self.expanded_allocation.iloc[:, i_2]

            if not i_1_allocation.equals(i_2_allocation):
                utilized_models.append(i)
            else:

                num_evals = i_1_allocation.sum()
                if num_evals > 0:
                    warnings.warn("Allocation Warning: Model %d is " % (i + 1)
                                  + "evaluated %d times but does " % num_evals
                                  + "not contribute to reduction in variance.")

        return utilized_models
