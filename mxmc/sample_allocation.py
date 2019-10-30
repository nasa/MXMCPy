import warnings

import h5py
import numpy as np
import pandas as pd


class SampleAllocation:
    def __init__(self, compressed_allocation, method=None):
        if type(compressed_allocation) is str:
            allocation_file = h5py.File(compressed_allocation, 'r')
            self.compressed_allocation = np.array(allocation_file[
                'Compressed_Allocation/compressed_allocation'])
            self.num_models = self._calculate_num_models()
            self.expanded_allocation = pd.DataFrame(
                allocation_file['Expanded_Allocation/expanded_allocation'])
            try:
                self.samples = np.array(allocation_file['Samples/samples'])
            except KeyError:
                self.samples = pd.DataFrame()
            self.method = allocation_file.attrs['Method']
            allocation_file.close()
        else:
            if method is None:
                raise ValueError("Must specify method")
            self.compressed_allocation = compressed_allocation.tolist()
            self.num_models = self._calculate_num_models()
            self.expanded_allocation = self._expand_allocation()
            self.samples = pd.DataFrame()
            self.method = method
        self._num_shared_samples = self._calculate_sample_sharing_matrix()
        self.utilized_models = self._find_utilized_models()

    def get_total_number_of_samples(self):
        return len(self.expanded_allocation)

    def get_number_of_samples_per_model(self):
        samples_per_model = np.zeros(self.num_models, dtype=int)
        for model_index in range(self.num_models):
            if model_index == 0:
                samples_per_model[model_index] = \
                    self.expanded_allocation[['0']].sum(axis=0).values[0]
            else:
                allocation_sums = self._convert_2_to_1(model_index)
                samples_per_model[model_index] = np.sum(allocation_sums)
        return samples_per_model

    def get_sample_indices_for_model(self, model):
        if model == 0:
            return list(self.expanded_allocation['0'].to_numpy().nonzero()[0])

        allocation_sums = self._convert_2_to_1(model)
        return list(allocation_sums.nonzero()[0])

    def generate_samples(self, input_generator):
        self.samples = input_generator.generate_samples(
            self.get_total_number_of_samples())

    def get_samples_for_model(self, model):
        return self.samples.iloc[self.get_sample_indices_for_model(model), :]

    def get_k0_matrix(self):
        k_indices = [i - 1 for i in self.utilized_models if i != 0]
        k0 = np.empty(self.num_models - 1)
        n = self.expanded_allocation.sum(axis=0).values
        for i in k_indices:
            i_1 = i * 2 + 1
            i_2 = i_1 + 1
            k0[i] = self._num_shared_samples[0, i_1] / n[0] / n[i_1] \
                - self._num_shared_samples[0, i_2] / n[0] / n[i_2]
        return k0

    def get_k_matrix(self):
        k_size = self.num_models - 1
        k = np.zeros((k_size, k_size))
        self._num_shared_samples = self._calculate_sample_sharing_matrix()
        n = self.expanded_allocation.sum(axis=0).values
        k_indices = [i - 1 for i in self.utilized_models if i != 0]
        for i in k_indices:
            i_1 = i * 2 + 1
            i_2 = i_1 + 1
            for j in k_indices:
                j_1 = j * 2 + 1
                j_2 = j_1 + 1
                k[i, j] = \
                    self._num_shared_samples[i_1, j_1] / n[i_1] / n[j_1] \
                    - self._num_shared_samples[i_1, j_2] / n[i_1] / n[j_2] \
                    - self._num_shared_samples[i_2, j_1] / n[i_2] / n[j_1] \
                    + self._num_shared_samples[i_2, j_2] / n[i_2] / n[j_2]
        return k

    def _calculate_sample_sharing_matrix(self):
        keys = list(self.expanded_allocation.columns.values)
        sample_sharing = np.empty((len(keys), len(keys)))
        for i, key in enumerate(keys):
            shared_with_key = self.expanded_allocation[key] == 1
            sample_sharing[i] = \
                self.expanded_allocation[shared_with_key].sum(axis=0).values
        return sample_sharing

    def save(self, file_path):
        f = h5py.File(file_path, 'w')
        f.attrs['Method'] = self.method
        group_compressed_allocation = f.create_group('Compressed_Allocation')
        group_expanded_allocation = f.create_group('Expanded_Allocation')
        group_samples = f.create_group('Samples')
        _ = f.create_group('Input_Names')
        for model in range(self.num_models):
            group_model = f.create_group('Samples_Model_' + str(model))
            if not self.samples.empty:
                group_model.create_dataset(name='samples_model_' + str(model),
                                           data=self.get_samples_for_model(
                                               model))
        group_compressed_allocation.create_dataset(
            name='compressed_allocation', data=self.compressed_allocation)
        group_expanded_allocation.create_dataset(name='expanded_allocation',
                                                 data=self.expanded_allocation)
        group_samples.create_dataset(name='samples', data=self.samples)
        f.close()

    def get_sample_split_for_model(self, i):
        col_1 = '%d_1' % i
        col_2 = '%d_2' % i
        model_filter = np.logical_or(self.expanded_allocation[col_1] == 1,
                                     self.expanded_allocation[col_2] == 1)
        filter_1 = self.expanded_allocation[col_1][model_filter] == 1
        filter_2 = self.expanded_allocation[col_2][model_filter] == 1
        return filter_1, filter_2

    def _expand_allocation(self):
        expanded_allocation_data_frames = []

        for index in range(len(self.compressed_allocation)):
            row = self.compressed_allocation[index].copy()
            sample_group_size = row.pop(0)
            expanded_allocation_data_frames.append(
                pd.DataFrame(columns=self._get_column_names(),
                             data=[row] * sample_group_size))
        expanded_dataframe = pd.concat(expanded_allocation_data_frames,
                                       ignore_index=True)
        return expanded_dataframe

    def _calculate_num_models(self):
        return int(1 + (np.shape(self.compressed_allocation)[1] - 2) / 2)

    def _get_column_names(self):
        column_names = []
        for i in range(self.num_models):
            if i == 0:
                column_names.append(str(i))
            else:
                column_names.append(str(i) + '_1')
                column_names.append(str(i) + '_2')
        return column_names

    def _convert_2_to_1(self, model):
        temp_sums = self.expanded_allocation[
            [str(model) + '_1', str(model) + '_2']].sum(axis=1).values
        for index, element in enumerate(temp_sums):
            if element == 2:
                temp_sums[index] = 1
        return temp_sums

    def _find_utilized_models(self):
        utilized_models = []

        if self.expanded_allocation.iloc[:, 0].sum() > 0:
            utilized_models.append(0)
        else:
            warnings.warn("Allocation Warning: Model 0 is not evaluated,")

        for i in range(1, self.num_models):
            i_1 = i * 2 - 1
            i_2 = i_1 + 1
            if not self.expanded_allocation.iloc[:, i_1].equals(
                    self.expanded_allocation.iloc[:, i_2]):
                utilized_models.append(i)
            else:
                num_evals = self.expanded_allocation.iloc[:, i_1].sum()
                if num_evals > 0:
                    warnings.warn("Allocation Warning: Model %d is " % (i + 1)
                                  + "evaluated %d times but does" % num_evals
                                  + "not contribute to reduction in variance.")
        return utilized_models
