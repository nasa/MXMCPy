import numpy as np

from .sample_allocation_base import SampleAllocationBase


class ACVSampleAllocation(SampleAllocationBase):

    def __init__(self, compressed_allocation):
        self._num_shared_samples = None
        super().__init__(compressed_allocation)

    @property
    def num_shared_samples(self):
        if self._num_shared_samples is None:
            self._num_shared_samples = self._calculate_sample_sharing_matrix()
        return self._num_shared_samples

    def get_k0_matrix(self):

        k_indices = [i - 1 for i in self.utilized_models if i != 0]
        k_0 = np.zeros(self.num_models - 1)
        n = self._get_num_samples_per_column()

        for i in k_indices:

            i_1 = i * 2 + 1
            i_2 = i_1 + 1
            k_0[i] = self.num_shared_samples[0, i_1] / n[0] / n[i_1] \
                - self.num_shared_samples[0, i_2] / n[0] / n[i_2]

        return k_0

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
            sample_sharing[i] = np.sum(pseudo_expanded[shared_with_key],
                                       axis=0)

        return sample_sharing

    def get_sample_split_for_model(self, model_index):
        col_1 = model_index * 2
        col_2 = col_1 + 1
        model_filter = np.logical_or(self.compressed_allocation[:, col_1] == 1,
                                     self.compressed_allocation[:, col_2] == 1)
        model_alloc = self.compressed_allocation[model_filter]
        ranges_1 = \
            self._get_ranges_from_samples_and_bool(model_alloc[:, 0],
                                                   model_alloc[:, col_1])
        ranges_2 = \
            self._get_ranges_from_samples_and_bool(model_alloc[:, 0],
                                                   model_alloc[:, col_2])
        return ranges_1, ranges_2

    def _get_column_names(self):

        column_names = ["0"]
        for i in range(1, self.num_models):

            column_names.append(str(i) + '_1')
            column_names.append(str(i) + '_2')

        return column_names
