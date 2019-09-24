import pandas as pd
import numpy as np


class SampleAllocation:
    def __init__(self, compressed_allocation):
        self.compressed_allocation = compressed_allocation.tolist()
        self.num_models = self._calculate_num_models()
        self.expanded_allocation = self._expand_allocation()

    def get_number_of_samples_per_model(self):
        samples_per_model = np.zeros(self.num_models)
        for i in range(self.num_models):
            if i == 0:
                samples_per_model[i] = self.expanded_allocation[['0']].sum(axis=0).values[0]
            else:
                temp_sums = self.expanded_allocation[[str(i)+'_1', str(i)+'_2']].sum(axis=1).values
                for j, n in enumerate(temp_sums):
                    if n == 2:
                        temp_sums[j] = 1
                samples_per_model[i] = np.sum(temp_sums)
        return samples_per_model


    def _expand_allocation(self):
        expanded_allocation_data_frames = []

        for i in range(len(self.compressed_allocation)):
            row = self.compressed_allocation[i].copy()
            multiplier = row.pop(0)
            expanded_allocation_data_frames.append(pd.DataFrame(columns=self._get_column_names(),
                                                                data=[row] * multiplier))
        expanded_dataframe = pd.concat(expanded_allocation_data_frames, ignore_index=True)
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
