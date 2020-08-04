import numpy as np

from .sample_allocation_base import SampleAllocationBase

class MBLUESampleAllocation(SampleAllocationBase):

    def __init__(self, compressed_allocation):
        super().__init__(compressed_allocation)

    def get_sample_indices_for_model(self, model_index):
        return np.arange(self.compressed_allocation[0,0])

    def _calculate_num_models(self):
        return self.compressed_allocation.shape[1] - 1

    def _find_utilized_models(self):
        pass
