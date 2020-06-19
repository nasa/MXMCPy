from .sample_allocation_base import SampleAllocationBase

class ACVSampleAllocation(SampleAllocationBase):

    def __init__(self, compressed_allocation):
        super().__init__(compressed_allocation)

    def _get_column_names(self):

        column_names = ["0"]
        for i in range(1, self.num_models):

            column_names.append(str(i) + '_1')
            column_names.append(str(i) + '_2')

        return column_names
