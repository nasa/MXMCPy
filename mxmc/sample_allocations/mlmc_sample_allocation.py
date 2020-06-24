from .acv_sample_allocation import ACVSampleAllocation


class MLMCSampleAllocation(ACVSampleAllocation):

    def __init__(self, compressed_allocation):
        super().__init__(compressed_allocation)
