from ..sample_allocation import SampleAllocation


class OptimizationResult:

    def __init__(self, cost, variance, sample_array, method=None):
        self.cost = cost
        self.variance = variance
        self.allocation = SampleAllocation(sample_array, method)
