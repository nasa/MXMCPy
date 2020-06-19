from ..sample_allocations.acv_sample_allocation import ACVSampleAllocation
from ..sample_allocations.mlmc_sample_allocation import MLMCSampleAllocation

ALLOC_MAP = {None: ACVSampleAllocation,    # noqa: F405
             "mlmc": MLMCSampleAllocation} # noqa: F405

class OptimizationResult:

    def __init__(self, cost, variance, sample_array, method=None):
        self.cost = cost
        self.variance = variance
        self.allocation = ALLOC_MAP[method](sample_array)
