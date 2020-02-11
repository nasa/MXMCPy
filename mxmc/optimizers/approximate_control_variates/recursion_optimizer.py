from .acv_optimizer import ACVOptimizer


class ACVRecursionOptimizer(ACVOptimizer):

    def __init__(self, model_costs, covariance, *args, **kwargs):
        super().__init__(model_costs, covariance, *args, **kwargs)
        self._recursion_refs = kwargs['recursion_refs']
