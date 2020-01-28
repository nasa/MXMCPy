from .generalized_multi_fidelity import GMFOrdered, GMFUnordered


class ACVMF(GMFOrdered):

    def __init__(self, model_cost, covariance, *args, **kwargs):
        recursion_refs = [0] * (len(model_cost) - 1)
        super().__init__(model_cost, covariance, recursion_refs=recursion_refs,
                         *args, **kwargs)


class ACVMFU(GMFUnordered):

    def __init__(self, model_cost, covariance, *args, **kwargs):
        recursion_refs = [0] * (len(model_cost) - 1)
        super().__init__(model_cost, covariance, recursion_refs=recursion_refs,
                         *args, **kwargs)


class ACVMFMC(GMFOrdered):

    def __init__(self, model_cost, covariance, *args, **kwargs):
        recursion_refs = [i for i in range(len(model_cost) - 1)]
        super().__init__(model_cost, covariance, recursion_refs=recursion_refs,
                         *args, **kwargs)
