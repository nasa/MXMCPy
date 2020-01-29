from .gmf_ordered import GMFOrdered


class ACVMFMC(GMFOrdered):

    def __init__(self, model_cost, covariance, *args, **kwargs):
        recursion_refs = [i for i in range(len(model_cost) - 1)]
        super().__init__(model_cost, covariance, recursion_refs=recursion_refs,
                         *args, **kwargs)