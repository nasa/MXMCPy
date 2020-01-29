from .gmf_ordered import GMFOrdered


class ACVMF(GMFOrdered):

    def __init__(self, model_cost, covariance, *args, **kwargs):
        recursion_refs = [0] * (len(model_cost) - 1)
        super().__init__(model_cost, covariance, recursion_refs=recursion_refs,
                         *args, **kwargs)


