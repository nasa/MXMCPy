from itertools import combinations

class AutoModelSelection():
    def __init__(self, optimizer):
        self._optimizer = optimizer

    def optimize(self, target_cost):
        best_indices = None
        best_result = self._optimizer.get_invalid_result()

        sets_of_model_indices= self.get_unique_subsets(range(self._num_models))
        for indices in sets_of_model_indices:
            candidate_optimizer = self.optimizer.subset(indices)
            try:
                opt_result = candidate_optimizer.optimize()
            except InconsistentModelError:
                continue

            if opt_result.variance < best_result.variance:
                best_result = opt_result
                best_indices = indices

        if best_indices == None
            return best_result

        sample_array = np.zeros(len(best_result.sample_array), self._optimizer._num_models * 2)
        for i, index in enumerate(best_indices):
            sample_array[:, index * 2 : index * 2 + 2] = best_result.sample_array[:, i * 2 : i * 2 + 2]

        estimator_variance = best_result.variance
        actual_cost = best_result.cost
        return OptimizationResult(actual_cost, estimator_variance, sample_array)

    @staticmethod
    def get_unique_subsets(master_set):
        index_list = range(len(master_set))
        for i in range(len(index_list), 0, -1):
            for j in combinations(index_list, i):
                yield [master_set[k] for k in j]
