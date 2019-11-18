import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from mxmc.acvmf import ACVMF
from mxmc.optimizer import Optimizer


def monomial_model_variances(powers):
    num_models = len(powers)
    cov = np.empty((num_models, num_models))
    vardiff = np.empty((num_models, num_models))
    for i, p_i in enumerate(powers):
        for j, p_j in enumerate(powers):
            cov[i, j] = 1.0 / (p_i + p_j + 1) - 1.0 / ((p_i + 1) * (p_j + 1))
            if i == j:
                vardiff[i, j] = cov[i, j]
            else:
                vardiff[i, j] = 1 / (2 * p_i + 1) - 2 / (p_i + p_j + 1) \
                                + 1 / (2 * p_j + 1) \
                                - (1 / (p_i + 1) - 1 / (p_j + 1)) ** 2
    return cov, vardiff


def monomial_model_costs(powers):
    return np.power(10.0, -np.arange(len(powers)))


def test_monomial_model(algorithm):
    optimizer = Optimizer(MODELCOSTS, covariance=COVARIANCE,
                          vardiff_matrix=VARDIFF_MATRIX)
    opt_result = optimizer.optimize(algorithm=algorithm,
                                    target_cost=TARGET_COST)
    return opt_result.variance


def get_variances_of_all_algos():
    algorithms = ["mfmc", "mlmc"]
    variances = dict()
    for algo in algorithms:
        variances[algo] = test_monomial_model(algo)
    optimizer = Optimizer(MODELCOSTS, covariance=COVARIANCE,
                          vardiff_matrix=VARDIFF_MATRIX)
    opt_result = optimizer.optimize(algorithm="acvmf",
                                    target_cost=TARGET_COST)
    variances["acvmf"] = opt_result.variance
    return variances


def plot_3d_acvmf_vars():
    optimizer = ACVMF(MODELCOSTS, covariance=COVARIANCE)
    sample_nums = [1, 1, 1]

    output = []
    max_i = TARGET_COST / np.sum(MODELCOSTS)
    for i in np.linspace(1, max_i, 50):
        sample_nums[0] = i
        remaining_cost = TARGET_COST - i*sum(MODELCOSTS)
        max_j = remaining_cost/MODELCOSTS[1]
        for j in np.linspace(1, max_j, 50):
            sample_nums[1] = j
            remaining_cost = TARGET_COST - i*sum(MODELCOSTS) - j*MODELCOSTS[1]
            k = remaining_cost/MODELCOSTS[2]
            sample_nums[2] = k
            try:
                var, _ = optimizer._compute_objective_function(sample_nums, TARGET_COST)
                output.append(sample_nums + [var])
            except:
                pass
    output = np.array(output)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(output[:, 0], output[:, 1], output[:, 2], c=output[:, 3])
    min_output = output[np.where(output[:, 3] == min(output[:, 3]))]
    print("min", min_output)
    ax.scatter(min_output[:, 0], min_output[:, 1], min_output[:, 2], c="red")

    opt_path = get_opt_path()
    ax.plot(opt_path[:, 0], opt_path[:, 1], zs=opt_path[:, 2], c="orange")
    opt_path = get_opt_path_2()
    ax.plot(opt_path[:, 0], opt_path[:, 1], zs=opt_path[:, 2], c="orange")

    ax.set_xlabel("model 0")
    ax.set_ylabel("model 1")
    ax.set_zlabel("model 2")
    plt.colorbar(sc)
    plt.show()


def plot_2d_acvmf_vars(recent_opt_path):
    optimizer = ACVMF(MODELCOSTS, covariance=COVARIANCE)
    output = []
    for r1 in np.linspace(1, 100, 100):
        for r2 in np.linspace(1, 1000, 500):
            N = TARGET_COST / np.dot(MODELCOSTS, [1, r1, r2])
            if N < 1:
                continue
            N1 = N*r1
            N2 = N*r2
            if N1 < N+1 or N2 < N+1:
                continue
            # if N2 < N1:
            #     continue
            if r2 > 100:
                continue
            sample_nums = [N, N1, N2]
            try:
                var, _ = optimizer._compute_objective_function(sample_nums,
                                                               TARGET_COST)
                # var, _ = optimizer._compute_objective_function_ratio(
                #                                                np.array([r1,r2]),
                #                                                TARGET_COST)

                if var < 0.0025:
                    output.append([r1, r2, var])
            except:
                pass
    output = np.array(output)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(output[:, 0], output[:, 1], output[:, 2], c=output[:, 2])
    min_output = output[np.where(output[:, 2] == min(output[:, 2]))]
    print("min", min_output)
    ax.scatter(min_output[:, 0], min_output[:, 1], min_output[:, 2], c="red")
    # opt_path = get_opt_path_ratios()
    # ax.plot(opt_path[:, 0], opt_path[:, 1], 'g.-')
    # ax.plot(opt_path[-1, 0], opt_path[-1, 1], 'gx')
    #
    # opt_path = get_opt_path_ratios_trust()
    # ax.plot(opt_path[:, 0], opt_path[:, 1], 'm.-')
    # ax.plot(opt_path[-1, 0], opt_path[-1, 1], 'rx')
    #
    ax.plot(recent_opt_path[:, 0], recent_opt_path[:, 1], 'b.-', zs=recent_opt_path[:, 2])
    ax.plot(recent_opt_path[-2:, 0], recent_opt_path[-2:, 1], 'kx', zs=recent_opt_path[-2:, 2])

    ax.set_xlabel("r1")
    ax.set_ylabel("r2")
    plt.colorbar(sc)
    plt.show()


if __name__ == "__main__":
    EXPONENTS = [4, 3, 2, 1]
    COVARIANCE, VARDIFF_MATRIX = monomial_model_variances(EXPONENTS)
    MODELCOSTS = monomial_model_costs(EXPONENTS)
    TARGET_COST = 100
    var = get_variances_of_all_algos()
    #assert(abs(var['acvmf'] - 4.61206448e-05) < 1e-12)
    # plot_2d_acvmf_vars(opt_path)

