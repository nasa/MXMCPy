import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mxmc.optimizers.optimizer_base import InconsistentModelError
from mxmc.optimizer import Optimizer, ALGORITHM_MAP

ALL_ALGORITHMS = ALGORITHM_MAP.keys()


def calculate_variances_for_target_costs(covariance, model_costs, target_costs,
                                         algorithms=None):
    t_start = time.time()

    optimizer = Optimizer(model_costs, covariance)
    if algorithms is None:
        algorithms = ALL_ALGORITHMS
    variances = dict()
    for algo in algorithms:
        for ams in [False]: #, True]:
            name = algo if not ams else algo + " ams"
            print("running {}...".format(name))
            target_cost_vars = dict()
            for t_c in target_costs:
                t0 = time.time()
                try:
                    opt_result = optimizer.optimize(algorithm=algo,
                                                    target_cost=t_c,
                                                    auto_model_selection=ams)
                    target_cost_vars[t_c] = float(opt_result.variance)
                except InconsistentModelError:
                    print("    Models inconsistent with {}".format(algo))
                    target_cost_vars[t_c] = np.nan
                print("    {} s".format(time.time() - t0))
            variances[name] = target_cost_vars

    variances = pd.DataFrame(variances)

    print("Total Time: {} s".format(time.time() - t_start))
    return variances


def print_variances_as_table(variances):
    pd.options.display.float_format = '{:.3e}'.format
    sorted_var = variances.sort_values(list(variances.index), axis=1,
                                       ascending=False)
    print(sorted_var.transpose())


def plot_variances(variances, title, include_algorithms=None,
                   exclude_algorithms=None, plot_type="line"):
    literature_algorithms = {"mfmc", "mlmc", "wrdiff", "acvmf", "acvis",
                             "acvkl"}
    if exclude_algorithms is None:
        exclude_algorithms = {"acvmfmc", "acvmfmc ams"}
    if include_algorithms is None:
        include_algorithms = variances.columns
    algorithms_to_plot = set(include_algorithms) - set(exclude_algorithms)
    for algo in list(algorithms_to_plot):
        if algo.endswith(' ams'):
            if (variances[algo] == variances[algo[:-4]]).all():
                algorithms_to_plot.remove(algo)
    variances_to_plot = variances.loc[:, algorithms_to_plot]
    variances_to_plot = variances_to_plot.sort_values(list(variances.index),
                                                      axis=1, ascending=False)

    if plot_type == "line":
        _line_plot(literature_algorithms, variances_to_plot, title)
    elif plot_type == "bar":
        _bar_plot(literature_algorithms, variances_to_plot, title)


def _line_plot(literature_algorithms, variances_to_plot, title):
    fig, ax = plt.subplots(figsize=(13.5, 9))
    for algo in variances_to_plot.columns:
        style_string = '.-' if algo in literature_algorithms else '.:'
        ax.loglog(variances_to_plot.index, variances_to_plot[algo],
                  style_string, label=algo)

    ax.set_xlabel('target cost')
    ax.set_ylabel('estimator variance')
    plt.title(title)
    plt.legend()
    plt.savefig(_make_fig_name(title))
    plt.show()


def _bar_plot(literature_algorithms, variances_to_plot, title):
    num_algos = len(variances_to_plot.columns)
    total_width = 0.8
    bar_width = total_width / num_algos

    fig, ax = plt.subplots(figsize=(13.5, 9))
    x = np.arange(len(variances_to_plot.index))
    for i, algo in enumerate(variances_to_plot.columns):
        y = variances_to_plot.loc[:, algo].values
        alpha = 1.0 if algo in literature_algorithms else 0.5
        edgecolor = "black" if algo in literature_algorithms else None
        linewidth = 1 if algo in literature_algorithms else 0
        b = ax.bar(x + i * bar_width, y, bar_width, bottom=0.001, label=algo,
                   alpha=alpha, linewidth=linewidth, edgecolor=edgecolor)

    ax.set_xticks(x + total_width / 2)
    ax.set_xticklabels([str(i) for i in variances_to_plot.index])
    ax.set_yscale('log')

    ax.set_xlabel('target cost')
    ax.set_ylabel('estimator variance')
    plt.title(title)
    plt.legend()
    plt.savefig(_make_fig_name(title))
    plt.show()


def _make_fig_name(title):
    res_dir = "results/"
    filename = title.lower().strip(",").replace(" ", "_") + ".png"
    return res_dir + filename

