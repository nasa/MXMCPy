from scipy import optimize as scipy_optimize


def perform_slsqp_then_nelder_mead(bounds, constraints, initial_guess,
                                   obj_func, obj_func_and_grad):
    slsqp_x = perform_slsqp(obj_func_and_grad, initial_guess, bounds,
                            constraints)
    nm_x = perform_nelder_mead(obj_func, slsqp_x, bounds, constraints)
    return nm_x


def perform_slsqp(obj_func_and_grad, initial_guess, bounds,
                  constraints):
    options = {"disp": False, "ftol": 1e-10}
    opt_result = scipy_optimize.minimize(
            obj_func_and_grad,
            initial_guess,
            constraints=constraints,
            bounds=bounds, jac=True,
            method='SLSQP',
            options=options)
    return opt_result.x


def perform_nelder_mead(obj_func, initial_guess, bounds, constraints):
    options = {"disp": False, "xatol": 1e-12, "fatol": 1e-12,
               "maxfev": 500 * len(initial_guess)}
    opt_result = scipy_optimize.minimize(
            _penalized_objective_function,
            initial_guess,
            args=(obj_func, bounds, constraints),
            method='Nelder-Mead',
            options=options)
    return opt_result.x


def _penalized_objective_function(x, obj_func, bounds, constraints):
    fun = obj_func(x)
    penalty = _calculate_penalty(x, bounds, constraints)

    return fun + penalty


def _calculate_penalty(x, bounds, constraints):
    penalty_weight = 1e6
    penalty = 0
    for constr in constraints:
        c_val = constr["fun"](x, *constr["args"])
        if c_val < 0:
            penalty -= c_val * penalty_weight
    for r, (lb, ub) in zip(x, bounds):
        lb_val = r - lb
        ub_val = ub - r
        if lb_val < 0:
            penalty -= lb_val * penalty_weight
        if ub_val < 0:
            penalty -= ub_val * penalty_weight
    return penalty