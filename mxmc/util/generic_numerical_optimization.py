from scipy import optimize as scipy_optimize


def perform_slsqp_then_nelder_mead(bounds, constraints, initial_guess,
                                   obj_func, obj_func_and_grad):
    slsqp_result = _slsqp(bounds, constraints, initial_guess,
                          obj_func_and_grad)

    slsqp_constr_violated = _calculate_penalty(slsqp_result.x, bounds,
                                               constraints) > 0
    if slsqp_constr_violated:
        nm_initial_guess = initial_guess
    else:
        nm_initial_guess = slsqp_result.x

    nm_x = perform_nelder_mead(bounds, constraints, nm_initial_guess, obj_func)
    return nm_x


def perform_slsqp(bounds, constraints, initial_guess, obj_func_and_grad):
    opt_result = _slsqp(bounds, constraints, initial_guess,
                        obj_func_and_grad)
    return opt_result.x


def _slsqp(bounds, constraints, initial_guess, obj_func_and_grad):
    options = {"disp": False, "ftol": 1e-10}
    opt_result = scipy_optimize.minimize(
            obj_func_and_grad,
            initial_guess,
            constraints=constraints,
            bounds=bounds, jac=True,
            method='SLSQP',
            options=options)
    return opt_result


def perform_nelder_mead(bounds, constraints, initial_guess, obj_func):
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
    penalty_weight = 1e16
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
