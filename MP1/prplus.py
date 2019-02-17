import warnings
import sys
import numpy
from scipy._lib.six import callable, xrange
from numpy import (atleast_1d, eye, mgrid, argmin, zeros, shape, squeeze,
                   vectorize, asarray, sqrt, Inf, asfarray, isinf)
import numpy as np
from scipy.optimize import minpack2
from scipy.optimize.linesearch import line_search_wolfe1

_epsilon = sqrt(numpy.finfo(float).eps)
# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.'}

def fmin_cg(f, x0, fprime=None, args=(), gtol=1e-5, norm=Inf, epsilon=_epsilon,
            maxiter=None, min_moment=None, full_output=0, disp=1, retall=0,
            callback=None):
    """
    Minimize a function using a nonlinear conjugate gradient algorithm.

    Parameters
    ----------
    f : callable, ``f(x, *args)``
        Objective function to be minimized.  Here `x` must be a 1-D array of
        the variables that are to be changed in the search for a minimum, and
        `args` are the other (fixed) parameters of `f`.
    x0 : ndarray
        A user-supplied initial estimate of `xopt`, the optimal value of `x`.
        It must be a 1-D array of values.
    fprime : callable, ``fprime(x, *args)``, optional
        A function that returns the gradient of `f` at `x`. Here `x` and `args`
        are as described above for `f`. The returned value must be a 1-D array.
        Defaults to None, in which case the gradient is approximated
        numerically (see `epsilon`, below).
    args : tuple, optional
        Parameter values passed to `f` and `fprime`. Must be supplied whenever
        additional fixed parameters are needed to completely specify the
        functions `f` and `fprime`.
    gtol : float, optional
        Stop when the norm of the gradient is less than `gtol`.
    norm : float, optional
        Order to use for the norm of the gradient
        (``-np.Inf`` is min, ``np.Inf`` is max).
    epsilon : float or ndarray, optional
        Step size(s) to use when `fprime` is approximated numerically. Can be a
        scalar or a 1-D array.  Defaults to ``sqrt(eps)``, with eps the
        floating point machine precision.  Usually ``sqrt(eps)`` is about
        1.5e-8.
    maxiter : int, optional
        Maximum number of iterations to perform. Default is ``200 * len(x0)``.
    min_moment : float, optional
        Minimum value of momentum before the procedure terminates.
    full_output : bool, optional
        If True, return `fopt`, `func_calls`, `grad_calls`, and `warnflag` in
        addition to `xopt`.  See the Returns section below for additional
        information on optional return values.
    disp : bool, optional
        If True, return a convergence message, followed by `xopt`.
    retall : bool, optional
        If True, add to the returned values the results of each iteration.
    callback : callable, optional
        An optional user-supplied function, called after each iteration.
        Called as ``callback(xk)``, where ``xk`` is the current value of `x0`.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e. ``f(xopt) == fopt``.
    fopt : float, optional
        Minimum value found, f(xopt).  Only returned if `full_output` is True.
    func_calls : int, optional
        The number of function_calls made.  Only returned if `full_output`
        is True.
    grad_calls : int, optional
        The number of gradient calls made. Only returned if `full_output` is
        True.
    warnflag : int, optional
        Integer value with warning status, only returned if `full_output` is
        True.

        0 : Success.

        1 : The maximum number of iterations was exceeded.

        2 : Gradient and/or function calls were not changing.  May indicate
            that precision was lost, i.e., the routine did not converge.

    allvecs : list of ndarray, optional
        List of arrays, containing the results at each iteration.
        Only returned if `retall` is True.

    See Also
    --------
    minimize : common interface to all `scipy.optimize` algorithms for
               unconstrained and constrained minimization of multivariate
               functions.  It provides an alternative way to call
               ``fmin_cg``, by specifying ``method='CG'``.

    Notes
    -----
    This conjugate gradient algorithm is based on that of Polak and Ribiere
    [1]_.

    Conjugate gradient methods tend to work better when:

    1. `f` has a unique global minimizing point, and no local minima or
       other stationary points,
    2. `f` is, at least locally, reasonably well approximated by a
       quadratic function of the variables,
    3. `f` is continuous and has a continuous gradient,
    4. `fprime` is not too large, e.g., has a norm less than 1000,
    5. The initial guess, `x0`, is reasonably close to `f` 's global
       minimizing point, `xopt`.

    References
    ----------
    .. [1] Wright & Nocedal, "Numerical Optimization", 1999, pp. 120-122.

    Examples
    --------
    Example 1: seek the minimum value of the expression
    ``a*u**2 + b*u*v + c*v**2 + d*u + e*v + f`` for given values
    of the parameters and an initial guess ``(u, v) = (0, 0)``.

    >>> args = (2, 3, 7, 8, 9, 10)  # parameter values
    >>> def f(x, *args):
    ...     u, v = x
    ...     a, b, c, d, e, f = args
    ...     return a*u**2 + b*u*v + c*v**2 + d*u + e*v + f
    >>> def gradf(x, *args):
    ...     u, v = x
    ...     a, b, c, d, e, f = args
    ...     gu = 2*a*u + b*v + d     # u-component of the gradient
    ...     gv = b*u + 2*c*v + e     # v-component of the gradient
    ...     return np.asarray((gu, gv))
    >>> x0 = np.asarray((0, 0))  # Initial guess.
    >>> from scipy import optimize
    >>> res1 = optimize.fmin_cg(f, x0, fprime=gradf, args=args)
    Optimization terminated successfully.
             Current function value: 1.617021
             Iterations: 4
             Function evaluations: 8
             Gradient evaluations: 8
    >>> res1
    array([-1.80851064, -0.25531915])

    Example 2: solve the same problem using the `minimize` function.
    (This `myopts` dictionary shows all of the available options,
    although in practice only non-default values would be needed.
    The returned value will be a dictionary.)

    >>> opts = {'maxiter' : None,    # default value.
    ...         'disp' : True,    # non-default value.
    ...         'gtol' : 1e-5,    # default value.
    ...         'norm' : np.inf,  # default value.
    ...         'eps' : 1.4901161193847656e-08}  # default value.
    >>> res2 = optimize.minimize(f, x0, jac=gradf, args=args,
    ...                          method='CG', options=opts)
    Optimization terminated successfully.
            Current function value: 1.617021
            Iterations: 4
            Function evaluations: 8
            Gradient evaluations: 8
    >>> res2.x  # minimum found
    array([-1.80851064, -0.25531915])

    """
    opts = {'gtol': gtol,
            'norm': norm,
            'eps': epsilon,
            'disp': disp,
            'maxiter': maxiter,
            'min_moment': min_moment,
            'return_all': retall}

    res = _minimize_cg(f, x0, args, fprime, callback=callback, **opts)

    if full_output:
        retlist = res['x'], res['fun'], res['nfev'], res['njev'], res['status']
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_cg(fun, x0, args=(), jac=None, callback=None,
                 gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None,
                 min_moment=None, disp=False, return_all=False,
                 **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    conjugate gradient algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    min_moment : float, optional
        Minimum value of momentum before the procedure terminates.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.

    """
    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    x0 = asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0) * 200
    if min_moment is None:
        min_moment = np.NINF
    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)
    gfk = myfprime(x0)
    k = 0
    xk = x0

    # Sets the initial step guess to dx ~ 1
    old_fval = f(xk)
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    if retall:
        allvecs = [xk]
    warnflag = 0
    pk = -gfk
    gnorm = vecnorm(gfk, ord=norm)
    beta_k = np.Inf
    while (gnorm > gtol) and (k < maxiter) and (beta_k > min_moment):
        deltak = numpy.dot(gfk, gfk)

        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, myfprime, xk, pk, gfk, old_fval,
                                          old_old_fval, c2=0.4, amin=1e-100, amax=1e100)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        xk = xk + alpha_k * pk
        if retall:
            allvecs.append(xk)
        if gfkp1 is None:
            gfkp1 = myfprime(xk)
        yk = gfkp1 - gfk
        beta_k = max(0, numpy.dot(yk, gfkp1) / deltak)
        pk = -gfkp1 + beta_k * pk
        gfk = gfkp1
        gnorm = vecnorm(gfk, ord=norm)
        if callback is not None:
            callback(xk)
        k += 1

    fval = old_fval
    if warnflag == 2:
        msg = _status_message['pr_loss']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    result = OptimizeResult(fun=fval, jac=gfk, nfev=func_calls[0],
                            njev=grad_calls[0], status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in Scipy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)


def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper


def vecnorm(x, ord=2):
    if ord == Inf:
        return numpy.amax(numpy.abs(x))
    elif ord == -Inf:
        return numpy.amin(numpy.abs(x))
    else:
        return numpy.sum(numpy.abs(x)**ord, axis=0)**(1.0 / ord)


class _LineSearchError(RuntimeError):
    pass


def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.

    Raises
    ------
    _LineSearchError
        If no suitable step size is found

    """
    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LineSearchWarning)
            ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                     old_fval, old_old_fval)

    if ret[0] is None:
        raise _LineSearchError()

    return ret


class OptimizeResult(dict):
    """ Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
