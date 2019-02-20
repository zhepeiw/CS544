from prplus import prplus
from newtoncg import newtoncg
import numpy as np
import pdb

args = (4, 12, 9, 4, 6, 2)  # parameter values
def f(x, *args):
    u, v = x
    a, b, c, d, e, f = args
    return a*u**2 + b*u*v + c*v**2 + d*u + e*v + f
def gradf(x, *args):
    u, v = x
    a, b, c, d, e, f = args
    gu = 2*a*u + b*v + d     # u-component of the gradient
    gv = b*u + 2*c*v + e     # v-component of the gradient
    return np.asarray((gu, gv))
def hessf(x, *args):
    u, v = x
    a, b, c, d, e, f = args
    return [[2*a, b],
            [b, 2*c]]
x0 = np.asarray((-100, 33))  # Initial guess.

print("Newton:")
res = newtoncg(f, x0, args, gradf, hessf, return_all=True)
allvecs = res['allvecs']
fun = res['fun']
jac = res['jac']
x = res['x']
for log in allvecs:
    k, time, x_k, grad, hess = log
    print("k={}\ttime={:.8f}\tx_k={}\tf_k={}\tgrad={}\thess={}".format(k, time, x_k, f(x_k, *args), grad, hess))
    
print()
print("Polak-Ribiere:")
xopt, fopt, n_f_eval, n_grad_eval, status, all_values = prplus(f, x0, fprime=gradf, args=args, stop_maxiter=200, restart_min_moment=-1, restart_gtol=0.00000000001, retall=True, full_output=True)
for log in all_values:
    k, time, x_k, p_k, gfk, beta_k = log
    print("k={}\ttime={:.8f}\tx_k={}\tf_k={:.8f}\tp_k={}\tgrad={}\tbeta_k={:.8f}".format(k, time, x_k, f(x_k,*args), p_k, gfk, beta_k))
