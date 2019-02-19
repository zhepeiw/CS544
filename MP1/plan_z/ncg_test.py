from scipy import optimize
import numpy as np
import pdb

def ros_fn(x):
    return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2

def ros_jac_fn(x):
    return np.array([-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)])

def ros_hess_fn(x):
    return np.array([[1 - 4*x[1] + 12*x[0]**2, -4*x[0]], [-4*x[0], 2]])


def my_fn(x):
    fun = 0
    for i in range(len(x) - 1):
        fun += x[i]**4 + x[i + 1]**4
    
    return fun

def my_jac_fn(x):
    jac = np.empty((len(x),))
    for i in range(len(x)):
        if i == 0:
            jac[i] = 4 * x[i]**3 * x[i + 1]**4
        elif i == len(x) - 1:
            jac[i] = 4 * x[i]**3 * x[i - 1]**4
        else:
            jac[i] = 4 * x[i]**3 * (x[i - 1]**4 + x[i + 1]**4)

    return jac

def my_hess_fn(x):
    hess = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        if i == 0:
            hess[i, i] = 12 * x[i]**2 * x[i + 1]**4
        elif i == len(x) - 1:
            hess[i, i] = 12 * x[i]**2 * x[i - 1]**4
        else:
            hess[i, i] = 12 * x[i]**2 * (x[i - 1]**4 + x[i + 1]**4)

        if i != len(x) - 1:
            hess[i, i + 1] = hess[i + 1, i] = 16 * x[i]**3 * x[i + 1]**3

    return hess


if __name__ == '__main__':
    x = np.random.randn(10)
    #  x = np.array([
    #      [1, -1],
    #      [-1, 1]
    #  ])
    pdb.set_trace()
    #  res = optimize.minimize(ros_fn, x, method='Newton-CG', jac=ros_jac_fn, hess=ros_hess_fn)
    res = optimize.fmin_ncg(my_fn, x, fprime=my_jac_fn, fhess=my_hess_fn, full_output=False, retall=False)
    pdb.set_trace()



