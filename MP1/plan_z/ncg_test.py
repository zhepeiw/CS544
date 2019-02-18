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
    return (x ** 4).sum()

def my_jac_fn(x):
    pdb.set_trace()
    return np.array([
        4 * x[0, 0]**3, 4 * x[0, 1]**3, 4 * x[1, 0]**3, 4 * x[1, 1]**3
    ])

def my_hess_fn(x):
    return np.array([
        [12 * x[0, 0]**2, 0, 0, 0],
        [0, 12 * x[0, 1]**2, 0, 0],
        [0, 0, 12 * x[1, 0]**2, 0],
        [0, 0, 0, 12 * x[1, 1]**2]
    ])

if __name__ == '__main__':
    x = np.array([2, -1])
    #  x = np.array([
    #      [1, -1],
    #      [-1, 1]
    #  ])
    pdb.set_trace()
    #  res = optimize.minimize(ros_fn, x, method='Newton-CG', jac=ros_jac_fn, hess=ros_hess_fn)
    res = optimize.fmin_ncg(ros_fn, x, fprime=ros_jac_fn, fhess=ros_hess_fn, full_output=True, retall=True)
    pdb.set_trace()



