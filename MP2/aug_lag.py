import numpy as np
import sys, os
import pdb
from scipy import optimize, sparse

def aug_lag_solver(f, fprime, g, x, lamb, c, r=2, n_epochs=50):
    """
       Solver for Augmented Lagrangian Method

       args:
           f: the objective function that takes x
           fprime: gradient of f
           g: the constraint function that takes x and returns a vector
           x: parameter to optimize
           lamb: vector of lagrange multipliers for the linear term
           c: scalar weight for the quadratic term
           r: multiplication factor for c

        returns:

    """
    for epoch in range(n_epochs):
        aug_lag = lambda z: f(z) - lamb.T @ g(z) + c * g(z).T @ g(z)
        aug_lag_prime = lambda z: fprime(z) - lamb + c * g(z)

        res = optimize.fmin_bfgs(aug_lag, x, aug_lag_prime)
        x = res[0]
        lamb -= c / 2 * g(x)
        c *= r

    return x

def get_finite_diff(n_points):
    '''
        assuming h is in row-major order with [h(0, 0), h(0, 1), ... , h(0, n-1), ...]
    '''
    Ax = -sparse.eye(n_points**2, format='csr')
    Ax += sparse.diags([1 for _ in range(n_points**2 - 1)], offsets=1, format='csr')
    Ax[n_points-1::n_points] = Ax[n_points-2::n_points]

    Ay = -sparse.eye(n_points**2, format='csr')
    Ay += sparse.diags([1 for _ in range(n_points * (n_points - 1))], 
                       offsets=n_points, format='csr')
    Ay[n_points*(n_points-1):] = Ay[n_points*(n_points-2):n_points*(n_points-1)]

    #  Ax = np.zeros((n_points**2, n_points**2))
    #  for i in range(Ax.shape[0] - n_points):
    #      Ax[i, i] = -1
    #      Ax[i, i + n_points] = 1
    #  for i in range(Ax.shape[0] - n_points, Ax.shape[0]):
    #      Ax[i, i] = 1
    #      Ax[i, i - n_points] = -1

    #  Ay = np.zeros_like(Ax)
    #  for i in range(Ax.shape[0]):
    #      if (i + 1) % n_points != 0:
    #          Ay[i, i] = -1
    #          Ay[i, i + 1] = 1
    #      else:
    #          Ay[i, i] = 1
    #          Ay[i, i - 1] = -1

    return Ax, Ay



if __name__ == "__main__":
    Ax, Ay = get_finite_diff(4)
    pdb.set_trace()

