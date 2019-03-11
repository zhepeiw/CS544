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

        returns:
            Ax: sparse matrix with shape n_points**2 x n_points**2
            Ay: sparse matrix with shape n_points**2 x n_points**2
    '''
    Ax = -sparse.eye(n_points**2, format='csr')
    Ax += sparse.diags([1 for _ in range(n_points**2 - 1)], offsets=1, format='csr')
    Ax[n_points-1::n_points] = Ax[n_points-2::n_points]

    Ay = -sparse.eye(n_points**2, format='csr')
    Ay += sparse.diags([1 for _ in range(n_points * (n_points - 1))], 
                       offsets=n_points, format='csr')
    Ay[n_points*(n_points-1):] = Ay[n_points*(n_points-2):n_points*(n_points-1)]

    return Ax, Ay

def get_constraints(G, H, n_points):
    '''
        

        returns:
            L_sp: sparse matrix with shape n_c x n_points**2
            C_sp: sparse vector with shape n_c x 1
    '''

    L_sp = sparse.csr_matrix((len(H), n_points**2))
    for (i, grid) in enumerate(G):
        x = int(grid[0] * (n_points-1))
        y = int(grid[1] * (n_points-1))
        idx = x * n_points + y
        L_sp[i, idx] = 1
    
    C_sp = sparse.csc_matrix(np.expand_dims(np.array(H), axis=1).reshape(-1, 1))

    return L_sp, C_sp

def fn_smooth(h, M, L, C, lamb, ro):
    '''
        

    '''
    f = 0.5 * h.T @ M @ h
    g = L @ h - C
    return f - lamb.T @ g + 0.5 * ro * g.T @ g

if __name__ == "__main__":
    #  Ax, Ay = get_finite_diff(256)

    G = [
        (0, 0),
        (0, 1/2),
        (0, 1),
        (1/2, 0),
        (1/2, 1/2),
        (1/2, 1),
        (1, 0),
        (1, 1/2),
        (1, 1)
    ]

    H = [1, 0, 1, 0, 1, 0, 1, 0, 1]
    L, C = get_constraints(G, H, 256)

    pdb.set_trace()

