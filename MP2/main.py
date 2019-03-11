import numpy as np
import sys, os
import pdb
from scipy import optimize, sparse

from utils import plot_surface

def fn_smooth(h, M, L, C, lamb, ro, add_constr=True):
    '''
        Smoothness cost
        
        Args:
            h
            M
            L
            C
            lamb
            ro
            add_constr: boolean, whether or not to include 
                contraint terms in the cost
        
        Returns:
            Cost value

    '''
    f = 0.5 * h.T @ M @ h
    if not add_constr:
        return f
    g = L @ h - C
    return f - lamb.T @ g + 0.5 * ro * g.T @ g

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
            C: vector with shape n_c x 1
    '''

    L_sp = sparse.csr_matrix((len(H), n_points**2))
    for (i, grid) in enumerate(G):
        x = int(grid[0] * (n_points-1))
        y = int(grid[1] * (n_points-1))
        idx = x * n_points + y
        L_sp[i, idx] = 1
    
    #  C_sp = sparse.csc_matrix(np.expand_dims(np.array(H), axis=1).reshape(-1, 1))
    C = np.expand_dims(np.array(H), axis=1).reshape(-1, 1)

    return L_sp, C

def aug_lag_solver(Ax, Ay, L, C, lamb, ro=1, r=2, n_epochs=10, thresh=1e-3):
    """
       Solver for Augmented Lagrangian Method

       args:
           f: the objective function that takes x
           fprime: gradient of f
           g: the constraint function that takes x and returns a vector
           x: parameter to optimize
           lamb: vector of lagrange multipliers for the linear term
           ro: scalar weight for the quadratic term
           r: multiplication factor for ro

        returns:

    """
    loss = np.infty
    M = Ax.T @ Ax + Ay.T @ Ay
    for epoch in range(n_epochs):
        h = sparse.linalg.spsolve(M + ro * L.T @ L, ro * L.T @ C + L.T @ lamb)
        h = np.expand_dims(h, 1)
        curr_loss = fn_smooth(h, M, L, C, lamb, ro)
        if abs(curr_loss - loss) <= thresh:
            print('Stopped at {} iteration with loss difference {}'.format(epoch, abs(curr_loss-loss)))
            break
        loss = curr_loss
        lamb -= ro / 2 * (L @ h - C)
        ro *= r

    return h

def lin_solver(Ax, Ay, L, C):
    '''
        Solver of smoothness cost using linear system  
    '''
    M = Ax.T @ Ax + Ay.T @ Ay 
    B_left = sparse.bmat([[M, -L.T], [L, sparse.csr_matrix((L.shape[0], L.shape[0]))]])
    B_right = sparse.csc_matrix((M.shape[1] + L.shape[0], 1))
    B_right[-L.shape[0]:] = C
    res = sparse.linalg.spsolve(B_left, B_right)
    h = np.expand_dims(res[:-L.shape[0]], axis=1)
    return h

def smoothness_aug_lag():
    '''
        Driver function to run part 1
    '''
    Ax, Ay = get_finite_diff(256)
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
    lamb0 = np.zeros((9, 1))
    
    h = aug_lag_solver(Ax, Ay, L, C, lamb0)
    plot_surface(h, file_path='./pr1.pdf')

def smoothness_lin():
    '''
        Driver function for part 2
    '''
    Ax, Ay = get_finite_diff(256)
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

    h = lin_solver(Ax, Ay, L, C) 
    plot_surface(h, file_path='./pr2.pdf')


if __name__ == "__main__":
    smoothness_aug_lag()
    pdb.set_trace()

