import numpy as np
from scipy import optimize, sparse
from utils import fn_smooth


def aug_lag_solver(Ax, Ay, L, C, lamb, ro=1, r=2, 
                   n_epochs=10, thresh=1e-3, mode='smooth'):
    """
       Solver for Augmented Lagrangian Method

       args:
           Ax: matrix finite difference approximation on x with shape n_points**2 x n_points**2
           Ay: matrix finite difference approximation on y
           L: constraint matrix with shape n_c x n_points**2
           C: constraint value vector with shape n_c x 1
           lamb: vector of lagrange multipliers for the linear term with shape n_c x 1
           ro: scalar weight for the quadratic term
           r: multiplication factor for ro

           mode: smooth or min_surf

        returns:
            h: vector of optimal values with shape n_points**2 x 1

    """
    assert mode in ['smooth', 'min_surf']

    loss = np.infty
    M = Ax.T @ Ax + Ay.T @ Ay
    for epoch in range(n_epochs):
        # solve for h and estimate loss
        if mode == 'smooth':
            h = sparse.linalg.spsolve(M + ro * L.T @ L, ro * L.T @ C + L.T @ lamb)
            h = np.expand_dims(h, 1)
            curr_loss = fn_smooth(h, M, L, C, lamb, ro)
        else:
            #  TODO: solving h for the min_surf case
            raise NotImplementedError
        if abs(curr_loss - loss) <= thresh:
            print('Stopped at {} iteration with loss difference {}'.format(epoch, abs(curr_loss-loss)))
            break
        # update lamb and ro
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
