import numpy as np
import sys, os
import pdb

from utils import plot_surface, get_finite_diff, get_constraints
from solvers import aug_lag_solver, lin_solver


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
    V = [1, 0, 1, 0, 1, 0, 1, 0, 1]
    L, C = get_constraints(G, V, 256)
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
    V = [1, 0, 1, 0, 1, 0, 1, 0, 1]
    L, C = get_constraints(G, V, 256)

    h = lin_solver(Ax, Ay, L, C) 
    plot_surface(h, file_path='./pr2.pdf')


if __name__ == "__main__":
    smoothness_lin()
    pdb.set_trace()

