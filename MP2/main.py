import numpy as np
import sys, os
import pdb

from utils import plot_surface, get_finite_diff, get_constraints
from solvers import aug_lag_solver, lin_solver


def smooth_aug_lag():
    '''
        Driver function to run part 1
    '''
    #  setting up constraints
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
    #  setting up other params
    Ax, Ay = get_finite_diff(256)
    lamb0 = np.zeros((9, 1))
    h = aug_lag_solver(Ax, Ay, L, C, lamb0, mode='smooth')
    plot_surface(h, file_path='./pr1.pdf')

def smooth_lin():
    '''
        Driver function for part 2
    '''
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
    #  setting up other params
    Ax, Ay = get_finite_diff(256)
    h = lin_solver(Ax, Ay, L, C) 
    plot_surface(h, file_path='./pr2.pdf')


def min_surf_edge():
    '''
        Driver function to run part 3
    '''
    #  setting up constraints using interpolation
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
    ] # TODO: rewrite this
    V = [1, 0, 1, 0, 1, 0, 1, 0, 1] # TODO: rewrite this
    L, C = get_constraints(G, V, 256)
    #  setting up other params
    Ax, Ay = get_finite_diff(256)
    lamb0 = np.zeros((256 * 6 - 9, 1))
    h = aug_lag_solver(Ax, Ay, L, C, lamb0, mode='min_surf')
    plot_surface(h, file_path='./pr3.pdf')

def min_surf_vert():
    '''
        Driver function to run part 4
    '''
    #  setting up constraints
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
    #  setting up other params
    Ax, Ay = get_finite_diff(256)
    lamb0 = np.zeros((9, 1))
    h = aug_lag_solver(Ax, Ay, L, C, lamb0, mode='min_surf')
    plot_surface(h, file_path='./pr4.pdf')

if __name__ == "__main__":
    smooth_aug_lag()
    pdb.set_trace()

