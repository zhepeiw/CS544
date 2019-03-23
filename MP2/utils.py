import numpy as np
import sys, os
import pdb
from scipy import sparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_surface(h, file_path, nrows=1, ncols=3):
    '''
        Plot the surface with height values h
    '''
    matplotlib.rcParams.update({'font.size': 17.5})
    X = np.linspace(0, 1, 256)
    Y = np.linspace(0, 1, 256)
    X, Y = np.meshgrid(X, Y)
    h = h.reshape(256, 256).T

    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    z_min, z_max = np.min(h), np.max(h)

    fig = plt.figure(figsize=(30, 10))
    angles = np.linspace(-45, 45, ncols).astype(np.int)
    elevs = np.linspace(30, 0, nrows).astype(np.int)
    for n in range(nrows * ncols):
        i = n % ncols
        j = n // ncols
        azim = angles[i]
        elev = elevs[j]
        ax = fig.add_subplot(nrows, ncols, n+1, projection='3d')
        ax.set_title("angle=" + str(azim) + " height=" + str(elev))
        ax.tick_params(labelsize=8)
        ax.view_init(azim=azim, elev=elev)
        ax.plot_surface(X, Y, h, rstride=10, cstride=10, alpha=0.8,
                        cmap=cm.bone_r, antialiased=False, linewidth=0)
        ax.contourf(X, Y, h, zdir='z', offset=z_min, cmap=cm.bone_r)
        #  ax.contourf(X, Y, h, zdir='x', offset=x_min, cmap=cm.coolwarm)

        ax.set_xlabel('X')
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel('Y')
        ax.set_ylim(y_min, y_max)
        ax.set_zlabel('H')
        ax.set_zlim(z_min, z_max)

    plt.savefig(file_path, dpi=80)
    plt.close()

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

def fn_area(h, N, L, C, lamb, ro, add_constr=True):
    '''
        Smoothness cost

        Args:
            h
            N
            L
            C
            lamb
            ro
            add_constr: boolean, whether or not to include
                contraint terms in the cost

        Returns:
            Cost value

    '''
    v = sparse.bmat([[N @ h], [sparse.csr_matrix(np.reshape([1]*h.shape[0],(-1,1)))]])
    f = v.T @ v
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
