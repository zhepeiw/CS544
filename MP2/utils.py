import numpy as np
import sys, os
import pdb

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
    h = h.reshape(256, 256)

    #  fig = plt.figure()
    #  ax = fig.gca(projection='3d')
    #  surf = ax.plot_surface(X, Y, h, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #  fig.colorbar(surf, shrink=0.5, aspect=5)
    #  plt.show()
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
        ax.plot_surface(X, Y, h, rstride=10, cstride=10, alpha=0.8, cmap=cm.coolwarm)
        ax.contourf(X, Y, h, zdir='z', offset=z_min, cmap=cm.coolwarm)
        #  ax.contourf(X, Y, h, zdir='x', offset=x_min, cmap=cm.coolwarm)

        ax.set_xlabel('X')
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel('Y')
        ax.set_ylim(y_min, y_max)
        ax.set_zlabel('H')
        ax.set_zlim(z_min, z_max)

    plt.savefig(file_path, dpi=80)
    plt.close() 
