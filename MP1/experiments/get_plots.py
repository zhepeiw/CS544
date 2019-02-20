import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, glob
import pdb

def plot_experiment(paths, expr_name):
    plt.figure(figsize=(10, 10))
    for path in paths:
        method = path.split('_')[2]
        if method == 'pr':
            restart = path.split('_')[-1].split('.')[0]
            method = 'pr-{}'.format(restart)
        f = np.load(path)
        curr_ls = f['losses']
        curr_t = f['times']
        plt.plot(curr_t, curr_ls, label=method)
    plt.legend(loc='best')
    plt.xlabel('Time (s)')
    plt.ylabel('Loss')
    plt.title('Loss Curve for {}'.format(expr_name))
    plt.grid(True)
    plt.savefig('../out/{}.pdf'.format(expr_name))

if __name__ == '__main__':
    paths = glob.iglob('../out/*.npz')
    expr_name = 'ICA-Small'
    plot_experiment(paths, expr_name)


