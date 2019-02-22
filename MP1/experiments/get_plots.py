import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, glob
from argparse import ArgumentParser, Namespace
import pdb

def get_args():
    parser = ArgumentParser(description='Argument parser for plotting')
    parser.add_argument("--dir", type=str,
                        help="""Path to outputs""",
                        default="../out/")
    parser.add_argument("--name", type=str,
                        help="""Name of the experiment""",
                        required=True)

    return parser.parse_args()

def plot_experiment(paths, expr_name, out_dir='../out'):
    matplotlib.rcParams.update({'font.size': 17.5})
    plt.figure(figsize=(10, 10))
    min_t = np.inf
    for path in paths:
        f = np.load(path)
        curr_t = f['times']
        min_t = min(max(curr_t), min_t)
    for path in sorted(paths):
        name = os.path.basename(path)
        method = name.split('_')[-2]
        if method == 'pr':
            restart = name.split('_')[-1].split('.')[0]
            method = 'pr-{}'.format(restart)
        f = np.load(path)
        curr_ls = f['losses']
        curr_t = f['times']
        if curr_t[-1] <= min_t:
            cutoff = len(curr_t)
        else:
            cutoff = np.argmax(curr_t > min_t)
        if cutoff == 1:
            curr_t = [0, min_t]
            curr_ls = [curr_ls[0], curr_ls[0]]
            cutoff = 2
        plt.plot(curr_t[:cutoff], np.log(curr_ls)[:cutoff], label=method)
    plt.legend(loc='best')
    #  plt.ylim(0, 0.1)
    plt.xlabel('Time (s)')
    plt.ylabel('Log Loss')
    plt.title('Log-Loss Curve for {}'.format(expr_name))
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, '{}.pdf'.format(expr_name)))

if __name__ == '__main__':
    args = get_args()
    paths = glob.glob(os.path.join(args.dir, "*.npz"))
    expr_name = args.name
    plot_experiment(paths, expr_name, args.dir)


