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
    #  plt.ylim(0, 0.1)
    plt.xlabel('Time (s)')
    plt.ylabel('Loss')
    plt.title('Loss Curve for {}'.format(expr_name))
    plt.grid(True)
    plt.savefig('../out/{}.pdf'.format(expr_name))

if __name__ == '__main__':
    args = get_args()
    pdb.set_trace()
    paths = glob.iglob(os.path.join(args.dir, "*.npz"))
    expr_name = args.name
    plot_experiment(paths, expr_name)


