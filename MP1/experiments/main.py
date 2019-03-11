import sys, os
import pdb
from argparse import ArgumentParser, Namespace
sys.path.append('../')

import numpy as np
from algorithms.newtoncg import newtoncg
from algorithms.prplus import prplus
from ica.ica import ICA
from data_loader.get_data import get_source_wavs

def get_args():
    """
        Command Line Argument Parser
    """
    parser = ArgumentParser(description='Argument Parser for CG vs PR')
    parser.add_argument("--n_samp", type=int,
                        help="""Number of samples per source""",
                        default=8000)
    parser.add_argument("--mode", type=str,
                        help="""Mode of model (ica, pca, or known_mix)""",
                        default='ica',
                        choices=['ica', 'pca', 'known_mix'])
    parser.add_argument("--alg", type=str,
                        help="""Algorithm: CG or PR""",
                        default='cg',
                        choices=['cg', 'pr'])
    parser.add_argument("--max_iter", type=int,
                        help="""Max iteration for PR""",
                        default=200)
    parser.add_argument("--min_gtol", type=float,
                        help="""Minimum tolerance in gradient in PR""",
                        default=1e-5)
    parser.add_argument("--min_moment", type=float,
                        help="""Minimum momentum for PR""",
                        default=np.NINF)
    parser.add_argument("--restart", type=str,
                        help="""Restart strategy for PR""",
                        default=''
                       )
    parser.add_argument('--out_dir', type=str,
                        help="""Output directory of the results""",
                        default='../out/')


    return parser.parse_args()


if __name__ == '__main__':
    # argparse
    args = get_args()

    # loading data 
    mixing_matrix = np.array([[0.8, 0.2],
                              [0.6, 0.4]])
    n_samples = args.n_samp
    normalize_by_std = False
    normalize_by_mean = True
    wavs_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../wavs_for_separation/')

    mixtures, mixing_matrix, sources = get_source_wavs(
        mixing_matrix=mixing_matrix,
        n_samples=n_samples,
        normalize_by_mean=normalize_by_mean,
        normalize_by_std=normalize_by_std,
        wavs_dir=wavs_dir)
    
    # initial guesses and model setup
    a = np.array([0.4, 0.6, 0.7, 0.3])

    t = np.linspace(-1, 1, n_samples)
    s1 = np.sin(t)
    s2 = np.cos(t)
    x1 = mixtures[0]
    x2 = mixtures[1]
    if args.mode != 'known_mix':
        v = np.concatenate([a, s1, s2])
    else:
        v = np.concatenate([s1, s2])
    X = np.stack([x1, x2])
    mode = args.mode
    if mode == 'ica':
        mode = 'full'
    model = ICA(X, A=mixing_matrix if mode == 'known_mix' else None,
                lamb=0 if mode == 'known_mix' else 1, ica_mode=mode)

    # optimization
    method = args.alg
    restart = args.restart

    if method == 'cg':
        res = newtoncg(model.loss, v, jac=model.grads, hess=model.hessian, return_all=True)
        losses = [model.loss(log[2]) for log in res['allvecs']]
        times = [log[1] for log in res['allvecs']]
        print('==> CG: Optimal loss: {}: #iters: {} time: {}'.format(losses[-1], 
                                                                     len(losses), times[-1]))

    elif method == 'pr':
        assert args.restart in ['iter', 'gtol', 'moment']
        max_iter, min_moment, min_gtol = None, np.NINF, 1e-10
        if args.restart == 'iter':
            max_iter = args.max_iter
        elif args.restart == 'gtol':
            min_gtol = args.min_gtol
        elif args.restart == 'moment':
            min_moment = args.min_moment 
        print(max_iter, min_gtol, min_moment)

        xopt, fopt, n_f_eval, n_grad_eval, status, res_cnt, all_values = prplus(model.loss, 
                                                                       v, 
                                                                       fprime=model.grads, 
                                                                       stop_maxiter=max_iter, 
                                                                       restart_min_moment=min_moment, 
                                                                       restart_gtol=min_gtol,
                                                                       retall=True, 
                                                                       full_output=True
                                                                      )
        losses = [model.loss(log[2]) for log in all_values]
        times = [log[1] for log in all_values]
        print('==> PR: Optimal loss: {}: #iters: {} time: {} #restarts: {}'.format(losses[-1], 
                                                                                   len(losses), 
                                                                                   times[-1],
                                                                                   res_cnt))

    # saving output
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    file_name = '{}_{}_{}_{}.npz'.format(args.mode, n_samples, method, restart)
    losses = [model.loss(v)] + losses
    times = [0] + times
    np.savez(os.path.join(out_dir, file_name), losses=losses, times=times)


