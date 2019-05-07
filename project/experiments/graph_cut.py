import sys, os
sys.path.append('../')

import numpy as np
import data_loader.data_loader as data_loader
from argparse import Namespace
import matplotlib.pyplot as plt
import maxflow
from librosa.core import stft, istft

def get_dataset():
    default_parameters = Namespace(batch_size=1, n_jobs=4, get_top=None)
    default_parameters.return_items = ['mixture_wav',
                                       'clean_sources_wavs',
                                       'rpd']
    default_parameters.input_dataset_p = '/mnt/data/CS544_data/timit_5400_1800_512_2_fm_random_taus_delays/val'
    data_gen = data_loader.get_numpy_data_generator(default_parameters)
    return data_gen

def build_graph_from_img(unary, binary, horiz_weight=0, vert_weight=0):
    height, width = binary.shape
    g = maxflow.Graph[float](height*width, height*width*2)
    g = maxflow.Graph[float]()


    nodeids = g.add_grid_nodes(binary.shape)
    horiz_diff = np.abs(binary[:, :-1] - binary[:, 1:])
    vert_diff = np.abs(binary[:-1] - binary[1:])
    horiz_diff = np.concatenate([horiz_diff, np.zeros((height, 1))], axis=1)
    vert_diff = np.concatenate([vert_diff, np.zeros((1, width))], axis=0)
    horiz_diff = horiz_diff.reshape(-1)
    vert_diff = vert_diff.reshape(-1)

    for row in nodeids:
        for nodeid in row:
            if nodeid % width != width - 1:
                weight = np.exp(-horiz_diff[nodeid]**2)
                weight *= horiz_weight
                g.add_edge(nodeid, nodeid + 1, weight, weight)

            if nodeid < width * (height - 1):
                weight = np.exp(-vert_diff[nodeid]**2)
                weight *= vert_weight
                g.add_edge(nodeid, nodeid + width, weight, weight)

    # make terminal edges
    g.add_grid_tedges(nodeids, unary, np.max(unary) - unary)
    return (g, nodeids)

def sisnr(s_pred, s, eps=10e-9):
    s_pred -= s_pred.mean()
    s -= s.mean()
    if not len(s_pred) == len(s):
        min_len = int(min(len(s), len(s_pred)))
        s = s[0:min_len]
        s_pred = s_pred[:min_len]
    coef = np.dot(s, s_pred) / (np.dot(s, s) + 10e-9)
    s_target = coef * s
    e_noise = s_pred - s_target
    sisnr = 10*np.log10(np.dot(s_target, s_target) /
                        (np.dot(e_noise, e_noise)+10e-9))
    return sisnr

def compute_sisnr_and_return_pair(s_recon, clean_sources, eps=10e-9):
    all_sisdrs = [(sisnr(s_recon, clean_sources[i], eps=eps), clean_sources[i])
                  for i in range(clean_sources.shape[0])]
    return sorted(all_sisdrs, key = lambda x: x[0])[-1]

def evaluate(data_gen, horiz_weight, vert_weight):
    total_sisnr = 0
    total_size = len(data_gen.dataset)
    for batch_data_list in data_gen:
        numpy_data_list = data_loader.convert_to_numpy(batch_data_list)
        mix_wav, clean_wavs, phase_diff = numpy_data_list
        mix_stft, mix_spec = get_normalized_spectrogram(mix_wav)
        g, nodeids = build_graph_from_img(phase_diff, mix_spec,
            horiz_weight=horiz_weight, vert_weight=vert_weight)
        g.maxflow()
        mask = g.get_grid_segments(nodeids)
        s1_estimate = istft(mix_stft * mask[::-1, :],
                    win_length=512,
                    hop_length=128)
        total_sisnr += compute_sisnr_and_return_pair(s1_estimate, clean_wavs)[0]
    return total_sisnr / total_size

def main():
    grid_size = 20
    data_gen = get_dataset()
    weights = np.log_scale = np.logspace(np.log10(1e-6), np.log10(2), grid_size)
    max_sisnr = 0
    best_weights = (0, 0)
    grid = np.zeros(grid_size, grid_size)
    for i, horiz_weight in enumerate(weights):
        for j, vert_weight in enumerate(weights):
            mean_sisnr = evaluate(data_gen, horiz_weight, vert_weight)
            grid[j, i] = mean_sisnr
            if mean_sisnr > max_sisnr:
                max_sisnr = mean_sisnr
                best_weights = (horiz_weight, vert_weight)
            print('H: {}, W: {}, SISNR: {}'.format(horiz_weight,
                vert_weight, mean_sisnr))
    np.savez('results.npz', grid=grid, X=weights, Y=weights)

if __name__=='__main__':
    main()
