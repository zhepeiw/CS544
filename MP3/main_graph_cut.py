import torch
import numpy as np
import sys, os, pdb
from utils import prepare_dataset, cluster_noise, visualize_result
from algs.mfa import Mean_Field_Approx
import matplotlib.pyplot as plt
from algs.graphcut import Graph_Cut_Approx

if __name__ == '__main__':
    if not os.path.exists('./data/mp4_data.t'):
        ds = prepare_dataset()
        torch.save(ds, './data/mp4_data.t')
    else:
        ds = torch.load('./data/mp4_data.t')

    for idx in range(10):
        print('==> Processing image {}'.format(idx))
        y_km, y_ns, centers = cluster_noise(ds[idx])
        y_re = Graph_Cut_Approx(y_ns, centers, 32, mode='mixed')
        d = 0
        for i in range(32):
            for j in range(32):
                d += np.linalg.norm(centers[y_re[i][j]] - centers[y_km[i][j]])
        print('d=', d)
        visualize_result(y_km, y_ns, y_re, centers,
                         save_name='res_{}.pdf'.format(idx))
