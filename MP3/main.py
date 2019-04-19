import torch
import numpy as np
import sys, os, pdb 
from utils import prepare_dataset, cluster_noise

if __name__ == '__main__':
    if not os.path.exists('./data/mp4_data.t'):
        ds = prepare_dataset()
        torch.save(ds, './data/mp4_data.t')
    else:
        ds = torch.load('./data/mp4_data.t')
    y_km, y_ns, centers = cluster_noise(ds[0])

