import torch
import numpy as np
import sys, os, pdb 
from utils import prepare_dataset, cluster_noise
from algs.mfa import Mean_Field_Approx

if __name__ == '__main__':
    if not os.path.exists('./data/mp4_data.t'):
        ds = prepare_dataset()
        torch.save(ds, './data/mp4_data.t')
    else:
        ds = torch.load('./data/mp4_data.t')
    y_km, y_ns, centers = cluster_noise(ds[0])
    y_recons = Mean_Field_Approx(y_ns, 32, 0.8) 
    pdb.set_trace()

