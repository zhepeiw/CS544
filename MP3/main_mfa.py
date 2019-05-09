import torch
import numpy as np
import sys, os, pdb 
from utils import prepare_dataset, cluster_noise, visualize_result
from algs.mfa import Mean_Field_Approx
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if not os.path.exists('./data/mp4_data.t'):
        ds = prepare_dataset()
        torch.save(ds, './data/mp4_data.t')
    else:
        ds = torch.load('./data/mp4_data.t')

    lambs = np.linspace(0.1, 1.0, 10)
    accs = []
    for lamb in lambs:
        curr_accs = []
        for idx in range(10):
            print('==> Processing image {}'.format(idx))
            y_km, y_ns, centers = cluster_noise(ds[idx])
            y_re = Mean_Field_Approx(y_ns, 32, lamb)

            acc = (y_re == y_km).astype(np.float).mean()
            curr_accs.append(acc)
            #  visualize_result(y_km, y_ns, y_re, centers,
            #                   save_name='res_{}.pdf'.format(idx))
        stat = np.array(curr_accs).mean()
        accs.append(stat)
        print('==> Lambda = {}, acc = {} \n'.format(lamb, stat))

    plt.figure(figsize=(10, 10))
    plt.plot(lambs, accs)
    plt.grid(True)
    plt.xlabel('lambda')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs lambda')
    if not os.path.exists('out'):
        os.mkdir('out/')
    plt.savefig('out/lamd_acc.pdf')
    plt.close()

    #  lamb = lambs[np.argmax(np.array(accs))]
    #  lamb = 0.8
    #  for idx in range(10):
    #      print('==> Processing image {}'.format(idx))
    #      y_km, y_ns, centers = cluster_noise(ds[idx])
    #      y_re = Mean_Field_Approx(y_ns, 32, lamb)
    #      visualize_result(y_km, y_ns, y_re, centers,
    #                       save_name='res_{}.pdf'.format(idx))

