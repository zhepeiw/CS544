import numpy as np
from algs.alphaexpansion import alpha_expansion, alpha_expansion_move
from algs.alphabetaswap import alpha_beta_swap, alpha_beta_swap_move

def energy_data(I, centers, p, alpha, lamb=2.75):
    return lamb * np.linalg.norm(centers[alpha] - centers[I[p[0]][p[1]]])

def energy_smoothness(I, centers, alpha, beta):
    return np.linalg.norm(centers[alpha] - centers[beta])

neighborhood = np.array([[1,0],[0,1]])
def energy(I, centers, hidden):
    e = 0
    for py in range(I.shape[0]):
        for px in range(I.shape[1]):
            p = [py, px]
            e += energy_data(I, centers, p, hidden[p[0]][p[1]])
            for d in neighborhood[:2,:]:
                q = p + d
                if q[0] < 0 or q[0] >= I.shape[0] or q[1] < 0 or q[1] >= I.shape[1]:
                    continue
                e += energy_smoothness(I, centers, hidden[p[0]][p[1]], hidden[q[0]][q[1]])
    return e

def Graph_Cut_Approx(Y, centers, classes, mode='mixed'):
    if mode == 'alpha-expansion':
        return alpha_expansion(Y, centers, classes)
    elif mode == 'alpha-beta-swap':
        return alpha_beta_swap(Y, centers, classes)
    elif mode == 'mixed':
        return mixed_optimization(Y, centers, classes)
    else:
        raise Exception('Invalid \'mode\' argument. Value \'{}\' is not recognized.'.format(mode))

def mixed_optimization(I, centers, classes):
    hidden = np.copy(I)
    e_old = float('inf')
    e_new = energy(I, centers, hidden)
    while e_new < e_old:
        e_old = e_new
        for alpha in range(32):
            hidden = alpha_expansion_move(I, centers, hidden, alpha)
            for beta in range(alpha):
                hidden = alpha_beta_swap_move(I, centers, hidden, alpha, beta)
        e_new = energy(I, centers, hidden)
    return hidden
