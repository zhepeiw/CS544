import numpy as np
import networkx as nx
import algs.graphcut as graphcut

def alpha_beta_swap(I, centers, classes):
    hidden = initialize(I)
    e_old = float('inf')
    e_new = graphcut.energy(I, centers, hidden)
    while e_new < e_old:
        e_old = e_new
        for alpha in range(32):
            for beta in range(alpha):
                hidden = alpha_beta_swap_move(I, centers, hidden, alpha, beta)
        e_new = graphcut.energy(I, centers, hidden)
    return hidden

def initialize(I):
    return np.copy(I)

def alpha_beta_swap_move(I, centers, hidden, alpha, beta):
    G = create_graph(I, centers, hidden, alpha, beta)
    _, sets = nx.minimum_cut(G, 'aux,alpha', 'aux,beta', capacity='weight')
    if 'aux,alpha' in sets[0]:
        s1 = sets[0]
        s2 = sets[1]
    else:
        s1 = sets[1]
        s2 = sets[0]
    for v in s1:
        split = v.split(',')
        type = split[0]
        if type == 'pixel':
            py = int(split[1])
            px = int(split[2])
            p = [py, px]
            hidden[p[0]][p[1]] = beta
    for v in s2:
        split = v.split(',')
        type = split[0]
        if type == 'pixel':
            py = int(split[1])
            px = int(split[2])
            p = [py, px]
            hidden[p[0]][p[1]] = alpha
    return hidden

def create_graph(I, centers, hidden, alpha, beta):
    neighborhood = np.array([[1,0],[0,1],[-1,0],[0,-1]])
    INF = 100.0 * I.shape[0] * I.shape[1] * 4
    G = nx.Graph()
    # Adding nodes
    G.add_node('aux,alpha')
    G.add_node('aux,beta')
    p = [0, 0]
    for p[0] in range(I.shape[0]):
        for p[1] in range(I.shape[1]):
            if hidden[p[0]][p[1]] == alpha or hidden[p[0]][p[1]] == beta:
                G.add_node('pixel,{},{}'.format(p[0], p[1]))
    # Adding edges
    for p[0] in range(I.shape[0]):
        for p[1] in range(I.shape[1]):
            if hidden[p[0]][p[1]] == alpha or hidden[p[0]][p[1]] == beta:
                w1 = graphcut.energy_data(I, centers, p, alpha)
                w2 = graphcut.energy_data(I, centers, p, beta)
                for d in neighborhood:
                    q = p + d
                    if q[0] < 0 or q[0] >= I.shape[0] or q[1] < 0 or q[1] >= I.shape[1]:
                        continue
                    if hidden[q[0]][q[1]] != alpha and hidden[q[0]][q[1]] != beta:
                        w1 += graphcut.energy_smoothness(I, centers, alpha, hidden[q[0]][q[1]])
                        w2 += graphcut.energy_smoothness(I, centers, beta, hidden[q[0]][q[1]])
                G.add_edge('aux,alpha', 'pixel,{},{}'.format(p[0], p[1]), weight=w1)
                G.add_edge('aux,beta', 'pixel,{},{}'.format(p[0], p[1]), weight=w2)
                for d in neighborhood[:2,:]:
                    q = p + d
                    if q[0] < 0 or q[0] >= I.shape[0] or q[1] < 0 or q[1] >= I.shape[1]:
                        continue
                    if hidden[q[0]][q[1]] == alpha or hidden[q[0]][q[1]] == beta:
                        G.add_edge('pixel,{},{}'.format(p[0], p[1]), 'pixel,{},{}'.format(q[0], q[1]), weight=graphcut.energy_smoothness(I, centers, alpha, beta))
    return G
