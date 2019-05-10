import numpy as np
import networkx as nx
import algs.graphcut as graphcut

def alpha_expansion(I, centers, classes):
    hidden = initialize(I)
    e_old = float('inf')
    e_new = graphcut.energy(I, centers, hidden)
    while e_new < e_old:
        e_old = e_new
        for alpha in range(len(centers)):
            alpha_expansion_move(I, centers, hidden, alpha)
        e_new = graphcut.energy(I, centers, hidden)
    return hidden

def initialize(I):
    return np.copy(I)

def alpha_expansion_move(I, centers, hidden, alpha):
    G = create_graph(I, centers, hidden, alpha)
    _, sets = nx.minimum_cut(G, 'aux,alpha', 'aux,non_alpha', capacity='weight')
    if 'aux,non_alpha' in sets[0]:
        s1 = sets[0]
        s2 = sets[1]
    else:
        s1 = sets[1]
        s2 = sets[0]
    for v in s1:
        split = v.split(',')
        type = split[0]
        if type == 'pixel':
            p = [int(split[1]), int(split[2])]
            hidden[p[0]][p[1]] = alpha
    return hidden

def create_graph(I, centers, hidden, alpha):
    neighborhood = np.array([[1,0],[0,1]])
    INF = 100.0 * I.shape[0] * I.shape[1] * 4
    G = nx.Graph()
    # Adding nodes
    G.add_node('aux,alpha')
    G.add_node('aux,non_alpha')
    p = [0, 0]
    for p[0] in range(I.shape[0]):
        for p[1] in range(I.shape[1]):
            G.add_node('pixel,{},{}'.format(p[0], p[1]))
    # Adding edges
    for p[0] in range(I.shape[0]):
        for p[1] in range(I.shape[1]):
            G.add_edge('aux,alpha', 'pixel,{},{}'.format(p[0], p[1]), weight=graphcut.energy_data(I, centers, p, alpha))
            w = INF if hidden[p[0]][p[1]] == alpha else graphcut.energy_data(I, centers, p, hidden[p[0]][p[1]])
            G.add_edge('aux,non_alpha', 'pixel,{},{}'.format(p[0], p[1]), weight=w)
            for d in neighborhood:
                q = p + d
                if q[0] < 0 or q[0] >= I.shape[0] or q[1] < 0 or q[1] >= I.shape[1]:
                    continue
                if hidden[p[0]][p[1]] == hidden[q[0]][q[1]]:
                    G.add_edge('pixel,{},{}'.format(p[0], p[1]), 'pixel,{},{}'.format(q[0], q[1]), weight=graphcut.energy_smoothness(I, centers, hidden[q[0]][q[1]], alpha))
                else:
                    G.add_node('aux,{},{},{},{}'.format(p[0], p[1], q[0], q[1]))
                    G.add_edge('pixel,{},{}'.format(p[0], p[1]), 'aux,{},{},{},{}'.format(p[0], p[1], q[0], q[1]), weight=graphcut.energy_smoothness(I, centers, hidden[p[0]][p[1]], alpha))
                    G.add_edge('pixel,{},{}'.format(q[0], q[1]), 'aux,{},{},{},{}'.format(p[0], p[1], q[0], q[1]), weight=graphcut.energy_smoothness(I, centers, hidden[q[0]][q[1]], alpha))
                    G.add_edge('aux,non_alpha', 'aux,{},{},{},{}'.format(p[0], p[1], q[0], q[1]), weight=graphcut.energy_smoothness(I, centers, hidden[p[0]][p[1]], hidden[q[0]][q[1]]))
    return G
