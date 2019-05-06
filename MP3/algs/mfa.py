import numpy as np
import pdb

def calculate_E(Y, Q, lamb):
    '''
        args:
            Y: n x n integer array
            Q: n x n x n_class integer array 

        returns:
            E: n x n x n_class array
    '''
    E = np.zeros_like(Q)

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # calculate expectation of EQ[log(Q)]
            eqlogq = Q[i, j] * np.log(Q[i, j])
            # calculate hidden nbr cost 
            hh = np.zeros_like(eqlogq)
            if i != 0:
                Hj = Q[i - 1, j].argmax()
                temp = -np.ones_like(hh)
                temp[Hj] = 1
                hh += lamb * temp 
            if i != Y.shape[0] - 1:
                Hj = Q[i + 1, j].argmax()
                temp = -np.ones_like(hh)
                temp[Hj] = 1
                hh += lamb * temp 
            if j != 0:
                Hj = Q[i, j - 1].argmax()
                temp = -np.ones_like(hh)
                temp[Hj] = 1
                hh += lamb * temp 
            if j != Y.shape[1] - 1:
                Hj = Q[i, j + 1].argmax()
                temp = -np.ones_like(hh)
                temp[Hj] = 1
                hh += lamb * temp 
            # calculate observation vs hidden cost 
            hx = -np.ones_like(hh)
            hx[Y[i, j]] = 1
            E[i, j] = eqlogq - Q[i, j] * (hh + hx)

    return E   



def calculate_Q(E):
    '''
        
        args:
            E: n x n x n_class

    '''
    exp_nE = np.exp(-E)
    Z = exp_nE.sum(axis=-1, keepdims=True)

    Q = exp_nE / Z
    return Q

def Mean_Field_Approx(Y, n_class=32, lamb=0.9, thresh=1e-5):
    #  Q = np.random.rand(*Y.shape, n_class)
    #  Q = Q / Q.sum(axis=-1, keepdims=True)
    Q = np.zeros(Y.shape + (32,))
    # one-hot initialize Q to be the observation
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            Q[i, j, Y[i, j]] = 1
    # smooth by softmax
    Q = np.exp(Q) / np.exp(Q).sum(axis=-1, keepdims=True)
    
    energy_prev = np.inf
    while True:
        E = calculate_E(Y, Q, lamb)
        Q = calculate_Q(E)
        if energy_prev - E.sum() < thresh:
            break
        energy_prev = E.sum()
        print('Energy: {}'.format(E.sum()))
    
    Y_pred = Q.argmax(axis=-1)

    return Y_pred

if __name__ == '__main__':
    pass
