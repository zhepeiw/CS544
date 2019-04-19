import numpy as np
import pdb

def calculate_E():
    pass

def calculate_Q():
    pass

def Mean_Field_Approx(Y, n_class=32, lamb=0.8):
    Q = np.zeros(Y.shape + (32,))
    # one-hot initialize Q to be the observation
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            Q[i, j, Y[i, j]] = 1
    
    pdb.set_trace()

    return Y

if __name__ == '__main__':
    pass
