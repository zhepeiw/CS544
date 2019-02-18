import numpy as np

class ICA():
    def __init__(self, X, lamb=1e-1):
        self.X = X
        self.lamb = lamb
        self.f = np.tanh
        self.g = lambda x : x**2
        self.f_prime = lambda x : 1 - np.tanh(x)**2
        self.g_prime = lambda x : 2*x

    def loss(self, vars):
        '''
        vars : Minimization variables are a 4 + 2*T vector where T is the length of each of
        the input signals.
        '''
        A = vars[:4].reshape(2, 2)
        S = vars[4:].reshape(2, -1)
        reconstruction_loss = 0.5*np.sum((self.X - A@S)**2)
        correlation_mat = self.f(S) @ self.g(S).T
        diag_mat = np.diag(np.diag(correlation_mat))
        independence_loss = 0.5*np.sum((correlation_mat - diag_mat)**2)
        return reconstruction_loss + self.lamb*independence_loss

    def grads(self, vars):
        A = vars[:4].reshape(2, 2)
        S = vars[4:].reshape(2, -1)
        A_grad = -(self.X - A @ S) @ S.T
        S_grad = -A.T @ (self.X - A @ S)
        c1 = self.f(S[0]).T @ self.g(S[1])
        c2 = self.f(S[1]).T @ self.g(S[0])
        S_grad[0] += self.lamb*c1*self.g(S[1])*self.f_prime(S[0])
        S_grad[0] += self.lamb*c2*self.g_prime(S[0])*self.f(S[1])
        S_grad[1] += self.lamb*c2*self.g(S[0])*self.f_prime(S[1])
        S_grad[1] += self.lamb*c1*self.g_prime(S[1])*self.f(S[0])
        grads = np.zeros_like(vars)
        grads[:4] = A_grad.reshape(-1)
        grads[4:] = S_grad.reshape(-1)
        return grads
