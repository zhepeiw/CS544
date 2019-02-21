import numpy as np

class ICA():
    def __init__(self, X, lamb=1e-1, ica_mode='full'):
        self.X = X
        self.lamb = lamb
        self.mode = ica_mode
        if ica_mode == 'full' or ica_mode == 'known_mix':
            self.f = np.tanh
            self.g = lambda x : x**2
            self.fd = lambda x : 1 - np.tanh(x)**2
            self.gd = lambda x : 2*x
            self.fdd = lambda x : -2*np.tanh(x)*(1 - np.tanh(x)**2)
            self.gdd = lambda x : 2*np.ones_like(x)
        elif ica_mode == 'pca':
            self.f = lambda x : x
            self.g = lambda x : x
            self.fd = lambda x : np.ones_like(x)
            self.gd = lambda x : np.ones_like(x)
            self.fdd = lambda x : np.zeros_like(x)
            self.gdd = lambda x : np.zeros_like(x)

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
        c1 = np.dot(self.f(S[0]), self.g(S[1]))
        c2 = np.dot(self.f(S[1]), self.g(S[0]))
        S_grad[0] += self.lamb*c1*self.g(S[1])*self.fd(S[0])
        S_grad[0] += self.lamb*c2*self.gd(S[0])*self.f(S[1])
        S_grad[1] += self.lamb*c2*self.g(S[0])*self.fd(S[1])
        S_grad[1] += self.lamb*c1*self.gd(S[1])*self.f(S[0])
        grads = np.zeros_like(vars)
        if self.mode != 'known_mix':
            grads[:4] = A_grad.reshape(-1)
        grads[4:] = S_grad.reshape(-1)
        return grads

    def hessian_l1(self, vars):
        A = vars[:4].reshape(2, 2)
        S = vars[4:].reshape(2, -1)
        T = S.shape[1]

        # top left
        hessian_mat = np.zeros((len(vars), len(vars)))
        s_squared = S @ S.T
        hessian_mat[:2, :2] = s_squared
        hessian_mat[2:4, 2:4] = s_squared

        # bottom right
        a_squared = A.T @ A
        id_T = np.eye(T)
        hessian_mat[4:4+T, 4:4+T] = a_squared[0, 0] * id_T
        hessian_mat[4:4+T, 4+T:] = a_squared[0, 1] * id_T
        hessian_mat[4+T:, 4:4+T] = a_squared[1, 0] * id_T
        hessian_mat[4+T:, 4+T:] = a_squared[1, 1] * id_T

        # other terms
        hessian_mat[0, 4:4+T] = -self.X[0]+2*A[0,0]*S[0]+A[0,1]*S[1]
        hessian_mat[2, 4:4+T] = -self.X[1]+2*A[1,0]*S[0]+A[1,1]*S[1]
        hessian_mat[1, 4:4+T] = A[0,0]*S[1]
        hessian_mat[3, 4:4+T] = A[1,0]*S[1]

        hessian_mat[1, 4+T:] = -self.X[0]+A[0,0]*S[0]+2*A[0,1]*S[1]
        hessian_mat[3, 4+T:] = -self.X[1]+A[1,0]*S[0]+2*A[1,1]*S[1]
        hessian_mat[0, 4+T:] = A[0,1]*S[0]
        hessian_mat[2, 4+T:] = A[1,1]*S[0]

        hessian_mat[4:, :4] = hessian_mat[:4, 4:].T
        return hessian_mat

    def hessian_l2(self, vars):
        S = vars[4:].reshape(2, -1)
        T = S.shape[1]
        hessian_mat = np.zeros((len(vars), len(vars)))
        sfg01 = np.dot(self.f(S[0]), self.g(S[1]))
        sfg10 = np.dot(self.f(S[1]), self.g(S[0]))
        f, fd, fdd, g, gd, gdd = self._precompute_derivatives(S)
        hessian_mat[4:4+T, 4:4+T] = self._s1s1_hessian(S, f, fd, fdd, g, gd,
                                                      gdd, sfg01, sfg10)
        hessian_mat[4+T:, 4+T:] = self._s2s2_hessian(S, f, fd, fdd, g, gd,
                                                      gdd, sfg01, sfg10)
        hessian_mat[4:4+T, 4+T:] = self._s1s2_hessian(S, f, fd, fdd, g, gd,
                                                      gdd, sfg01, sfg10).T
        hessian_mat[4+T:, 4:4+T] = hessian_mat[4:4+T, 4+T:].T
        return hessian_mat

    def hessian(self, vars):
        temp = self.hessian_l1(vars) + self.lamb * self.hessian_l2(vars)
        hess = np.zeros_like(temp)
        if self.mode == 'known_mix':
            hess[4:, 4:] = temp[4:, 4:]
        else:
            hess = temp
        return hess

    def _precompute_derivatives(self, S):
        f = self.f(S)
        fd = self.fd(S)
        fdd = self.fdd(S)
        g = self.g(S)
        gd = self.gd(S)
        gdd = self.gdd(S)
        return f, fd, fdd, g, gd, gdd

    def _s1s1_hessian(self, S, f, fd, fdd, g, gd, gdd, sfg01, sfg10):
        s1s1_hessian = np.outer(fd[0], fd[0]) * np.outer(g[1], g[1])
        s1s1_hessian += np.outer(gd[0], gd[0]) * np.outer(f[1], f[1])
        s1s1_hessian += np.diag(sfg01 * fdd[0] * g[1])
        s1s1_hessian += np.diag(sfg10 * gdd[0] * f[1])
        return s1s1_hessian

    def _s2s2_hessian(self, S, f, fd, fdd, g, gd, gdd, sfg01, sfg10):
        s2s2_hessian = np.outer(fd[1], fd[1]) * np.outer(g[0], g[0])
        s2s2_hessian += np.outer(gd[1], gd[1]) * np.outer(f[0], f[0])
        s2s2_hessian += np.diag(sfg10 * fdd[1] * g[0])
        s2s2_hessian += np.diag(sfg01 * gdd[1] * f[0])
        return s2s2_hessian

    def _s1s2_hessian(self, S, f, fd, fdd, g, gd, gdd, sfg01, sfg10):
        s1s2_hessian = np.outer(f[0], fd[0]) * np.outer(gd[1], g[1])
        s1s2_hessian += np.outer(fd[1], f[1]) * np.outer(g[0], gd[0])
        s1s2_hessian += np.diag(sfg01 * fd[0] * gd[1])
        s1s2_hessian += np.diag(sfg10 * gd[0] * fd[1])
        return s1s2_hessian
