import unittest
import numpy as np
import ica as ica

def finite_differences(x, f, delta=1e-3):
    '''
    Finds numerical gradients at f(x)
    x : input vector
    f : input function
    delta : difference
    '''
    numerical_grads = np.zeros_like(x)
    for i in range(len(x)):
        x[i] += delta
        plus = f(x)
        x[i] -= 2*delta
        minus = f(x)
        x[i] += delta
        numerical_grads[i] = (plus - minus) / (2 * delta)
    return numerical_grads

class TestICA(unittest.TestCase):
    def setUp(self):
        a = np.array([1, 2, 3, 4])
        t = np.linspace(-1, 1, 100)
        s1 = np.sin(t)
        s2 = np.tanh(t)
        x1 = s1 + s2
        x2 = s1 - s2
        self.vars = np.concatenate([a, s1, s2])
        self.known_vars = np.concatenate([s1, s2])
        X = np.stack([x1, x2])
        self.model = ica.ICA(X, lamb=1)
        self.known_model = ica.ICA(X, A=a.reshape(2, 2), lamb=1,
            ica_mode='known_mix')

    def test_gradient(self):
        '''
        Check gradients with method of finite differences
        '''
        grads = self.model.grads(self.vars)
        numerical_grads = finite_differences(self.vars, self.model.loss)
        self.assertTrue(np.allclose(grads, numerical_grads, rtol=0, atol=1e-6))

    def test_hessian(self):
        hessian = self.model.hessian(self.vars)
        numerical_hessian = np.zeros((len(self.vars), len(self.vars)))
        for i in range(len(self.vars)):
            deriv = lambda x : self.model.grads(x)[i]
            numerical_hessian[i] = finite_differences(self.vars, deriv)
        T = (len(self.vars) - 4)//2
        self.assertTrue(np.allclose(hessian,
                        numerical_hessian, rtol=0, atol=1e-5))

    def test_known_gradient(self):
        grads = self.known_model.grads(self.known_vars)
        numerical_grads = finite_differences(self.known_vars,
            self.known_model.loss)
        self.assertTrue(np.allclose(grads, numerical_grads, rtol=0, atol=1e-6))

    def test_known_hessian(self):
        hessian = self.known_model.hessian(self.known_vars)
        numerical_hessian = np.zeros((len(self.known_vars), len(self.known_vars)))
        for i in range(len(self.known_vars)):
            deriv = lambda x : self.known_model.grads(x)[i]
            numerical_hessian[i] = finite_differences(self.known_vars, deriv)
        T = (len(self.known_vars) - 4)//2
        self.assertTrue(np.allclose(hessian,
                        numerical_hessian, rtol=0, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
