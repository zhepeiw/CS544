import unittest
import numpy as np
import ica as ica

class TestICA(unittest.TestCase):
    def setUp(self):
        a = np.array([1, 2, 3, 4])
        t = np.linspace(1, 10, 100)
        s1 = np.sin(t)
        s2 = np.tanh(t)
        x1 = s1 + s2
        x2 = s1 - s2
        self.vars = np.concatenate([a, s1, s2])
        X = np.stack([x1, x2])
        self.model = ica.ICA(X)

    def test_gradient_check(self):
        '''
        Check gradients with method of finite differences
        '''
        delta = 1e-3
        grads = self.model.grads(self.vars)
        numerical_grads = np.zeros_like(self.vars)
        for i in range(len(self.vars)):
            self.vars[i] += delta
            plus = self.model.loss(self.vars)
            self.vars[i] -= 2*delta
            minus = self.model.loss(self.vars)
            self.vars[i] += delta
            numerical_grads[i] = (plus - minus) / (2 * delta)
        self.assertTrue(np.allclose(grads, numerical_grads, rtol=0, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
