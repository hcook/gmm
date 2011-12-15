import unittest2 as unittest
from itertools import chain
import sys
import math
import copy

from gmm import *


class BasicTests(unittest.TestCase):
    def test_init(self):
        gmm = GMM(3, 2, cvtype='diag')
        self.assertIsNotNone(gmm)

    def test_pure_python(self):
        self.assertTrue(True)

class SyntheticDataTests(unittest.TestCase):
    def setUp(self):
        self.D = 2
        self.N = 600
        self.M = 3
        np.random.seed(0)
        C = np.array([[0., -0.7], [3.5, .7]])
        C1 = np.array([[-0.4, 1.7], [0.3, .7]])
        Y = np.r_[
            np.dot(np.random.randn(self.N/3, 2), C1),
            np.dot(np.random.randn(self.N/3, 2), C),
            np.random.randn(self.N/3, 2) + np.array([3, 3]),
            ]
        self.X = Y.astype(np.float32)
    
    def test_pure_python(self):
        gmm = GMM(self.M, self.D, cvtype='diag')
        means, covars = gmm.train_using_python(self.X)
        Y = gmm.predict_using_python(self.X)
        self.assertTrue(len(set(Y)) > 1)

    def test_training_once(self):
        gmm0 = GMM(self.M, self.D, cvtype='diag')
        likelihood0 = gmm0.train(self.X)
        means0  = gmm0.components.means.flatten()
        covars0 = gmm0.components.covars.flatten()

        gmm1 = GMM(self.M, self.D, cvtype='diag')
        likelihood1 = gmm1.train(self.X)
        means1  = gmm1.components.means.flatten()
        covars1 = gmm1.components.covars.flatten()

        self.assertAlmostEqual(likelihood0, likelihood1)
        for a,b in zip(means0, means1):   self.assertAlmostEqual(a,b)
        for a,b in zip(covars0, covars1): self.assertAlmostEqual(a,b)

    def test_prediction_once(self):
        gmm0 = GMM(self.M, self.D, cvtype='diag')
        likelihood0 = gmm0.train(self.X)
        Y0 = gmm0.predict(self.X)

        gmm1 = GMM(self.M, self.D, cvtype='diag')
        likelihood1 = gmm1.train(self.X)
        Y1 = gmm1.predict(self.X)

        for a,b in zip(Y0, Y1): self.assertAlmostEqual(a,b)

class SpeechDataTests(unittest.TestCase):
    def setUp(self):
        self.X = np.ndfromtxt('IS1000a.csv', delimiter=',', dtype=np.float32)
        self.N = self.X.shape[0]
        self.D = self.X.shape[1]


if __name__ == '__main__':
    unittest.main()
