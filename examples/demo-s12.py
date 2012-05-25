import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as pl
import itertools
import copy

from gmm_specializer.gmm import *

fig = plt.figure()
ax = fig.add_subplot(111)

def generate_synthetic_data_old(N):
    np.random.seed(0)
    C = np.array([[0., -0.7], [3.5, .7]])
    Y = np.r_[np.dot(np.random.randn(N, 2), C),
              np.random.randn(N, 2) + np.array([20, 20])]
    return Y.astype(np.float32)

def generate_synthetic_data(N):
    np.random.seed(0)
    C = np.array([[0., -0.7], [3.5, .7]])
    C1 = np.array([[-0.4, 1.7], [0.3, .7]])
    Y = np.r_[
        np.dot(np.random.randn(N/3, 2), C1) - np.array([1,-5]),
        np.dot(np.random.randn(N/3, 2), C),
        np.random.randn(N/3, 2) + np.array([3, 5]),
        ]
    return Y.astype(np.float32)

class EventHandler:
    def __init__(self):
        fig.canvas.mpl_connect('button_press_event', self.mousepress)
        fig.canvas.mpl_connect('key_press_event', self.keypress)
        self.size = 8
        self.cvtype = 'full'
        self.D = 2
        self.N = 600
        self.gmm = None
        self.X = generate_synthetic_data(self.N)
        pl.scatter(self.X.T[0], self.X.T[1], self.size, color="black")

    def mousepress(self, event):
        if event.inaxes!=ax:
            return

        new_point = np.array([[event.xdata, event.ydata]])
        self.X = np.ascontiguousarray(np.r_[self.X, new_point],dtype=np.float32)
        self.N += 1
        pl.scatter(new_point.T[0], new_point.T[1], self.size, color="black", marker='x')
        fig.canvas.draw()

    def keypress(self, event):
        if event.key in ['1','2','3','4','5','6','7','8','9']:
            self.M = int(event.key)
            self.gmm = GMM(self.M, self.D, cvtype=self.cvtype)
            likelihood = self.gmm.train(self.X)
            means = self.gmm.components.means.reshape((self.M, self.D))
            covars = self.gmm.components.covars.reshape((self.M, self.D, self.D))
            Y = self.gmm.predict(self.X)
            self.plot(means, covars, Y)

        if event.key == 'e' and self.gmm:
            means = self.gmm.components.means
            covars = self.gmm.components.covars
            Y = self.gmm.predict(self.X)
            self.plot(means, covars, Y)
            
    def plot(self, means, covars, Y_):
        ax.clear()
        color_iter = itertools.cycle (['r', 'g', 'b', 'c'])
        for i, (mean, covar, color) in enumerate(zip(means, covars, color_iter)):
            v, w = np.linalg.eigh(covar)
            u = w[0] / np.linalg.norm(w[0])
            pl.scatter(self.X.T[0,Y_==i], self.X.T[1,Y_==i], self.size, color=color)
            angle = np.arctan(u[1]/u[0])
            angle = 180 * angle / np.pi
            ell = mpl.patches.Ellipse (mean, v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.3)
            ax.add_artist(ell)
        fig.canvas.draw()


if __name__ == '__main__':
    handler = EventHandler()
    plt.show()
