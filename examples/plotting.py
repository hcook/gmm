import matplotlib as mpl
#mpl.use('PDF')  # must be called immediately, and before import pylab
                 # sets the back-end for matplotlib
import pylab as pl
import itertools
import copy

from gmm import *

def generate_synthetic_data(N):
    np.random.seed(0)
    C = np.array([[0., -0.7], [3.5, .7]])
    C1 = np.array([[-0.4, 1.7], [0.3, .7]])
    Y = np.r_[
        np.dot(np.random.randn(N/3, 2), C1) - np.array([-1,-5]),
        np.dot(np.random.randn(N/3, 2), C),
        np.random.randn(N/3, 2) + np.array([3, 3]),
        ]
    return Y.astype(np.float32)

class Plotter(object):

    def __init__(self, num_rows, num_cols):
        self.results = {}
        self.plot_base = num_rows*100+num_cols*10
        self.D = 2
        self.N = 600
        self.X = generate_synthetic_data(self.N)

    def pure(self, M, cvtype, plot_id):
        gmm = GMM(M, self.D, cvtype=cvtype)
        means, covars = gmm.train_using_python(self.X)
        Y = gmm.predict_using_python(self.X)
        self.results['Pure Python '+cvtype] = (str(self.plot_base+plot_id), means, covars, Y.T)

    def special(self, M, cvtype, plot_id):
        gmm = GMM(M, self.D, cvtype=cvtype)
        likelihood = gmm.train(self.X)
        means = gmm.components.means.reshape((M, self.D))
        covars = gmm.components.covars.reshape((M, self.D, self.D))
        Y = gmm.predict(self.X)
        self.results[' '.join(['ASP',cvtype,str(self.D),str(M),str(self.N)])] = (str(self.plot_base+plot_id), copy.deepcopy(means), copy.deepcopy(covars), copy.deepcopy(Y))
        return likelihood
        
    def plot(self):
        for t, r in self.results.iteritems():
            splot = pl.subplot(r[0], title=t)
            color_iter = itertools.cycle (['r', 'g', 'b', 'c'])
            Y_ = r[3]
            for i, (mean, covar, color) in enumerate(zip(r[1], r[2], color_iter)):
                v, w = np.linalg.eigh(covar)
                u = w[0] / np.linalg.norm(w[0])
                pl.scatter(self.X.T[0,Y_==i], self.X.T[1,Y_==i], .8, color=color)
                angle = np.arctan(u[1]/u[0])
                angle = 180 * angle / np.pi
                ell = mpl.patches.Ellipse (mean, v[0], v[1], 180 + angle, color=color)
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(0.5)
                splot.add_artist(ell)
        pl.show()
        pl.savefig('gmm_test')
        
if __name__ == '__main__':
    num_rows = 3
    num_cols = 2
    p = Plotter(num_rows,num_cols)
    p.pure(3, 'diag', 1)
    p.pure(3, 'full', 2)
    p.special(3, 'diag', 3)
    p.special(3, 'full', 4)
    p.special(6, 'diag', 5)
    p.special(6, 'full', 6)
    p.plot()
 
