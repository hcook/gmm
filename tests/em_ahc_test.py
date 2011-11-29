import unittest
import pylab as pl
import matplotlib as mpl
import itertools
import sys
import math
import timeit
import copy

from em import *

def generate_synthetic_data(N):
    np.random.seed(0)
    C = np.array([[0., -0.7], [3.5, .7]])
    C1 = np.array([[-0.4, 1.7], [0.3, .7]])
    Y = np.r_[
        np.dot(np.random.randn(N/3, 2), C1),
        np.dot(np.random.randn(N/3, 2), C),
        np.random.randn(N/3, 2) + np.array([3, 3]),
        ]
    return Y.astype(np.float32)

class EMTester(object):

    def __init__(self, from_file, variant_param_space, device_id, num_subps):
        self.results = {}
        self.variant_param_space = variant_param_space
        self.device_id = device_id
        self.num_subplots = num_subps
        self.plot_id = num_subps/2*100 + 21
        if from_file:
            self.X = np.ndfromtxt('IS1000a.csv', delimiter=',', dtype=np.float32)
            self.N = self.X.shape[0]
            self.D = self.X.shape[1]
        else:
            N = 1000
            self.X = generate_synthetic_data(N)
            self.N = self.X.shape[0]
            self.D = self.X.shape[1]

    def new_gmm(self, M):
        self.M = M
        self.gmm = GMM(self.M, self.D, self.variant_param_space, self.device_id)

    def new_gmm_list(self, M, k):
        self.M = M
        self.init_num_clusters = k
        self.gmm_list = [GMM(self.M, self.D, self.variant_param_space, self.device_id) for i in range(k)]

    def test_speech_ahc(self):
        
        # Get the events, divide them into an initial k clusters and train each GMM on a cluster
        per_cluster = self.N/self.init_num_clusters
        init_training = zip(self.gmm_list,np.vsplit(self.X, range(per_cluster, self.N, per_cluster)))
        for g, x in init_training:
            g.train(x)

        # Perform hierarchical agglomeration based on BIC scores
        best_BIC_score = 1.0
        while (best_BIC_score > 0 and len(self.gmm_list) > 1):
            print "Num GMMs: %d, last score: %d" % (len(self.gmm_list), best_BIC_score)

            num_clusters = len(self.gmm_list)
            # Resegment data based on likelihood scoring
            likelihoods = self.gmm_list[0].score(self.X)
            for g in self.gmm_list[1:]:
                likelihoods = np.column_stack((likelihoods, g.score(self.X)))
            most_likely = likelihoods.argmax(axis=1)
            # Across 2.5 secs of observations, vote on which cluster they should be associated with
            iter_training = {}
            for i in range(250, self.N, 250):
                votes = np.zeros(num_clusters)
                for j in range(i-250, i):
                    votes[most_likely[j]] += 1
                #print votes.argmax()
                iter_training.setdefault(self.gmm_list[votes.argmax()],[]).append(self.X[i-250:i,:])
            votes = np.zeros(num_clusters)
            for j in range((self.N/250)*250, self.N):
                votes[most_likely[j]] += 1
            #print votes.argmax()
            iter_training.setdefault(self.gmm_list[votes.argmax()],[]).append(self.X[(self.N/250)*250:self.N,:])

            # Retrain the GMMs on the clusters for which they were voted most likely and
            # make a list of candidates for merging
            iter_bic_list = []
            for g, data_list in iter_training.iteritems():
                cluster_data =  data_list[0]
                for d in data_list[1:]:
                    cluster_data = np.concatenate((cluster_data, d))
                cluster_data = np.ascontiguousarray(cluster_data)
                g.train(cluster_data)
                iter_bic_list.append((g,cluster_data))
    
            # Keep any GMMs that lost all votes in candidate list for merging
            for g in self.gmm_list:
                if g not in iter_training.keys():
                    iter_bic_list.append((g,None))            

            # Score all pairs of GMMs using BIC
            best_merged_gmm = None
            best_BIC_score = 0.0
            merged_tuple = None
            for gmm1idx in range(len(iter_bic_list)):
                for gmm2idx in range(gmm1idx+1, len(iter_bic_list)):
                    g1, d1 = iter_bic_list[gmm1idx]
                    g2, d2 = iter_bic_list[gmm2idx] 
                    score = 0.0
                    if d1 is not None or d2 is not None:
                        if d1 is not None and d2 is not None:
                            new_gmm, score = compute_distance_BIC(g1, g2, np.concatenate((d1, d2)))
                        elif d1 is not None:
                            new_gmm, score = compute_distance_BIC(g1, g2, d1)
                        else:
                            new_gmm, score = compute_distance_BIC(g1, g2, d2)
                    print "Comparing BIC %d with %d: %f" % (gmm1idx, gmm2idx, score)
                    if score > best_BIC_score: 
                        best_merged_gmm = new_gmm
                        merged_tuple = (g1, g2)
                        best_BIC_score = score
            
            # Merge the winning candidate pair if its deriable to do so
            if best_BIC_score > 0.0:
                self.gmm_list.remove(merged_tuple[0]) 
                self.gmm_list.remove(merged_tuple[1]) 
                self.gmm_list.append(best_merged_gmm)

        print "Final size of each cluster:", [ g.M for g in self.gmm_list]

    def test_cytosis_ahc(self):
        M_start = self.M
        M_end = 0
        plot_counter = 2
        
        for M in reversed(range(M_end, M_start)):

            print "======================== AHC loop: M = ", M+1, " ==========================="
            self.gmm.train(self.X)
    
            #plotting
            means = self.gmm.components.means.reshape((self.gmm.M, self.gmm.D))
            covars = self.gmm.components.covars.reshape((self.gmm.M, self.gmm.D, self.gmm.D))
            Y = self.gmm.predict(self.X)
            if(self.plot_id % 10 <= self.num_subplots):
                self.results['_'.join(['ASP v',str(self.plot_id-(100*self.num_subplots+11)),'@',str(self.gmm.D),str(self.gmm.M),str(self.N)])] = (str(self.plot_id), copy.deepcopy(means), copy.deepcopy(covars), copy.deepcopy(Y))
                self.plot_id += 1

	    #find closest components and merge
	    if M > 0: #don't merge if there is only one component
	        gmm_list = []
	        for c1 in range(0, self.gmm.M):
		    for c2 in range(c1+1, self.gmm.M):
		        new_component, dist = self.gmm.compute_distance_rissanen(c1, c2)
		        gmm_list.append((dist, (c1, c2, new_component)))
		        #print "gmm_list after append: ", gmm_list
		    
	        #compute minimum distance
	        min_c1, min_c2, min_component = min(gmm_list, key=lambda gmm: gmm[0])[1]
	        self.gmm.merge_components(min_c1, min_c2, min_component)

    def time_cytosis_ahc(self):
        M_start = self.M
        M_end = 0

        for M in reversed(range(M_end, M_start)):

            print "======================== AHC loop: M = ", M+1, " ==========================="
            self.gmm.train(self.X)

            #find closest components and merge
            if M > 0: #don't merge if there is only one component
                gmm_list = []
                for c1 in range(0, self.gmm.M):
                    for c2 in range(c1+1, self.gmm.M):
                        new_component, dist = self.gmm.compute_distance_rissanen(c1, c2)
                        gmm_list.append((dist, (c1, c2, new_component)))
                        
                #compute minimum distance
                min_c1, min_c2, min_component = min(gmm_list, key=lambda gmm: gmm[0])[1]
                self.gmm.merge_components(min_c1, min_c2, min_component)

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
        
if __name__ == '__main__':
    device_id = 0
    num_subplots = 6
    variant_param_space = {
            'num_blocks_estep': ['16'],
            'num_threads_estep': ['512'],
            'num_threads_mstep': ['256'],
            'num_event_blocks': ['128'],
            'max_num_dimensions': ['50'],
            'max_num_components': ['128'],
            'max_num_dimensions_covar_v3': ['40'],
            'max_num_components_covar_v3': ['82'],
            'diag_only': ['0'],
            'max_iters': ['10'],
            'min_iters': ['1'],
            'covar_version_name': ['V2A']
    }
    emt = EMTester(True, variant_param_space, device_id, num_subplots)
    #emt.new_gmm(6)
    #t = timeit.Timer(emt.time_cytosis_ahc)
    #print t.timeit(number=1)
    #emt.test_cytosis_ahc()
    #emt.plot()
    emt.new_gmm_list(5, 16)
    emt.test_speech_ahc() 

