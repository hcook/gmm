import pylab as pl
import matplotlib as mpl
import itertools
import sys
import pickle
import timeit
from gmm import *

class EMAutotuner(object):

    def __init__(self, variant_param_space, input_param_space, ifile_name, func_name, device_id, max_iters_per_point):
        fromfile = np.recfromcsv('IS1000a.csv', names=None, dtype=np.float32)
        self.orig_X = fromfile
        self.ifile_name = ifile_name
        self.func_name = func_name
        self.device_id = device_id
        self.input_list = [] #tuple (M, X)
        self.shaped_Xs = {}
        self.variant_param_space = variant_param_space
        self.variant_param_space_size = max_iters_per_point or reduce(lambda x, y: x*y, [len(v) for v in self.variant_param_space.values()])
        self.input_param_space = input_param_space
        self.generate_shaped_inputs(input_param_space.keys(), input_param_space.values(), {})
        self.gmm = GMM(1, 1, self.variant_param_space, self.device_id)
        mod = self.gmm.get_asp_mod()

    def add_to_input_list(self, param_dict):
        D = int(param_dict['D'])
        N = int(param_dict['N'])
        M = int(param_dict['M'])
        new_X = self.shaped_Xs.setdefault((D,N), np.resize(self.orig_X,(N,D)))
        self.input_list.append((M, D, N, new_X))

    def generate_shaped_inputs(self, key_arr, val_arr_arr, current):
        idx = len(current)
        name = key_arr[idx]
        for v in val_arr_arr[idx]:
            current[name]  = v
            if idx == len(key_arr)-1:
                self.add_to_input_list(current)
            else:
                self.generate_shaped_inputs(key_arr, val_arr_arr, current)
        del current[name]

    def test_point(self, input_tuple):
        self.gmm = GMM(input_tuple[0], input_tuple[1], device_id = self.device_id)
        likelihood = getattr(self.gmm, self.func_name)(input_tuple[3])

    def search_space(self):
        if self.ifile_name:
            self.gmm.asp_mod.restore_func_variant_timings(self.func_name, self.ifile_name)
        for i in self.input_list:
            print "M=%d, D=%d, N=%d" % i[:3]
            for j in range(0, self.variant_param_space_size):
                self.test_point(i)
            self.gmm.asp_mod.save_func_variant_timings(self.func_name)    

if __name__ == '__main__':
    ifile_name = None #".aspcache/train.vardump" #"data/285.210x16.train.vardump"
    func_name = "train"
    device_id = 1
    max_iters_per_point = None #1
    variant_param_space = {
            'num_blocks_estep': ['16'],
            'num_threads_estep': ['128','256','512'],
            'num_threads_mstep': ['128','256','512'],
            'num_event_blocks': ['32','64','128'],
            'max_num_dimensions': ['50'],
            'max_num_components': ['122'],
            'max_num_dimensions_covar_v3': ['43'],
            'max_num_components_covar_v3': ['72'],
            'covar_version_name': ['V1', 'V2A', 'V2B', 'V3']
    }
    input_param_space =  {
            'D': np.arange(2, 45, 4),
            'N': np.arange(10000, 90001, 20000),
            'M': np.arange(1, 102, 10)
    }
    emt = EMAutotuner(variant_param_space, input_param_space, ifile_name, func_name, device_id, max_iters_per_point)
    emt.search_space()
 
