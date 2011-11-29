import numpy as np
from numpy.random import *
from numpy import s_
import asp.codegen.templating.template as AspTemplate
import asp.jit.asp_module as asp_module
from codepy.cgen import *
from codepy.cuda import CudaModule
import pyublas
import sys
from imp import find_module
from os.path import join

class DeviceParameters(object):
    pass

class DeviceCUDA10(DeviceParameters):
    def __init__(self):
        self.params = {}
        # Feature support
        self.params['supports_32b_floating_point_atomics'] = 0
        # Technical specifications
        self.params['max_xy_grid_dim'] = 65535
        self.params['max_threads_per_block'] = 512
        self.params['max_shared_memory_capacity_per_SM'] = 16348
        # Device parameters
        self.params['max_gpu_memory_capacity'] = 1073741824

class DeviceCUDA20(DeviceParameters):
    def __init__(self):
        self.params = {}
        # Feature support
        self.params['supports_32b_floating_point_atomics'] = 1
        # Technical specifications
        self.params['max_xy_grid_dim'] = 65535
        self.params['max_threads_per_block'] = 1024
        self.params['max_shared_memory_capacity_per_SM'] = 16384*3
        # Device parameters
        self.params['max_gpu_memory_capacity'] = 1610612736

#TODO: Change to GMMComponents
class Components(object):
    
    def __init__(self, M, D, weights = None, means = None, covars = None):

        self.M = M
        self.D = D
        self.weights = weights if weights is not None else np.empty(M, dtype=np.float32)
        self.means = means if means is not None else  np.empty(M*D, dtype=np.float32)
        self.covars = covars if covars is not None else  np.empty(M*D*D, dtype=np.float32)

    def init_random_weights(self):
        self.weights = numpy.random.random((self.M))

    def init_random_means(self):
        self.means = numpy.random.random((self.M,self.D))

    def init_random_covars(self):
        self.covars = numpy.random.random((self.M, self.D, self.D))

    def shrink_components(self, new_M):
        self.weights = np.resize(self.weights, new_M) 
        self.means = np.resize(self.means, new_M*self.D) 
        self.covars = np.resize(self.covars, new_M*self.D*self.D)

class EvalData(object):

    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.memberships = np.zeros((M,N), dtype=np.float32)
        self.loglikelihoods = np.zeros(N, dtype=np.float32)
        self.likelihood = 0.0

    def resize(self, N, M):
        self.memberships.resize((M,N))
        self.memberships = np.ascontiguousarray(self.memberships)
        self.loglikelihoods.resize(N)
        self.loglikelihoods = np.ascontiguousarray(self.loglikelihoods)
        self.M = M
        self.N = N

class GMM(object):

    #Singleton ASP modules shared by all instances of GMM
    asp_mod = None    
    def get_asp_mod(self): return GMM.asp_mod or self.initialize_asp_mod()
    gpu_util_mod = None
    def get_gpu_util_mod(self): return GMM.gpu_util_mod or self.initialize_gpu_util_mod()
    device_id = None

    #Default parameter space for code variants
    variant_param_defaults = { 'base': {},
        'cuda_boost': {
            'num_blocks_estep': ['16'],
            'num_threads_estep': ['512'],
            'num_threads_mstep': ['256'],
            'num_event_blocks': ['128'],
            'max_num_dimensions': ['50'],
            'max_num_components': ['122'],
            'max_num_dimensions_covar_v3': ['41'],
            'max_num_components_covar_v3': ['81'],
            'diag_only': ['0'],
            'max_iters': ['10'],
            'min_iters': ['1'],
            'covar_version_name': ['V1', 'V2A', 'V2B', 'V3'] },
        'cilk_boost': {
            'diag_only': ['0'],
            'max_iters': ['10'],
            'min_iters': ['1'] }
    }

    
    def cuda_limits(param_dict, device):
            for k, v in device.params.iteritems():
                param_dict[k] = str(v)

            tpb = int(device.params['max_threads_per_block'])
            shmem = int(device.params['max_shared_memory_capacity_per_SM'])
            gpumem = int(device.params['max_gpu_memory_capacity'])
            vname = param_dict['covar_version_name']
            eblocks = int(param_dict['num_blocks_estep'])
            ethreads = int(param_dict['num_threads_estep'])
            mthreads = int(param_dict['num_threads_mstep'])
            blocking = int(param_dict['num_event_blocks'])
            max_d = int(param_dict['max_num_dimensions'])
            max_d_v3 = int(param_dict['max_num_dimensions_covar_v3'])
            max_m = int(param_dict['max_num_components'])
            max_m_v3 = int(param_dict['max_num_components_covar_v3'])
            max_n = gpumem / (max_d*4)
            max_arg_values = (max_m, max_d, max_n) #TODO: get device mem size

            compilable = False
            comp_func = lambda *args, **kwargs: False

            if ethreads <= tpb and mthreads <= tpb and (max_d*max_d+max_d)*4 < shmem and ethreads*4 < shmem and mthreads*4 < shmem: 
                if vname.upper() == 'V1':
                    if (max_d + mthreads)*4 < shmem:
                        compilable = True
                        comp_func = lambda *args, **kwargs: all([(a <= b) for a,b in zip(args, max_arg_values)])
                elif vname.upper() == 'V2A':
                    if max_d*4 < shmem:
                        compilable = True
                        comp_func = lambda *args, **kwargs: all([(a <= b) for a,b in zip(args, max_arg_values)]) and args[1]*(args[1]-1)/2 < tpb
                elif vname.upper() == 'V2B':
                    if (max_d*max_d+max_d)*4 < shmem:
                        compilable = True
                        comp_func = lambda *args, **kwargs: all([(a <= b) for a,b in zip(args, max_arg_values)]) and args[1]*(args[1]-1)/2 < tpb
                else:
                    if (max_d_v3*max_m_v3 + mthreads + max_m_v3)*4 < shmem:
                        compilable = True
                        comp_func = lambda *args, **kwargs: all([(a <= b) for a,b in zip(args, (max_m_v3, max_d_v3, max_n))])

            return compilable, comp_func


    backend_compilability_and_input_limits = { 
        'base': lambda param_dict, device: (True,lambda *args, **kwargs: True),
        'cilk_boost': lambda param_dict, device: (True,lambda *args, **kwargs: True),
        'cilk': lambda param_dict, device: (True,lambda *args, **kwargs: True),
        'cuda_boost': cuda_limits,
        'cuda': cuda_limits
    }

    def cuda_backend_render_func(param_dict, vals):
        cu_kern_tpl = AspTemplate.Template(filename="templates/em_cuda_kernels.mako")
        cu_kern_rend = cu_kern_tpl.render( param_val_list = vals, **param_dict)
        GMM.asp_mod.add_to_module([Line(cu_kern_rend)],'cuda')
        c_decl_tpl = AspTemplate.Template(filename="templates/em_cuda_launch_decl.mako") 
        c_decl_rend  = c_decl_tpl.render( param_val_list = vals, **param_dict)
        GMM.asp_mod.add_to_preamble(c_decl_rend,'cuda_boost')
        
    def cilk_backend_render_func(param_dict, vals):
        cilk_kern_tpl = AspTemplate.Template(filename="templates/em_cilk_kernels.mako")
        cilk_kern_rend = cilk_kern_tpl.render( param_val_list = vals, **param_dict)
        GMM.asp_mod.add_to_module([Line(cilk_kern_rend)],'cilk')
        c_decl_tpl = AspTemplate.Template(filename="templates/em_cilk_kernel_decl.mako") 
        c_decl_rend  = c_decl_tpl.render( param_val_list = vals, **param_dict)
        GMM.asp_mod.add_to_preamble(c_decl_rend,'cilk_boost')

    backend_specific_funcs = {
        'base': lambda param_dict, vals: None,
        'cuda_boost': cuda_backend_render_func,
        'cuda': cuda_backend_render_func,
        'cilk_boost': cilk_backend_render_func,
        'cilk': cilk_backend_render_func
    }

    #Flags to keep track of memory allocations, singletons
    event_data_gpu_copy = None
    event_data_cpu_copy = None
    component_data_gpu_copy = None
    component_data_cpu_copy = None
    eval_data_gpu_copy = None
    eval_data_cpu_copy = None

    # nternal functions to allocate and deallocate component and event data on the CPU and GPU
    def internal_alloc_event_data(self, X):
        if not np.array_equal(GMM.event_data_cpu_copy, X) and X is not None:
            if GMM.event_data_cpu_copy is not None:
                self.internal_free_event_data()
            self.get_asp_mod().alloc_events_on_CPU(X, X.shape[0], X.shape[1])
            GMM.event_data_cpu_copy = X
            if self.use_cuda:
                self.get_asp_mod().alloc_events_on_GPU(X.shape[0], X.shape[1])
                self.get_asp_mod().copy_event_data_CPU_to_GPU(X.shape[0], X.shape[1])
                GMM.event_data_gpu_copy = X

    def internal_free_event_data(self):
        if self.event_data_cpu_copy is not None:
            self.get_asp_mod().dealloc_events_on_CPU()
            GMM.event_data_cpu_copy = None
        if GMM.event_data_gpu_copy is not None:
            self.get_asp_mod().dealloc_events_on_GPU()
            GMM.event_data_gpu_copy = None

    def internal_alloc_component_data(self):
        if GMM.component_data_cpu_copy != self.components:
            if GMM.component_data_cpu_copy:
                self.internal_free_component_data()
            self.get_asp_mod().alloc_components_on_CPU(self.M, self.D, self.components.weights, self.components.means, self.components.covars)
            GMM.component_data_cpu_copy = self.components
            if self.use_cuda:
                self.get_asp_mod().alloc_components_on_GPU(self.M, self.D)
                self.get_asp_mod().copy_component_data_CPU_to_GPU(self.M, self.D)
                GMM.component_data_gpu_copy = self.components
            
    def internal_free_component_data(self):
        if GMM.component_data_cpu_copy is not None:
            self.get_asp_mod().dealloc_components_on_CPU()
            GMM.component_data_cpu_copy = None
        if GMM.component_data_gpu_copy is not None:
            self.get_asp_mod().dealloc_components_on_GPU()
            GMM.component_data_gpu_copy = None

    def internal_alloc_eval_data(self, X):
        if X is not None:
            if self.eval_data.M != self.M or self.eval_data.N != X.shape[0] or GMM.eval_data_cpu_copy != self.eval_data:
                if GMM.eval_data_cpu_copy is not None:
                    self.internal_free_eval_data()
                self.eval_data.resize(X.shape[0], self.M)
                self.get_asp_mod().alloc_evals_on_CPU(self.eval_data.memberships, self.eval_data.loglikelihoods)
                GMM.eval_data_cpu_copy = self.eval_data
                if self.use_cuda:
                    self.get_asp_mod().alloc_evals_on_GPU(X.shape[0], self.M)
                    GMM.eval_data_gpu_copy = self.eval_data
            
    def internal_free_eval_data(self):
        if GMM.eval_data_cpu_copy is not None:
            self.get_asp_mod().dealloc_evals_on_CPU()
            GMM.eval_data_cpu_copy = None
        if GMM.eval_data_gpu_copy is not None:
            self.get_asp_mod().dealloc_evals_on_GPU()
            GMM.eval_data_gpu_copy = None

    def __init__(self, M, D, means=None, covars=None, weights=None, names_of_backends_to_use=['cuda'], variant_param_spaces=None, device_id=0): #TODO: Make default backend 'base'
        self.M = M
        self.D = D
        self.variant_param_spaces = variant_param_spaces or GMM.variant_param_defaults
        self.components = Components(M, D, weights, means, covars)
        self.eval_data = EvalData(1, M)
        self.clf = None # pure python mirror module

        self.use_cuda = False
        self.use_cilk = False
        self.device = None
        if 'cuda' in names_of_backends_to_use:
            self.use_cuda = True
            if GMM.device_id == None:
                GMM.device_id = device_id
                self.get_gpu_util_mod().set_GPU_device(device_id)
            elif GMM.device_id != device_id:
                #TODO: May actually be allowable if deallocate all GPU allocations first?
                print "WARNING: As python only has one thread context, it can only use one GPU at a time, and you are attempting to run on a second GPU."
            self.capability = self.get_gpu_util_mod().get_GPU_device_capability_as_tuple(self.device_id)
            #TODO: Figure out some kind of class inheiritance to deal with the complexity of functionality and perf params
            self.device = DeviceCUDA10() if self.capability[0] < 2 else DeviceCUDA20()
        if 'cilk' in names_of_backends_to_use:
            self.use_cilk = True

    #Called the first time a GMM instance tries to use a GPU utility function
    def initialize_gpu_util_mod(self):
        GMM.gpu_util_mod = asp_module.ASPModule(use_cuda=True)
        GMM.gpu_util_mod.backends['cuda_boost'].codepy_toolchain.cc = 'gcc'
        GMM.gpu_util_mod.backends['cuda_boost'].codepy_toolchain.cflags.append('-fPIC')
        GMM.gpu_util_mod.backends['cuda'].codepy_toolchain.cc = 'gcc'
        GMM.gpu_util_mod.backends['cuda'].codepy_toolchain.cflags.append('-fPIC')
        GMM.gpu_util_mod.backends['cuda'].nvcc_toolchain.cflags.extend(["-Xcompiler","-fPIC"])
        GMM.gpu_util_mod.backends['cuda'].nvcc_toolchain.add_library("project",['.','./include'],[],[])  

        #TODO: Figure out what kind of file to put this in
        #TODO: Or, redo these using more robust functionality stolen from PyCuda
        util_funcs = [ ("""
            void set_GPU_device(int device) {
              int GPUCount;
              cudaGetDeviceCount(&GPUCount);
              if(GPUCount == 0) {
                device = 0;
              } else if (device >= GPUCount) {
                device  = GPUCount-1;
              }
              cudaSetDevice(device);
            }""", "set_GPU_device") ,
            ("""
            boost::python::tuple get_GPU_device_capability_as_tuple(int device) {
              int major, minor;
              cuDeviceComputeCapability(&major, &minor, device);
              return boost::python::make_tuple(major, minor);
            }
            """, "get_GPU_device_capability_as_tuple")]
        for fbody, fname in util_funcs:
            GMM.gpu_util_mod.add_function(fbody, fname, backend_name='cuda_boost')
        host_project_header_names = [ 'cuda_runtime.h'] 
        for x in host_project_header_names: GMM.gpu_util_mod.add_to_preamble([Include(x, False)], 'cuda_boost')
        GMM.gpu_util_mod.compile_module('cuda')
        #GMM.gpu_util_mod.compile_module('cuda_boost')
        return GMM.gpu_util_mod

    #Called the first time a GMM instance tries to use a specialized function
    def initialize_asp_mod(self):

        # Create ASP module
        GMM.asp_mod = asp_module.ASPModule(use_cuda=self.use_cuda, use_cilk=self.use_cilk)

        base_system_header_names = [ 'stdlib.h', 'stdio.h', 'string.h', 'math.h', 'time.h','pyublas/numpy.hpp']
        for header in base_system_header_names: 
            GMM.asp_mod.add_to_preamble([Include(header, True)], 'base')
        GMM.asp_mod.add_to_module([Line("""void boost_test(pyublas::numpy_vector<float> in, int a){ float* f = in.data().data(); a+=20; printf("TEST %d",a);}""")],'base')
        GMM.asp_mod.add_helper_function('boost_test', 'base')

        if self.use_cuda:
            self.insert_base_code_into_listed_modules(['cuda_boost'])
            self.insert_non_rendered_code_into_cuda_module()
            self.insert_rendered_code_into_module('cuda_boost')
            GMM.asp_mod.backends['cuda'].codepy_toolchain.cc = 'gcc'
            GMM.asp_mod.backends['cuda'].codepy_toolchain.cflags.append('-fPIC')
            GMM.asp_mod.backends['cuda'].nvcc_toolchain.cflags.extend(["-Xcompiler","-fPIC","-arch=sm_%s%s" % self.capability ])
            GMM.asp_mod.backends['cuda'].nvcc_toolchain.add_library("project",['.','./include'],[],[])  

        if self.use_cilk:
            self.insert_base_code_into_listed_modules(['cilk_boost'])
            self.insert_non_rendered_code_into_cilk_module()
            self.insert_rendered_code_into_module('cilk_boost')
            GMM.asp_mod.backends['cilk'].codepy_toolchain.cc = 'icc'
            GMM.asp_mod.backends['cilk'].codepy_toolchain.cflags = ['-O2','-gcc', '-ip','-fPIC']

        # Setup toolchain and compile

	from codepy.libraries import add_pyublas, add_numpy
        for mod in GMM.asp_mod.backends.itervalues():
            mod.codepy_toolchain.add_library("project",['.','./include'],[],[])
            add_pyublas(mod.codepy_toolchain)
            add_numpy(mod.codepy_toolchain)
        
        #print GMM.asp_mod.backends['cuda'].codepy_module.boost_module.generate()
        GMM.asp_mod.compile_all()
	GMM.asp_mod.restore_method_timings('train')
	GMM.asp_mod.restore_method_timings('eval')
        return GMM.asp_mod

    def insert_base_code_into_listed_modules(self, names_of_modules):
        c_base_tpl = AspTemplate.Template(filename="templates/em_base_helper_funcs.mako")
        c_base_rend = c_base_tpl.render()
        component_t_decl =""" 
            typedef struct components_struct {
                float* N;        // expected # of pixels in component: [M]
                float* pi;       // probability of component in GMM: [M]
                float* constant; // Normalizing constant [M]
                float* avgvar;    // average variance [M]
                float* means;   // Spectral mean for the component: [M*D]
                float* R;      // Covariance matrix: [M*D*D]
                float* Rinv;   // Inverse of covariance matrix: [M*D*D]
            } components_t;"""
        names_of_helper_funcs = ["alloc_events_on_CPU", "alloc_components_on_CPU", "alloc_evals_on_CPU", "dealloc_events_on_CPU", "dealloc_components_on_CPU", "dealloc_temp_components_on_CPU", "dealloc_evals_on_CPU", "get_temp_component_pi", "get_temp_component_means", "get_temp_component_covars", "relink_components_on_CPU", "compute_distance_rissanen", "merge_components" ]
        base_system_header_names = [ 'stdlib.h', 'stdio.h', 'string.h', 'math.h', 'time.h','pyublas/numpy.hpp']
        for mname in names_of_modules:
            for fname in names_of_helper_funcs:
                GMM.asp_mod.add_helper_function(fname, mname)
            for header in base_system_header_names: 
                GMM.asp_mod.add_to_preamble([Include(header, True)], mname)
            #Add Boost interface links for components and distance objects
            GMM.asp_mod.add_to_init("""boost::python::class_<components_struct>("Components");
                boost::python::scope().attr("components") = boost::python::object(boost::python::ptr(&components));""", mname)
            GMM.asp_mod.add_to_init("""boost::python::class_<return_component_container>("ReturnClusterContainer")
                .def(pyublas::by_value_rw_member( "new_component", &return_component_container::component))
                .def(pyublas::by_value_rw_member( "distance", &return_component_container::distance)) ;
                 boost::python::scope().attr("component_distance") = boost::python::object(boost::python::ptr(&ret));""", mname)
            GMM.asp_mod.add_to_module([Line(c_base_rend)],mname)
            GMM.asp_mod.add_to_preamble(component_t_decl, mname)

    def insert_non_rendered_code_into_cuda_module(self):
        #Add C/CUDA source code that is not based on code variant parameters

        #Add decls to preamble necessary for linking to compiled CUDA sources
        component_t_decl =""" 
            typedef struct components_struct {
                float* N;        // expected # of pixels in component: [M]
                float* pi;       // probability of component in GMM: [M]
                float* constant; // Normalizing constant [M]
                float* avgvar;    // average variance [M]
                float* means;   // Spectral mean for the component: [M*D]
                float* R;      // Covariance matrix: [M*D*D]
                float* Rinv;   // Inverse of covariance matrix: [M*D*D]
            } components_t;"""
        GMM.asp_mod.add_to_preamble(component_t_decl,'cuda')

        #Add necessary headers
        host_system_header_names = ['cuda_runtime.h']
        for x in host_system_header_names: 
            GMM.asp_mod.add_to_preamble([Include(x, True)],'cuda_boost')

        #Add bodies of helper functions
        c_base_tpl = AspTemplate.Template(filename="templates/em_cuda_host_helper_funcs.mako")
        c_base_rend  = c_base_tpl.render()
        GMM.asp_mod.add_to_module([Line(c_base_rend)],'cuda_boost')
        cu_base_tpl = AspTemplate.Template(filename="templates/em_cuda_device_helper_funcs.mako")
        cu_base_rend = cu_base_tpl.render()
        GMM.asp_mod.add_to_module([Line(cu_base_rend)],'cuda')
        #Add Boost interface links for helper functions
        names_of_helper_funcs = ["alloc_events_on_GPU","alloc_components_on_GPU","alloc_evals_on_GPU","copy_event_data_CPU_to_GPU", "copy_component_data_CPU_to_GPU", "copy_component_data_GPU_to_CPU", "copy_evals_data_GPU_to_CPU","dealloc_events_on_GPU","dealloc_components_on_GPU", "dealloc_evals_on_GPU"]
        for fname in names_of_helper_funcs:
            GMM.asp_mod.add_helper_function(fname,'cuda_boost')

    def insert_non_rendered_code_into_cilk_module(self):
        component_t_decl =""" 
            typedef struct components_struct {
                float* N;        // expected # of pixels in component: [M]
                float* pi;       // probability of component in GMM: [M]
                float* constant; // Normalizing constant [M]
                float* avgvar;    // average variance [M]
                float* means;   // Spectral mean for the component: [M*D]
                float* R;      // Covariance matrix: [M*D*D]
                float* Rinv;   // Inverse of covariance matrix: [M*D*D]
            } components_t;"""
        GMM.asp_mod.add_to_preamble(component_t_decl,'cilk')
        cilk_base_tpl = AspTemplate.Template(filename="templates/em_cilk_helper_funcs.mako")
        cilk_base_rend = cilk_base_tpl.render()
        GMM.asp_mod.add_to_module([Line(cilk_base_rend)],'cilk')

        #Add Cilk source code that is not based on code variant parameters
        base_system_header_names = [ 'stdlib.h', 'stdio.h', 'string.h', 'math.h', 'time.h']
        system_header_names = ['cilk/cilk.h','cilk/reducer_opadd.h']  
        for x in base_system_header_names+system_header_names: 
            GMM.asp_mod.add_to_preamble([Include(x, True)],'cilk')

    def render_and_add_to_module( self, param_dict, limit_generator, render_backend_specific_funcs, backend_name):
        # Evaluate whether these particular parameters can be compiled on this particular device
        can_be_compiled, comparison_function_for_input_args = limit_generator(param_dict, self.device)

        # Get vals based on alphabetical order of keys
        param_names = param_dict.keys()
        param_names.sort()
        vals = map(param_dict.get, param_names)
        # Use vals to render templates 
        backend_filename = backend_name[:-6] if backend_name.endswith('_boost') else backend_name
        c_train_tpl = AspTemplate.Template(filename="templates/em_"+backend_filename+"_train.mako")
        c_eval_tpl = AspTemplate.Template(filename="templates/em_"+backend_filename+"_eval.mako")
        c_train_rend  = c_train_tpl.render( param_val_list = vals, **param_dict)
        c_eval_rend  = c_eval_tpl.render( param_val_list = vals, **param_dict)

        def var_name_generator(base):
            return '_'.join(['em',backend_filename,base]+vals)
        def dummy_func_body_gen(base):
            return "void "+var_name_generator(base)+"(int m, int d, int n, pyublas::numpy_vector<float> data){}"

        key_func = lambda name, *args, **kwargs: (name, args[0], args[1], args[2])
        if can_be_compiled: 
            render_backend_specific_funcs(param_dict, vals)
            train_body = c_train_rend
            eval_body = c_eval_rend
        else:
            train_body = dummy_func_body_gen('train')
            eval_body = dummy_func_body_gen('eval')

        GMM.asp_mod.add_function_with_variants( [train_body],
                                                'train', 
                                                [var_name_generator('train')],
                                                key_func,
                                                lambda results, time: float(time)/float(results[1]),
                                                [comparison_function_for_input_args],
                                                [can_be_compiled],
                                                param_names,
                                                backend_name
                                              )
        GMM.asp_mod.add_function_with_variants( [eval_body], 
                                                'eval', 
                                                [var_name_generator('eval')],
                                                key_func,
                                                lambda results, time: time,
                                                [comparison_function_for_input_args],
                                                [can_be_compiled],
                                                param_names,
                                                backend_name
                                              )


    def generate_permutations (self, key_arr, val_arr_arr, current, limit_func, backend_specific_func, backend_name):
        idx = len(current)
        name = key_arr[idx]
        for v in val_arr_arr[idx]:
            current[name]  = v
            if idx == len(key_arr)-1:
                self.render_and_add_to_module(current.copy(), limit_func, backend_specific_func, backend_name)
            else:
                self.generate_permutations(key_arr, val_arr_arr, current, limit_func, backend_specific_func, backend_name)
        del current[name]

    def insert_rendered_code_into_module(self, backend_name):
        self.generate_permutations( self.variant_param_spaces[backend_name].keys(), self.variant_param_spaces[backend_name].values(), {}, GMM.backend_compilability_and_input_limits[backend_name], GMM.backend_specific_funcs[backend_name], backend_name)

    def __del__(self):
        self.internal_free_event_data()
        self.internal_free_component_data()
        self.internal_free_eval_data()
	GMM.asp_mod.save_method_timings('train')
	GMM.asp_mod.save_method_timings('eval')
    
    def train_using_python(self, input_data):
        from scikits.learn import mixture
        self.clf = mixture.GMM(n_states=self.M, cvtype='full')
        self.clf.fit(input_data)
        return self.clf.means, self.clf.covars
    
    def eval_using_python(self, obs_data):
        from scikits.learn import mixture
        if self.clf is not None:
            return self.clf.eval(obs_data)
        else: return []

    def predict_using_python(self, obs_data):
        from scikits.learn import mixture
        if self.clf is not None:
            return self.clf.predict(obs_data)
        else: return []

    def train(self, input_data):
        N = input_data.shape[0] 
        if input_data.shape[1] != self.D:
            print "Error: Data has %d features, model expects %d features." % (input_data.shape[1], self.D)
        self.internal_alloc_event_data(input_data)
        self.internal_alloc_eval_data(input_data)
        self.internal_alloc_component_data()
        self.eval_data.likelihood = self.get_asp_mod().train(self.M, self.D, N)[0]
        return self

    def eval(self, obs_data):
        N = obs_data.shape[0]
        if obs_data.shape[1] != self.D:
            print "Error: Data has %d features, model expects %d features." % (obs_data.shape[1], self.D)
        self.internal_alloc_event_data(obs_data)
        self.internal_alloc_eval_data(obs_data)
        self.internal_alloc_component_data()
        self.eval_data.likelihood = self.get_asp_mod().eval(self.M, self.D, N)
        logprob = self.eval_data.loglikelihoods
        posteriors = self.eval_data.memberships
        return logprob, posteriors # N log probabilities, NxM posterior probabilities for each component

    def score(self, obs_data):
        logprob, posteriors = self.eval(obs_data)
        return logprob # N log probabilities

    def decode(self, obs_data):
        logprob, posteriors = self.eval(obs_data)
        return logprob, posteriors.argmax(axis=0) # N log probabilities, N indexes of most likely components 

    def predict(self, obs_data):
        logprob, posteriors = self.eval(obs_data)
        return posteriors.argmax(axis=0) # N indexes of most likely components

    def merge_components(self, c1, c2, new_component):
        self.get_asp_mod().dealloc_temp_components_on_CPU()
        self.get_asp_mod().merge_components(c1, c2, new_component, self.M, self.D)
        self.M -= 1
        self.components.shrink_components(self.M)
        self.get_asp_mod().relink_components_on_CPU(self.components.weights, self.components.means, self.components.covars)
        return 

    def compute_distance_rissanen(self, c1, c2):
        self.get_asp_mod().compute_distance_rissanen(c1, c2, self.D)
        new_component = self.get_asp_mod().compiled_module.component_distance.new_component
        dist = self.get_asp_mod().compiled_module.component_distance.distance
        return new_component, dist

    def get_new_component_means(self, new_component):
        return self.get_asp_mod().get_temp_component_means(new_component, self.D).reshape((1, self.D))

    def get_new_component_covars(self, new_component):
        return self.get_asp_mod().get_temp_component_covars(new_component, self.D).reshape((1, self.D, self.D))
    
def compute_distance_BIC(gmm1, gmm2, data):
    cd1_M = gmm1.M
    cd2_M = gmm2.M
    nComps = cd1_M + cd2_M

    ratio1 = float(cd1_M)/float(nComps)
    ratio2 = float(cd2_M)/float(nComps)

    w = np.append(ratio1*gmm1.components.weights, ratio2*gmm2.components.weights)
    m = np.append(gmm1.components.means, gmm2.components.means)
    c = np.append(gmm1.components.covars, gmm2.components.covars)
    temp_GMM = GMM(nComps, gmm1.D, weights=w, means=m, covars=c)

    temp_GMM.train(data)

    score = temp_GMM.eval_data.likelihood - (gmm1.eval_data.likelihood + gmm2.eval_data.likelihood)
    #print temp_GMM.eval_data.likelihood, gmm1.eval_data.likelihood, gmm2.eval_data.likelihood, score
    return temp_GMM, score

