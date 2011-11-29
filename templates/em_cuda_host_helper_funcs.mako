
#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
   }

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);            \


//=== Data structure pointers ===

//GPU copies of events
float* d_fcs_data_by_event;
float* d_fcs_data_by_dimension;

//GPU copies of components
components_t temp_components;
components_t* d_components;

//GPU copies of eval data
float *d_component_memberships;
float *d_loglikelihoods;

//Copy functions to ensure CPU data structures are up to date
void copy_component_data_GPU_to_CPU(int num_components, int num_dimensions);
void copy_evals_data_GPU_to_CPU(int num_events, int num_components);


//=== Memory Alloc/Free Functions ===

// ================== Event data allocation on GPU  ================= :

void alloc_events_on_GPU(int num_dimensions, int num_events) {
  //printf("Alloc events on GPU\n");
  int mem_size = num_dimensions*num_events*sizeof(float);
    
  // allocate device memory for FCS data
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_event, mem_size));
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_dimension, mem_size));

  return;
}

// ================== Cluster data allocation on GPU  ================= :
void alloc_components_on_GPU(int original_num_components, int num_dimensions) {

  //printf("Alloc components on GPU\n");

  // Setup the component data structures on device
  // First allocate structures on the host, CUDA malloc the arrays
  // Then CUDA malloc structures on the device and copy them over
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.N),sizeof(float)*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.pi),sizeof(float)*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.constant),sizeof(float)*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.avgvar),sizeof(float)*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.means),sizeof(float)*num_dimensions*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.R),sizeof(float)*num_dimensions*num_dimensions*original_num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_components.Rinv),sizeof(float)*num_dimensions*num_dimensions*original_num_components));
   
  // Allocate a struct on the device 
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_components, sizeof(components_t)));
    
  // Copy Cluster data to device
  CUDA_SAFE_CALL(cudaMemcpy(d_components,&temp_components,sizeof(components_t),cudaMemcpyHostToDevice));

  return;
}

// ================= Eval data alloc on GPU =============== 

void alloc_evals_on_GPU(int num_events, int num_components){
  CUDA_SAFE_CALL(cudaMalloc((void**) &(d_component_memberships),sizeof(float)*num_events*num_components));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(d_loglikelihoods),sizeof(float)*num_events));
}

// ======================== Copy event data from CPU to GPU ================
void copy_event_data_CPU_to_GPU(int num_events, int num_dimensions) {

  //printf("Copy events to GPU\n");
  int mem_size = num_dimensions*num_events*sizeof(float);
  // copy FCS to device
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_event, fcs_data_by_event, mem_size,cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_dimension, fcs_data_by_dimension, mem_size,cudaMemcpyHostToDevice) );
  return;
}

// ======================== Copy component data from CPU to GPU ================
void copy_component_data_CPU_to_GPU(int num_components, int num_dimensions) {

   CUDA_SAFE_CALL(cudaMemcpy(temp_components.N, components.N, sizeof(float)*num_components,cudaMemcpyHostToDevice)); 
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.pi, components.pi, sizeof(float)*num_components,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.constant, components.constant, sizeof(float)*num_components,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.avgvar, components.avgvar, sizeof(float)*num_components,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.means, components.means, sizeof(float)*num_dimensions*num_components,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.R, components.R, sizeof(float)*num_dimensions*num_dimensions*num_components,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_components.Rinv, components.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_components,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(d_components,&temp_components,sizeof(components_t),cudaMemcpyHostToDevice));
   return;
}
// ======================== Copy component data from GPU to CPU ================
void copy_component_data_GPU_to_CPU(int num_components, int num_dimensions) {

  CUDA_SAFE_CALL(cudaMemcpy(&temp_components, d_components, sizeof(components_t),cudaMemcpyDeviceToHost));
  // copy all of the arrays from the structs
  CUDA_SAFE_CALL(cudaMemcpy(components.N, temp_components.N, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.pi, temp_components.pi, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.constant, temp_components.constant, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.avgvar, temp_components.avgvar, sizeof(float)*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.means, temp_components.means, sizeof(float)*num_dimensions*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.R, temp_components.R, sizeof(float)*num_dimensions*num_dimensions*num_components,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(components.Rinv, temp_components.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_components,cudaMemcpyDeviceToHost));
}

// ======================== Copy eval data from GPU to CPU ================
void copy_evals_data_GPU_to_CPU(int num_events, int num_components){
  CUDA_SAFE_CALL(cudaMemcpy(component_memberships, d_component_memberships, sizeof(float)*num_events*num_components, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(loglikelihoods, d_loglikelihoods, sizeof(float)*num_events, cudaMemcpyDeviceToHost));
}


// ================== Event data dellocation on GPU  ================= :
void dealloc_events_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_event));
  CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_dimension));
}


// ==================== Cluster data deallocation on GPU =================  
void dealloc_components_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(temp_components.N));
  CUDA_SAFE_CALL(cudaFree(temp_components.pi));
  CUDA_SAFE_CALL(cudaFree(temp_components.constant));
  CUDA_SAFE_CALL(cudaFree(temp_components.avgvar));
  CUDA_SAFE_CALL(cudaFree(temp_components.means));
  CUDA_SAFE_CALL(cudaFree(temp_components.R));
  CUDA_SAFE_CALL(cudaFree(temp_components.Rinv));
  
  CUDA_SAFE_CALL(cudaFree(d_components));
}

// ==================== Eval data deallocation GPU =================  
void dealloc_evals_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(d_component_memberships));
  CUDA_SAFE_CALL(cudaFree(d_loglikelihoods));
  return;
}

