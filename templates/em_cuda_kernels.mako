<%
tempbuff_type_name = 'unsigned int' if supports_32b_floating_point_atomics == '0' else 'float'
%>

__device__ void average_variance${'_'+'_'.join(param_val_list)}(float* fcs_data, float* means, int num_dimensions, int num_events, float* avgvar) {
    // access thread id
    int tid = threadIdx.x;
    
    __shared__ float variances[${max_num_dimensions}];
    __shared__ float total_variance;
    
    // Compute average variance for each dimension
    if(tid < num_dimensions) {
        variances[tid] = 0.0f;
        // Sum up all the variance
        for(int j=0; j < num_events; j++) {
            // variance = (data - mean)^2
            variances[tid] += (fcs_data[j*num_dimensions + tid])*(fcs_data[j*num_dimensions + tid]);
        }
        variances[tid] /= (float) num_events;
        variances[tid] -= means[tid]*means[tid];
    }
    
    __syncthreads();
    
    if(tid == 0) {
        total_variance = 0.0f;
        for(int i=0; i<num_dimensions;i++) {
            ////printf("%f ",variances[tid]);
            total_variance += variances[i];
        }
        ////printf("\nTotal variance: %f\n",total_variance);
        *avgvar = total_variance / (float) num_dimensions;
        ////printf("Average Variance: %f\n",*avgvar);
    }
}

__device__ void compute_constants${'_'+'_'.join(param_val_list)}(components_t* components, int num_components, int num_dimensions) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_elements = num_dimensions*num_dimensions;
    
    __shared__ float determinant_arg; // only one thread computes the inverse so we need a shared argument
    
    float log_determinant;
    
    __shared__ float matrix[${max_num_dimensions}*${max_num_dimensions}];
    
    // Invert the matrix for every component
    int c = blockIdx.x;
    // Copy the R matrix into shared memory for doing the matrix inversion
    for(int i=tid; i<num_elements; i+= num_threads ) {
        matrix[i] = components->R[c*num_dimensions*num_dimensions+i];
    }
    
    __syncthreads(); 
    
    invert(matrix,num_dimensions,&determinant_arg);

    __syncthreads(); 
    
    log_determinant = determinant_arg;
    
    // Copy the matrx from shared memory back into the component memory
    for(int i=tid; i<num_elements; i+= num_threads) {
        components->Rinv[c*num_dimensions*num_dimensions+i] = matrix[i];
    }
    
    __syncthreads();
    
    // Compute the constant
    // Equivilent to: log(1/((2*PI)^(M/2)*det(R)^(1/2)))
    // This constant is used in all E-step likelihood calculations
    if(tid == 0) {
        components->constant[c] = -num_dimensions*0.5*logf(2*PI) - 0.5*log_determinant;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! @param fcs_data         FCS data: [num_events]
//! @param components         Clusters: [num_components]
//! @param num_dimensions   number of dimensions in an FCS event
//! @param num_events       number of FCS events
////////////////////////////////////////////////////////////////////////////////
__global__ void
seed_components${'_'+'_'.join(param_val_list)}( float* fcs_data, components_t* components, int num_dimensions, int num_components, int num_events) 
{
    // access thread id
    int tid = threadIdx.x;
    // access number of threads in this block
    int num_threads = blockDim.x;

    // shared memory
    __shared__ float means[${max_num_dimensions}];
    
    // Compute the means
    mvtmeans(fcs_data, num_dimensions, num_events, means);

    __syncthreads();
    
    __shared__ float avgvar;
    
    // Compute the average variance
    average_variance${'_'+'_'.join(param_val_list)}(fcs_data, means, num_dimensions, num_events, &avgvar);
        
    int num_elements;
    int row, col;
        
    // Number of elements in the covariance matrix
    num_elements = num_dimensions*num_dimensions; 

    __syncthreads();

    float seed;
    if(num_components > 1) {
        seed = (num_events-1.0f)/(num_components-1.0f);
    } else {
        seed = 0.0f;
    }
    
    // Seed the pi, means, and covariances for every component
    for(int c=0; c < num_components; c++) {
        if(tid < num_dimensions) {
            components->means[c*num_dimensions+tid] = fcs_data[((int)(c*seed))*num_dimensions+tid];
        }
          
        for(int i=tid; i < num_elements; i+= num_threads) {
            // Add the average variance divided by a constant, this keeps the cov matrix from becoming singular
            row = (i) / num_dimensions;
            col = (i) % num_dimensions;

            if(row == col) {
                components->R[c*num_dimensions*num_dimensions+i] = 1.0f;
            } else {
                components->R[c*num_dimensions*num_dimensions+i] = 0.0f;
            }
        }
        if(tid == 0) {
            components->pi[c] = 1.0f/((float)num_components);
            components->N[c] = ((float) num_events) / ((float)num_components);
            components->avgvar[c] = avgvar / COVARIANCE_DYNAMIC_RANGE;
        }
    }
}

__device__ void compute_indices${'_'+'_'.join(param_val_list)}(int num_events, int* start, int* stop) {
    // Break up the events evenly between the blocks
    int num_pixels_per_block = num_events / ${num_blocks_estep};
    // Make sure the events being accessed by the block are aligned to a multiple of 16
    num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);
    
    *start = blockIdx.x * num_pixels_per_block + threadIdx.x;
    
    // Last block will handle the leftover events
    if(blockIdx.x == ${num_blocks_estep}-1) {
        *stop = num_events;
    } else {
        *stop = (blockIdx.x+1) * num_pixels_per_block;
    }
}

__global__ void
estep1${'_'+'_'.join(param_val_list)}(float* fcs_data, components_t* components, float *component_memberships, int num_dimensions, int num_events, float* likelihood, float* loglikelihoods) {
    
    // Cached component parameters
    __shared__ float means[${max_num_dimensions}];
    __shared__ float Rinv[${max_num_dimensions}*${max_num_dimensions}];
    float component_pi;
    float constant;
    const unsigned int tid = threadIdx.x;
 
    int start_index;
    int end_index;

    int c = blockIdx.y;

    compute_indices${'_'+'_'.join(param_val_list)}(num_events,&start_index,&end_index);
    
    float like;

    // This loop computes the expectation of every event into every component
    //
    // P(k|n) = L(x_n|mu_k,R_k)*P(k) / P(x_n)
    //
    // Compute log-likelihood for every component for each event
    // L = constant*exp(-0.5*(x-mu)*Rinv*(x-mu))
    // log_L = log_constant - 0.5*(x-u)*Rinv*(x-mu)
    // the constant stored in components[c].constant is already the log of the constant
    
    // copy the means for this component into shared memory
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }

    // copy the covariance inverse into shared memory
    for(int i=tid; i < num_dimensions*num_dimensions; i+= ${num_threads_estep}) {
        Rinv[i] = components->Rinv[c*num_dimensions*num_dimensions+i]; 
    }
    
    component_pi = components->pi[c];
    constant = components->constant[c];

    // Sync to wait for all params to be loaded to shared memory
    __syncthreads();

    
    for(int event=start_index; event<end_index; event += ${num_threads_estep}) {
        like = 0.0f;
        // this does the loglikelihood calculation
        #if ${diag_only}
            for(int j=0; j<num_dimensions; j++) {
                like += (fcs_data[j*num_events+event]-means[j]) * (fcs_data[j*num_events+event]-means[j]) * Rinv[j*num_dimensions+j];
            }
        #else
            for(int i=0; i<num_dimensions; i++) {
                for(int j=0; j<num_dimensions; j++) {
                    like += (fcs_data[i*num_events+event]-means[i]) * (fcs_data[j*num_events+event]-means[j]) * Rinv[i*num_dimensions+j];
                }
            }
        #endif
        loglikelihoods[event] = like;
        component_memberships[c*num_events+event] = -0.5f * like + constant + logf(component_pi); // numerator of the probability computation
    }
}

    
__global__ void
estep2${'_'+'_'.join(param_val_list)}(float* fcs_data, components_t* components, float* component_memberships, int num_dimensions, int num_components, int num_events, float* likelihood) {
    float temp;
    float thread_likelihood = 0.0f;
    __shared__ float total_likelihoods[${num_threads_estep}];
    float max_likelihood;
    float denominator_sum;
    
    // Break up the events evenly between the blocks
    int num_pixels_per_block = num_events / ${num_blocks_estep};
    // Make sure the events being accessed by the block are aligned to a multiple of 16
    num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);
    int tid = threadIdx.x;
    
    int start_index;
    int end_index;
    start_index = blockIdx.x * num_pixels_per_block + tid;
    
    // Last block will handle the leftover events
    if(blockIdx.x == ${num_blocks_estep}-1) {
        end_index = num_events;
    } else {
        end_index = (blockIdx.x+1) * num_pixels_per_block;
    }
    
    total_likelihoods[tid] = 0.0f;

    // P(x_n) = sum of likelihoods weighted by P(k) (their probability, component[c].pi)
    // However we use logs to prevent under/overflow
    //  log-sum-exp formula:
    //  log(sum(exp(x_i)) = max(z) + log(sum(exp(z_i-max(z))))
    for(int pixel=start_index; pixel<end_index; pixel += ${num_threads_estep}) {
        // find the maximum likelihood for this event
        max_likelihood = component_memberships[pixel];
        for(int c=1; c<num_components; c++) {
            max_likelihood = fmaxf(max_likelihood,component_memberships[c*num_events+pixel]);
        }

        // Compute P(x_n), the denominator of the probability (sum of weighted likelihoods)
        denominator_sum = 0.0f;
        for(int c=0; c<num_components; c++) {
            temp = expf(component_memberships[c*num_events+pixel]-max_likelihood);
            denominator_sum += temp;
        }
        denominator_sum = max_likelihood + logf(denominator_sum);
        thread_likelihood += denominator_sum;
        
        // Divide by denominator, also effectively normalize probabilities
        for(int c=0; c<num_components; c++) {
            component_memberships[c*num_events+pixel] = expf(component_memberships[c*num_events+pixel] - denominator_sum);
            //printf("Probability that pixel #%d is in component #%d: %f\n",pixel,c,components->memberships[c*num_events+pixel]);
        }
    }
    
    total_likelihoods[tid] = thread_likelihood;
    __syncthreads();

    parallelSum(total_likelihoods);
    if(tid == 0) {
        likelihood[blockIdx.x] = total_likelihoods[0];
    }
}

__global__ void
mstep_means${'_'+'_'.join(param_val_list)}(float* fcs_data, components_t* components, float* component_memberships, int num_dimensions, int num_components, int num_events) {
    // One block per component, per dimension:  (M x D) grid of blocks
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int c = blockIdx.x; // component number
    int d = blockIdx.y; // dimension number

    __shared__ float temp_sum[${num_threads_mstep}];
    float sum = 0.0f;
    
    for(int event=tid; event < num_events; event+= num_threads) {
        sum += fcs_data[d*num_events+event]*component_memberships[c*num_events+event];
    }
    temp_sum[tid] = sum;
    __syncthreads();
    
    parallelSum(temp_sum);
    if(tid == 0) {
        components->means[c*num_dimensions+d] = temp_sum[0] / components->N[c];
    }
    
}

__global__ void
mstep_means_transpose${'_'+'_'.join(param_val_list)}(float* fcs_data, components_t* components, float* component_memberships, int num_dimensions, int num_components, int num_events) {
    // One block per component, per dimension:  (M x D) grid of blocks
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int c = blockIdx.y; // component number
    int d = blockIdx.x; // dimension number

    __shared__ float temp_sum[${num_threads_mstep}];
    float sum = 0.0f;
    
    for(int event=tid; event < num_events; event+= num_threads) {
        sum += fcs_data[d*num_events+event]*component_memberships[c*num_events+event];
    }
    temp_sum[tid] = sum;
    
    __syncthreads();
    
    parallelSum(temp_sum);
    if(tid == 0) {
        components->means[c*num_dimensions+d] = temp_sum[0] / components->N[c];
    }
    
}

__global__ void
mstep_N${'_'+'_'.join(param_val_list)}(float* fcs_data, components_t* components, float* component_memberships, int num_dimensions, int num_components, int num_events) {
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int c = blockIdx.x;
 
    
    // Need to store the sum computed by each thread so in the end
    // a single thread can reduce to get the final sum
    __shared__ float temp_sums[${num_threads_mstep}];

    // Compute new N
    float sum = 0.0f;
    // Break all the events accross the threads, add up probabilities
    for(int event=tid; event < num_events; event += num_threads) {
        sum += component_memberships[c*num_events+event];
    }
    temp_sums[tid] = sum;
 
    __syncthreads();
    
    parallelSum(temp_sums);
    if(tid == 0) {
        components->N[c] = temp_sums[0];
        components->pi[c] = temp_sums[0];
    }
}
   
 
%if covar_version_name.upper() in ['1','V1','_V1']:

__global__ void
mstep_covariance${'_'+'_'.join(param_val_list)}(float* fcs_data, components_t* components, float* component_memberships, int num_dimensions, int num_components, int num_events) {
    int tid = threadIdx.x; // easier variable name for our thread ID
    int row,col,c;
    compute_row_col(num_dimensions, &row, &col);
    c = blockIdx.x; // Determines what component this block is handling    

    int matrix_index = row * num_dimensions + col;

    // Store the means of this component in shared memory
    __shared__ float means[${max_num_dimensions}];
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }
    __syncthreads();

    __shared__ float temp_sums[${num_threads_mstep}];
    
    float cov_sum = 0.0f;
    for(int event=tid; event < num_events; event+=${num_threads_mstep}) {
      cov_sum += (fcs_data[row*num_events+event]-means[row])*(fcs_data[col*num_events+event]-means[col])*component_memberships[c*num_events+event];
    }
    temp_sums[tid] = cov_sum;

    __syncthreads();
    
    parallelSum(temp_sums);
    if(tid == 0) {
      cov_sum = temp_sums[0];
      if(components->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
        cov_sum /= components->N[c];
        components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
      } else {
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
      }
      if(row == col) {
        components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
      }
    }
}

void mstep_covar_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, components_t* d_components, float* d_component_memberships, int num_dimensions, int num_components, int num_events, ${tempbuff_type_name}* temp_buffer_2b)
{
  dim3 gridDim2(num_components,num_dimensions*(num_dimensions+1)/2);
  mstep_covariance${'_'+'_'.join(param_val_list)}<<<gridDim2, ${num_threads_mstep}>>>(d_fcs_data_by_dimension,d_components,d_component_memberships,num_dimensions,num_components,num_events);
}

%elif covar_version_name.upper() in ['2','2A','V2','V2A','_V2','_V2A']:

/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a M blocks and D x D/2 threads: 
 */
__global__ void
mstep_covariance${'_'+'_'.join(param_val_list)}(float* fcs_data, components_t* components, float* component_memberships, int num_dimensions, int num_components, int num_events) {
    int tid = threadIdx.x; // easier variable name for our thread ID
    int row,col,c;
    compute_row_col_thread(num_dimensions, &row, &col);

    __syncthreads();
    
    c = blockIdx.x; // Determines what component this block is handling    

    int matrix_index = row * num_dimensions + col;

    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[${max_num_dimensions}];
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }
    __syncthreads();

    float cov_sum = 0.0f; //my local sum for the matrix element, I (thread) sum up over all N events into this var

    if(tid < num_dimensions*(num_dimensions+1)/2) {
        for(int event=0; event < num_events; event++) {
          cov_sum += (fcs_data[event*num_dimensions+row]-means[row])*(fcs_data[event*num_dimensions+col]-means[col])*component_memberships[c*num_events+event];
        }
    }

    __syncthreads();

    if(tid < num_dimensions*(num_dimensions+1)/2) {
        if(components->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
          cov_sum /= components->N[c];
          components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
          // Set the symmetric value
          matrix_index = col*num_dimensions+row;
          components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
        } else {
          components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty component...?
          // Set the symmetric value
          matrix_index = col*num_dimensions+row;
          components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty component...?
        }
        
        // Regularize matrix - adds some variance to the diagonal elements
        // Helps keep covariance matrix non-singular (so it can be inverted)
        // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined at top of this file
        if(row == col) {
          components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
        }
    }   
}

void mstep_covar_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, components_t* d_components, float* d_component_memberships, int num_dimensions, int num_components, int num_events, ${tempbuff_type_name}* temp_buffer_2b)
{
  int num_threads = num_dimensions*(num_dimensions+1)/2;
  mstep_covariance${'_'+'_'.join(param_val_list)}<<<num_components, num_threads>>>(d_fcs_data_by_event,d_components,d_component_memberships,num_dimensions,num_components,num_events);
}

%elif covar_version_name.upper() in ['2B','V2B','_V2B']:
/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a M x B blocks and D x D/2 threads:
 * B is the number of event blocks (N/events_per_block)
 */
__global__ void
mstep_covariance${'_'+'_'.join(param_val_list)}(float* fcs_data, components_t* components, float* component_memberships, int num_dimensions, int num_components, int num_events, int event_block_size, int num_b, ${tempbuff_type_name}* temp_buffer) {
  int tid = threadIdx.x; // easier variable name for our thread ID
    
    // Determine what row,col this matrix is handling, also handles the symmetric element
    int row,col,c;
    compute_row_col_thread(num_dimensions, &row, &col);

    int e_start, e_end;
    compute_my_event_indices(num_events, event_block_size, num_b, &e_start, &e_end);
    c = blockIdx.x; // Determines what component this block is handling    
    int matrix_index = row * num_dimensions + col;

    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[${max_num_dimensions}];
    __shared__ float myR[${max_num_dimensions}*${max_num_dimensions}];
    
    // copy the means for this component into shared memory
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }
    __syncthreads();

    float cov_sum = 0.0f; //my local sum for the matrix element, I (thread) sum up over all N events into this var

    if(tid < num_dimensions*(num_dimensions+1)/2) {
        for(int event=e_start; event < e_end; event++) {
          cov_sum += (fcs_data[event*num_dimensions+row]-means[row])*(fcs_data[event*num_dimensions+col]-means[col])*component_memberships[c*num_events+event];
        }

        myR[matrix_index] = cov_sum;
     
%if supports_32b_floating_point_atomics != '0':
        float old = atomicAdd(&(temp_buffer[c*num_dimensions*num_dimensions+matrix_index]), myR[matrix_index]); 
%else:
        unsigned int fixp_myR = (unsigned int)floor((myR[matrix_index])*1000000.0f);
        unsigned int old = atomicAdd(&(temp_buffer[c*num_dimensions*num_dimensions+matrix_index]), fixp_myR); 
%endif
    }

    __syncthreads();

    if(tid < num_dimensions*(num_dimensions+1)/2) {
      if(components->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
%if supports_32b_floating_point_atomics != '0':
        float cs = temp_buffer[c*num_dimensions*num_dimensions+matrix_index];
%else:
        float cs = (((float)temp_buffer[c*num_dimensions*num_dimensions+matrix_index])/1000000.0f);
%endif
        cs /= components->N[c];
        components->R[c*num_dimensions*num_dimensions+matrix_index] = cs;
        // Set the symmetric value
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = cs;
      } else {
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty component...?
        // Set the symmetric value
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; // what should the variance be for an empty component...?
      }
    
      // Regularize matrix - adds some variance to the diagonal elements
      // Helps keep covariance matrix non-singular (so it can be inverted)
      // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined at top of this file
      if(row == col) {
        components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
      }
    }
}
 
void mstep_covar_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, components_t* d_components, float* d_component_memberships, int num_dimensions, int num_components, int num_events, ${tempbuff_type_name}* temp_buffer_2b)
{
  int num_event_blocks = ${num_event_blocks};
  int event_block_size = num_events%${num_event_blocks} == 0 ? num_events/${num_event_blocks}:num_events/(${num_event_blocks}-1);
  dim3 gridDim2(num_components,num_event_blocks);
  int num_threads = num_dimensions*(num_dimensions+1)/2;
  mstep_covariance${'_'+'_'.join(param_val_list)}<<<gridDim2, num_threads>>>(d_fcs_data_by_event,d_components,d_component_memberships,num_dimensions,num_components,num_events, event_block_size, num_event_blocks, temp_buffer_2b);
}

%else:

/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a D*D/2 blocks: 
 */
__global__ void
mstep_covariance${'_'+'_'.join(param_val_list)}(float* fcs_data, components_t* components, float* component_memberships, int num_dimensions, int num_components, int num_events) {
  int tid = threadIdx.x; // easier variable name for our thread ID
  int row,col;
  compute_row_col_block(num_dimensions, &row, &col);
      
  int matrix_index;
    
  // Store ALL the means in shared memory
  __shared__ float means[${max_num_components_covar_v3}*${max_num_dimensions_covar_v3}];
  for(int i = tid; i<num_components*num_dimensions; i+=${num_threads_mstep}) {
    means[i] = components->means[i];
  }
  __syncthreads();

  __shared__ float temp_sums[${num_threads_mstep}];
  __shared__ float component_sum[${max_num_components_covar_v3}]; //local storage for component results
  
  for(int c = 0; c<num_components; c++) {
    float cov_sum = 0.0f;
    for(int event=tid; event < num_events; event+=${num_threads_mstep}) {
      cov_sum += (fcs_data[row*num_events+event]-means[c*num_dimensions+row])*(fcs_data[col*num_events+event]-means[c*num_dimensions+col])*component_memberships[c*num_events+event];
    }
    temp_sums[tid] = cov_sum;
    
    __syncthreads();
      
    parallelSum(temp_sums);
    if(tid == 0) {
      component_sum[c] = temp_sums[0]; 
    }
    __syncthreads();
  }
  __syncthreads();
    
  for(int c = tid; c<num_components; c+=${num_threads_mstep}) {
    matrix_index =  row * num_dimensions + col;
    if(components->N[c] >= 1.0f) { // Does it need to be >=1, or just something non-zero?
      component_sum[c] /= components->N[c];
      components->R[c*num_dimensions*num_dimensions+matrix_index] = component_sum[c];
      matrix_index = col*num_dimensions+row;
      components->R[c*num_dimensions*num_dimensions+matrix_index] = component_sum[c];
    } else {
      components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
      matrix_index = col*num_dimensions+row;
      components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
    }
    if(row == col) {
      components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
    }
  } 
}

void mstep_covar_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, components_t* d_components, float* d_component_memberships, int num_dimensions, int num_components, int num_events, ${tempbuff_type_name}* temp_buffer_2b)
{
  int num_blocks = num_dimensions*(num_dimensions+1)/2;
  mstep_covariance${'_'+'_'.join(param_val_list)}<<<num_blocks, ${num_threads_mstep}>>>(d_fcs_data_by_dimension,d_components,d_component_memberships,num_dimensions,num_components,num_events);
}

%endif

/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a M x D*D grid of blocks: 
 *  i.e. dim3 gridDim(num_components,num_dimensions*num_dimensions)
 */
__global__ void
mstep_covariance_transpose${'_'+'_'.join(param_val_list)}(float* fcs_data, components_t* components, float* component_memberships, int num_dimensions, int num_components, int num_events) {
  int tid = threadIdx.x; // easier variable name for our thread ID

    // Determine what row,col this matrix is handling, also handles the symmetric element
    int row,col,c;
    compute_row_col_transpose(num_dimensions, &row, &col);

    __syncthreads();
    
    c = blockIdx.y; // Determines what component this block is handling    

    int matrix_index = row * num_dimensions + col;

    #if ${diag_only}
    if(row != col) {
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        matrix_index = col*num_dimensions+row;
        components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f;
        return;
    }
    #endif 

    // Store the means in shared memory to speed up the covariance computations
    __shared__ float means[${max_num_dimensions}];
    // copy the means for this component into shared memory
    if(tid < num_dimensions) {
        means[tid] = components->means[c*num_dimensions+tid];
    }

    // Sync to wait for all params to be loaded to shared memory
    __syncthreads();

    __shared__ float temp_sums[${num_threads_mstep}];
    
    float cov_sum = 0.0f;

    for(int event=tid; event < num_events; event+=${num_threads_mstep}) {
        cov_sum += (fcs_data[row*num_events+event]-means[row])*(fcs_data[col*num_events+event]-means[col])*component_memberships[c*num_events+event]; 
    }
    temp_sums[tid] = cov_sum;

    __syncthreads();
    
    parallelSum(temp_sums);
    if(tid == 0) {
        cov_sum = temp_sums[0];
        if(components->N[c] >= 1.0f) { 
            cov_sum /= components->N[c];
            components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
            matrix_index = col*num_dimensions+row;
            components->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
        } else {
            components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
            matrix_index = col*num_dimensions+row;
            components->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0f; 
        }
        if(row == col) {
            components->R[c*num_dimensions*num_dimensions+matrix_index] += components->avgvar[c];
        }
    }
}


void seed_components_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_event, components_t* d_components, int num_dimensions, int original_num_components, int num_events)
{
  seed_components${'_'+'_'.join(param_val_list)}<<< 1, ${num_threads_mstep} >>>( d_fcs_data_by_event, d_components, num_dimensions, original_num_components, num_events);
}

void estep1_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, components_t* d_components, float* d_component_memberships, int num_dimensions, int num_events, float* d_likelihoods, int num_components, float* d_loglikelihoods)
{
  estep1${'_'+'_'.join(param_val_list)}<<<dim3(${num_blocks_estep},num_components), ${num_threads_estep}>>>(d_fcs_data_by_dimension,d_components,d_component_memberships,num_dimensions,num_events,d_likelihoods, d_loglikelihoods);
}

void estep2_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, components_t* d_components, float* d_component_memberships, int num_dimensions, int num_components, int num_events, float* d_likelihoods)
{
  estep2${'_'+'_'.join(param_val_list)}<<<${num_blocks_estep}, ${num_threads_estep}>>>(d_fcs_data_by_dimension,d_components,d_component_memberships,num_dimensions,num_components,num_events,d_likelihoods);
}

void mstep_N_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_event, components_t* d_components, float* d_component_memberships, int num_dimensions, int num_components, int num_events)
{
  mstep_N${'_'+'_'.join(param_val_list)}<<<num_components, ${num_threads_mstep}>>>(d_fcs_data_by_event,d_components,d_component_memberships,num_dimensions,num_components,num_events);
}

void mstep_means_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, components_t* d_components, float* d_component_memberships, int num_dimensions, int num_components, int num_events)
{
  dim3 gridDim1(num_components,num_dimensions);
  mstep_means${'_'+'_'.join(param_val_list)}<<<gridDim1, ${num_threads_mstep}>>>(d_fcs_data_by_dimension,d_components,d_component_memberships,num_dimensions,num_components,num_events);
}

/*
 * Computes the constant for each component and normalizes pi for every component
 * In the process it inverts R and finds the determinant
 * 
 * Needs to be launched with the number of blocks = number of components
 */
__global__ void
constants_kernel${'_'+'_'.join(param_val_list)}(components_t* components, int num_components, int num_dimensions) {
    compute_constants${'_'+'_'.join(param_val_list)}(components,num_components,num_dimensions);
    
    __syncthreads();
    
    if(blockIdx.x == 0) {
        normalize_pi(components,num_components);
    }
}

void constants_kernel_launch${'_'+'_'.join(param_val_list)}(components_t* d_components, int original_num_components, int num_dimensions)
{
  constants_kernel${'_'+'_'.join(param_val_list)}<<<original_num_components, 64>>>(d_components,original_num_components,num_dimensions);
}

