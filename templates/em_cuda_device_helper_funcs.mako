#define PI  3.1415926535897931
#define COVARIANCE_DYNAMIC_RANGE 1E6

/*
 * Compute the multivariate mean of the FCS data
 */ 
__device__ void mvtmeans(float* fcs_data, int num_dimensions, int num_events, float* means) {
    // access thread id
    int tid = threadIdx.x;

    if(tid < num_dimensions) {
        means[tid] = 0.0f;

        // Sum up all the values for the dimension
        for(int i=0; i < num_events; i++) {
            means[tid] += fcs_data[i*num_dimensions+tid];
        }

        // Divide by the # of elements to get the average
        means[tid] /= (float) num_events;
    }
}

// Inverts an NxN matrix 'data' stored as a 1D array in-place
// 'actualsize' is N
// Computes the log of the determinant of the origianl matrix in the process
__device__ void invert(float* data, int actualsize, float* log_determinant)  {
    int maxsize = actualsize;
    int n = actualsize;
    
    if(threadIdx.x == 0) {
        *log_determinant = 0.0f;
      // sanity check        
      if (actualsize == 1) {
        *log_determinant = logf(data[0]);
        data[0] = 1.0 / data[0];
      } else {

          for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0
          for (int i=1; i < actualsize; i++)  { 
            for (int j=i; j < actualsize; j++)  { // do a column of L
              float sum = 0.0f;
              for (int k = 0; k < i; k++)  
                  sum += data[j*maxsize+k] * data[k*maxsize+i];
              data[j*maxsize+i] -= sum;
              }
            if (i == actualsize-1) continue;
            for (int j=i+1; j < actualsize; j++)  {  // do a row of U
              float sum = 0.0f;
              for (int k = 0; k < i; k++)
                  sum += data[i*maxsize+k]*data[k*maxsize+j];
              data[i*maxsize+j] = 
                 (data[i*maxsize+j]-sum) / data[i*maxsize+i];
              }
            }
            
            for(int i=0; i<actualsize; i++) {
                *log_determinant += logf(fabs(data[i*n+i]));
            }
            
          for ( int i = 0; i < actualsize; i++ )  // invert L
            for ( int j = i; j < actualsize; j++ )  {
              float x = 1.0f;
              if ( i != j ) {
                x = 0.0f;
                for ( int k = i; k < j; k++ ) 
                    x -= data[j*maxsize+k]*data[k*maxsize+i];
                }
              data[j*maxsize+i] = x / data[j*maxsize+j];
              }
          for ( int i = 0; i < actualsize; i++ )   // invert U
            for ( int j = i; j < actualsize; j++ )  {
              if ( i == j ) continue;
              float sum = 0.0f;
              for ( int k = i; k < j; k++ )
                  sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
              data[i*maxsize+j] = -sum;
              }
          for ( int i = 0; i < actualsize; i++ )   // final inversion
            for ( int j = 0; j < actualsize; j++ )  {
              float sum = 0.0f;
              for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
                  sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
              data[j*maxsize+i] = sum;
              }
        }
    }
}

__device__ void normalize_pi(components_t* components, int num_components) {
    __shared__ float sum;
    
    // TODO: could maybe use a parallel reduction..but the # of elements is really small
    // What is better: having thread 0 compute a shared sum and sync, or just have each one compute the sum?
    if(threadIdx.x == 0) {
        sum = 0.0f;
        for(int i=0; i<num_components; i++) {
            sum += components->pi[i];
        }
    }
    
    __syncthreads();
    
    for(int c=threadIdx.x; c < num_components; c += blockDim.x) {
        components->pi[threadIdx.x] /= sum;
    }
 
    __syncthreads();
}

__device__ void parallelSum(float* data) {
  const unsigned int tid = threadIdx.x;
  for(unsigned int s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s)
      data[tid] += data[tid + s];  
    __syncthreads();
  }
  if (tid < 32) {
    volatile float* sdata = data;
    sdata[tid] += sdata[tid+32];
    sdata[tid] += sdata[tid+16];
    sdata[tid] += sdata[tid+8];
    sdata[tid] += sdata[tid+4];
    sdata[tid] += sdata[tid+2];
    sdata[tid] += sdata[tid+1];
  }
}

/*
 * Computes the row and col of a square matrix based on the index into
 * a lower triangular (with diagonal) matrix
 * 
 * Used to determine what row/col should be computed for covariance
 * based on a block index.
 */
__device__ void compute_row_col(int n, int* row, int* col) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if(i == blockIdx.y) {  
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}

//CODEVAR_2
__device__ void compute_row_col_thread(int n, int* row, int* col) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if(i == threadIdx.x) {  
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}
//CODEVAR_3
__device__ void compute_row_col_block(int n, int* row, int* col) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if(i == blockIdx.x) {  
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}

//CODEVAR_2B and CODEVAR_3B
__device__ void compute_my_event_indices(int n, int bsize, int num_b, int* e_start, int* e_end) {
  int myId = blockIdx.y;
  *e_start = myId*bsize;
  if(myId==(num_b-1)) {
    *e_end = ((myId*bsize)-n < 0 ? n:myId*bsize);
  } else {
    *e_end = myId*bsize + bsize;
  }
  
  return;
}


__device__ void compute_row_col_transpose(int n, int* row, int* col) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if(i == blockIdx.x) {  
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}

