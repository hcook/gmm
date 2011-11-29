
void seed_components${'_'+'_'.join(param_val_list)}(float *data, components_t* components, int D, int M, int N) {
    float* variances = (float*) malloc(sizeof(float)*D);
    float* means = (float*) malloc(sizeof(float)*D);

    // Compute means
    for(int d=0; d < D; d++) {
        means[d] = 0.0;
        for(int n=0; n < N; n++) {
            means[d] += data[n*D+d];
        }
        means[d] /= (float) N;
    }

    // Compute variance of each dimension
    for(int d=0; d < D; d++) {
        variances[d] = 0.0;
        for(int n=0; n < N; n++) {
            variances[d] += data[n*D+d]*data[n*D+d];
        }
        variances[d] /= (float) N;
        variances[d] -= means[d]*means[d];
    }

    // Average variance
    float avgvar = 0.0;
    for(int d=0; d < D; d++) {
        avgvar += variances[d];
    }
    avgvar /= (float) D;

    // Initialization for random seeding and uniform seeding    
    float fraction;
    int seed;
    if(M > 1) {
        fraction = (N-1.0f)/(M-1.0f);
    } else {
        fraction = 0.0;
    }

    // Cilk Plus: Needed to be set below to make the program run.
    srand(clock());

    for(int m=0; m < M; m++) {
        components->N[m] = (float) N / (float) M;
        components->pi[m] = 1.0f / (float) M;
        components->avgvar[m] = avgvar / COVARIANCE_DYNAMIC_RANGE;

        // Choose component centers
        #if UNIFORM_SEED
            for(int d=0; d < D; d++) {
                components->means[m*D+d] = data[((int)(m*fraction))*D+d];
            }
        #else
            seed = rand() % N;
            for(int d=0; d < D; d++) {
                components->means[m*D+d] = data[seed*D+d];
            }
        #endif

        // Set covariances to identity matrices
        for(int i=0; i < D; i++) {
            for(int j=0; j < D; j++) {
                if(i == j) {
                    components->R[m*D*D+i*D+j] = 1.0f;
                } else {
                    components->R[m*D*D+i*D+j] = 0.0f;
                }
            }
        }
    }
    free(variances);
    free(means);
}

void constants${'_'+'_'.join(param_val_list)}(components_t* components, int M, int D) {
	float log_determinant;
    float* matrix = (float*) malloc(sizeof(float)*D*D);

    float sum = 0.0;
    for(int m=0; m < M; m++) {
        // Invert covariance matrix
        memcpy(matrix,&(components->R[m*D*D]),sizeof(float)*D*D);
        invert_cpu(matrix,D,&log_determinant);
        memcpy(&(components->Rinv[m*D*D]),matrix,sizeof(float)*D*D);
    
        // Compute constant
        components->constant[m] = -D*0.5f*logf(2.0f*PI) - 0.5f*log_determinant;

        // Sum for calculating pi values
        sum += components->N[m];
    }

    // Compute pi values
    for(int m=0; m < M; m++) {
        components->pi[m] = components->N[m] / sum;
    }
    
    free(matrix);
}

void estep1${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N, float* likelihood, float* loglikelihoods) {
    // Compute likelihood for every data point in each component
    cilk_for(int m=0; m < M; m++) {
        cilk_for(int n=0; n < N; n++) {
            float* means = (float*) &(components->means[m*D]);
            float* Rinv = (float*) &(components->Rinv[m*D*D]);
            float like = 0.0;
            #if ${diag_only}
            for(int i=0; i < D; i++) {
                like += (data[i+n*D]-means[i])*(data[i+n*D]-means[i])*Rinv[i*D+i];
            }
            #else
            for(int i=0; i < D; i++) {
                for(int j=0; j < D; j++) {
                    like += (data[i+n*D]-means[i])*(data[j+n*D]-means[j])*Rinv[i*D+j];
                }
            }
            #endif  
            loglikelihoods[n] = like;
            component_memberships[m*N+n] = -0.5f * like + components->constant[m] + log(components->pi[m]); 
        }
    }
}

float estep2_events${'_'+'_'.join(param_val_list)}(components_t* components, float* component_memberships, int M, int n, int N) {
	// Finding maximum likelihood for this data point
	float max_likelihood;
	max_likelihood = __sec_reduce_max(component_memberships[n:M:N]);

	// Computes sum of all likelihoods for this event
	float denominator_sum;
	denominator_sum = 0.0f;
	for(int m=0; m < M; m++) {
		denominator_sum += exp(component_memberships[m*N+n] - max_likelihood);
	}
	denominator_sum = max_likelihood + log(denominator_sum);

	// Divide by denominator to get each membership
	for(int m=0; m < M; m++) {
		component_memberships[m*N+n] = exp(component_memberships[m*N+n] - denominator_sum);
	}
        //or component_memberships[n:M:N] = exp(component_memberships[n:M:N] - denominator_sum);

	return denominator_sum;
}

void estep2${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N, float* likelihood) {
    cilk::reducer_opadd<float> total(0.0f);
    cilk_for(int n=0; n < N; n++) {
        total += estep2_events${'_'+'_'.join(param_val_list)}(components, component_memberships, M, n, N);
    }
    *likelihood = total.get_value();
}

void mstep_n${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N) {
    for(int m=0; m < M; m++) {
        components->N[m] = 0.0;
        for(int n=0; n < N; n++) {
            components->N[m] += component_memberships[m*N+n];
        }
    }
}

void mstep_mean${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N) {
    cilk_for(int m=0; m < M; m++) {
        for(int d=0; d < D; d++) {
	    components->means[m*D+d] = 0.0;
	    for(int n=0; n < N; n++) {
		components->means[m*D+d] += data[d*N+n]*component_memberships[m*N+n];
	    }
	    components->means[m*D+d] /= components->N[m];
        }
    }
}

void mstep_covar${'_'+'_'.join(param_val_list)}(float* data, components_t* components,float* component_memberships, int D, int M, int N) {
    cilk_for(int m=0; m < M; m++) {
        float* means = &(components->means[m*D]);
        float sum;
        for(int i=0; i < D; i++) {
            for(int j=0; j <= i; j++) {
                #if ${diag_only}
                if(i != j) {
                    components->R[m*D*D+i*D+j] = 0.0f;
                    components->R[m*D*D+j*D+i] = 0.0f;
                    continue;
                }
                #endif
                sum = 0.0;
                for(int n=0; n < N; n++) {
                    sum += (data[i*N+n]-means[i])*(data[j*N+n]-means[j])*component_memberships[m*N+n];
                }
                if(components->N[m] >= 1.0f) {
                    components->R[m*D*D+i*D+j] = sum / components->N[m];
                    components->R[m*D*D+j*D+i] = sum / components->N[m];
                } else {
                    components->R[m*D*D+i*D+j] = 0.0f;
                    components->R[m*D*D+j*D+i] = 0.0f;
                }
            }
        }
    }
}
