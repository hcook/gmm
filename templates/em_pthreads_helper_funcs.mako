#define NUM_THREADS 2

struct args_for_pthreads {
    int tid;
    float* data;
    components_t* components;
    float* component_memberships;
    int D;
    int M;
    int N;
};

struct args_for_pthreads_idx {
    int tid;
    float* data;
    components_t* components;
    float* component_memberships;
    int D;
    int M;
    int N;
    int* indices;
    int num_indices;
};

struct args_for_estep2 {
    int tid;
    components_t* components;
    float* component_memberships;
    int M;
    int N;
    float* likelihoods;
};

struct args_for_pthreads args_for_pthreads_array[NUM_THREADS];
struct args_for_pthreads_idx args_for_pthreads_idx_array[NUM_THREADS];
struct args_for_estep2 args_for_estep2_array[NUM_THREADS];

void pthreads_launch_estep2(void *(*kernel)(void*), components_t* components, float* component_memberships, int M, int N, float* likelihoods) {
    args_for_estep2* t_data_arr = args_for_estep2_array;
    pthread_t threads[NUM_THREADS];
    pthread_attr_t attr;
    int rc;
    long t;
    void *status;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for(t=0; t<NUM_THREADS; t++){
        t_data_arr[t].tid = t;
        t_data_arr[t].components = components;
        t_data_arr[t].component_memberships = component_memberships;
        t_data_arr[t].M = M;
        t_data_arr[t].N = N;
        t_data_arr[t].likelihoods = likelihoods;
        rc = pthread_create(&threads[t], NULL, kernel, (void *) &t_data_arr[t]);
        if (rc){
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
    pthread_attr_destroy(&attr);
    for(t=0; t<NUM_THREADS; t++) {
      rc = pthread_join(threads[t], &status);
      if (rc) {
         printf("ERROR; return code from pthread_join() is %d\n", rc);
         exit(-1);
         }
    }
}

void pthreads_launch_mstep(void *(*kernel)(void*), float* data, components_t* components,float* component_memberships, int D, int M, int N) {
    args_for_pthreads* t_data_arr = args_for_pthreads_array;
    pthread_t threads[NUM_THREADS];
    pthread_attr_t attr;
    int rc;
    long t;
    void *status;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for(t=0; t<NUM_THREADS; t++){
        t_data_arr[t].tid = t;
        t_data_arr[t].data = data;
        t_data_arr[t].components = components;
        t_data_arr[t].component_memberships = component_memberships;
        t_data_arr[t].D = D;
        t_data_arr[t].M = M;
        t_data_arr[t].N = N;
        rc = pthread_create(&threads[t], NULL, kernel, (void *) &t_data_arr[t]);
        if (rc){
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
    pthread_attr_destroy(&attr);
    for(t=0; t<NUM_THREADS; t++) {
      rc = pthread_join(threads[t], &status);
      if (rc) {
         printf("ERROR; return code from pthread_join() is %d\n", rc);
         exit(-1);
         }
    }
}

void pthreads_launch_mstep_idx(void *(*kernel)(void*), float* data, int* indices, int num_indices, components_t* components, float* component_memberships, int D, int M, int N) {
    args_for_pthreads_idx* t_data_arr = args_for_pthreads_idx_array;
    pthread_t threads[NUM_THREADS];
    pthread_attr_t attr;
    int rc;
    long t;
    void *status;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for(t=0; t<NUM_THREADS; t++){
        t_data_arr[t].tid = t;
        t_data_arr[t].data = data;
        t_data_arr[t].indices = indices;
        t_data_arr[t].num_indices = num_indices;
        t_data_arr[t].components = components;
        t_data_arr[t].component_memberships = component_memberships;
        t_data_arr[t].D = D;
        t_data_arr[t].M = M;
        t_data_arr[t].N = N;
        rc = pthread_create(&threads[t], NULL, kernel, (void *) &t_data_arr[t]);
        if (rc){
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
    pthread_attr_destroy(&attr);
    for(t=0; t<NUM_THREADS; t++) {
      rc = pthread_join(threads[t], &status);
      if (rc) {
         printf("ERROR; return code from pthread_join() is %d\n", rc);
         exit(-1);
         }
    }
}

