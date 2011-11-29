
void seed_components${'_'+'_'.join(param_val_list)}(float *data, components_t* components, int D, int M, int N);
void constants${'_'+'_'.join(param_val_list)}(components_t* components, int M, int D);
void estep1${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N, float* likelihood, float* loglikelihoods);
void estep2${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N, float* likelihood);
void mstep_n${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N);
void mstep_mean${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N);
void mstep_covar${'_'+'_'.join(param_val_list)}(float* data, components_t* components,float* component_memberships, int D, int M, int N);
