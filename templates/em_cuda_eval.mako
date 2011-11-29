
float em_cuda_eval${'_'+'_'.join(param_val_list)} (
                             int num_components, 
                             int num_dimensions, 
                             int num_events) 
{
  float likelihood;
  float* likelihoods = (float*) malloc(sizeof(float)*${num_blocks_estep});
  float* d_likelihoods;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_likelihoods, sizeof(float)*${num_blocks_estep}));

  //TODO: Is this necessary, or can we assume the values are still set?
  // Computes the R matrix inverses, and the gaussian constant
  constants_kernel_launch${'_'+'_'.join(param_val_list)}(d_components,num_components,num_dimensions);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Constants Kernel execution failed: ");

  estep1_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_components, d_component_memberships, num_dimensions,num_events,d_likelihoods,num_components,d_loglikelihoods);
  estep2_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_components, d_component_memberships, num_dimensions,num_components,num_events,d_likelihoods);
  cudaThreadSynchronize();

  CUT_CHECK_ERROR("Kernel execution failed");

  // Copy the likelihood totals from each block, sum them up to get a total
  CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*${num_blocks_estep},cudaMemcpyDeviceToHost));
  likelihood = 0.0;
  for(int i=0;i<${num_blocks_estep};i++) {
    likelihood += likelihoods[i]; 
  }

  copy_evals_data_GPU_to_CPU(num_events, num_components);

  return likelihood; 

}
