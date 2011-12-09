<%
tempbuff_type_name = 'unsigned int' if supports_float32_atomic_add == '0' else 'float'
%>

void seed_components_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_event, components_t* d_components, int num_dimensions, int original_num_components, int num_events);
void constants_kernel_launch${'_'+'_'.join(param_val_list)}(components_t* d_components, int original_num_components, int num_dimensions);
void estep1_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, components_t* d_components, float* component_memberships, int num_dimensions, int num_events, float* d_likelihoods, int num_components, float* d_loglikelihoods);
void estep2_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, components_t* d_components, float* component_memberships, int num_dimensions, int num_components, int num_events, float* d_likelihoods);
void mstep_N_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_event, components_t* d_components, float* component_memberships, int num_dimensions, int num_components, int num_events);
void mstep_means_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, components_t* d_components, float* component_memberships, int num_dimensions, int num_components, int num_events);
void mstep_covar_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, components_t* d_components, float* component_memberships, int num_dimensions, int num_components, int num_events, ${tempbuff_type_name}* temp_buffer_2b);
