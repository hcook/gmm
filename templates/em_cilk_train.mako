
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

boost::python::tuple em_cilk_train${'_'+'_'.join(param_val_list)} (
                             int num_components, 
                             int num_dimensions, 
                             int num_events) 
{
    
    // seed_components sets initial pi values, 
    // finds the means / covariances and copies it to all the components
    seed_components${'_'+'_'.join(param_val_list)}(fcs_data_by_event, &components, num_dimensions, num_components, num_events);
   
    // Computes the R matrix inverses, and the gaussian constant
    constants${'_'+'_'.join(param_val_list)}(&components,num_components,num_dimensions);

    // Calculate an epsilon value
    //int ndata_points = num_events*num_dimensions;
    float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)num_events*num_dimensions)*0.0001;
    float likelihood, old_likelihood;
    int iters;
    
    /*************** EM ALGORITHM *****************************/
    
    // do initial regrouping
    estep1${'_'+'_'.join(param_val_list)}(fcs_data_by_event,&components,component_memberships,num_dimensions,num_components,num_events,&likelihood,loglikelihoods);
    estep2${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events,&likelihood);

    float change = epsilon*2;
    
    iters = 0;
    // This is the iterative loop for the EM algorithm.
    // It re-estimates parameters, re-computes constants, and then regroups the events
    // These steps keep repeating until the change in likelihood is less than some epsilon        
    while(iters < ${min_iters} || (fabs(change) > epsilon && iters < ${max_iters})) {
        old_likelihood = likelihood;
        
        // This kernel computes a new N, pi isn't updated until compute_constants though
        mstep_n${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events);
        mstep_mean${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events);
        mstep_covar${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events);
        
        // Inverts the R matrices, computes the constant, normalizes cluster probabilities
        constants${'_'+'_'.join(param_val_list)}(&components,num_components,num_dimensions);

        // Compute new cluster membership probabilities for all the events
        estep1${'_'+'_'.join(param_val_list)}(fcs_data_by_event,&components,component_memberships,num_dimensions,num_components,num_events,&likelihood,loglikelihoods);
        estep2${'_'+'_'.join(param_val_list)}(fcs_data_by_dimension,&components,component_memberships,num_dimensions,num_components,num_events,&likelihood);
    
        change = likelihood - old_likelihood;
        //printf("likelihood = %f\n",likelihood);
        //printf("Change in likelihood: %f\n",change);

        iters++;
    }
    
  return boost::python::make_tuple(likelihood, iters);
}
