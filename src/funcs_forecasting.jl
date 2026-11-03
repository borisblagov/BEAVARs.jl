
#--------------------------------------
# forecast evaluation 
#--------------------------------------



"""
    Houses the forecast errors, averaged across draws and predictive likelihood for a specific vintage
"""
struct eval_vint_CPZ2023{T <: AbstractFloat, N} <: BVARmodelEval
    pred_lik_mat::Array{T,N}  
    fcast_errors_mAd_mat::Array{T,N}
    fcastDatesOverlap::Union{Array{Date, 1}, Array{DateTime, 1}}
    data_true_VecView::Array{T,N}
    data_true_dates::Union{Array{Date, 1}, Array{DateTime, 1}}
end





"""
    eval_forecast(out_struct::BVARmodelOutput, data_struct::BVARmodelDataSetup, set_struct::BVARmodelSetup, dataLF_true_ftab::TimeArray)

    Evaluates the forecasts from the model output against the true values in dataLF_true_ftab. It calculates the predictive likelihood and the mean forecast error for each variable and each forecast horizon.

    Arguments:
        out_struct: A structure with the model output, including the store_β and YY matrices
        data_struct: A structure with the data setup, including the dataHF_tab and dataLF_tab
        set_struct: A structure with the model setup, including n_fcst and p
        dataLF_true_ftab: A TimeArray with the true values of the low-frequency variables for evaluation

    Returns:
        pred_lik_mat: A matrix with the predictive likelihood for each evaluated variable and forecast horizon
        fcast_errors_mean_mat: A matrix with the mean forecast error for each evaluated variable and forecast horizon

    See also:
        - BEAVARs.pred_lik_CCM() for calculating the predictive likelihood as in Carriero et al 2013 JAE

"""
function eval_forecast(out_struct::BVARmodelOutput, data_struct::BVARmodelDataSetup, set_struct::BVARmodelSetup, dataLF_true_ftab::TimeArray)  
    @unpack fdatesLF,store_YY_LF,freq_mix_tp = out_struct;  
    @unpack dataLF_tab, var_list = data_struct
    @unpack n_fcst, p, n_save = set_struct

    # check if we are working in levels to transform the true data to growth rates for evaluation
    if freq_mix_tp[3] == 1
        dataLF_tab = percentchange(dataLF_tab);
        dataLF_true_ftab = percentchange(dataLF_true_ftab);
        fdatesLF = out_struct.fdatesLF[2:end];              # we lose the first date when we take growth rates, so we need to adjust the forecast dates accordingly
        store_YY_LF = out_struct.store_YY_LF[2:end,:,:]./out_struct.store_YY_LF[1:end-1,:,:].-1;
    end

    var_list_true = colnames(dataLF_true_ftab);             # list of symbols of variables that will be evaluated
    data_eval_tab = dataLF_tab[:,var_list_true];            # sorting a subset of all low-frequency variables so that they will be evaluated in the correct order

    no_nan_flag_mat = .!isnan.(values(data_eval_tab));      # whether in the low-frequency data we have missing values, in the order of the variables in the true table (i.e. unbalanced panel for the low-freq variable)
    # if no_nan_flag_mat is the same across all variables, we can do the evaluation for all variables at the same time, otherwise we need to do it separately for each variable
    data_truef_mat = values(dataLF_true_ftab);
    datesLF_true = timestamp(dataLF_true_ftab)
    n_fvar = size(dataLF_true_ftab,2)                   # evaluating forecasts only for the variables for which we have true data
    # logdensKDE_3dmat = fill(NaN,(n_fcst,n_fvar));   # houses the log-density of each evaluated variable and forecast horizon
    datesLF = timestamp(data_eval_tab);

    pred_lik_mat = fill(NaN,(n_fcst,n_fvar));           # houses the predictive likelihood for each evaluated variable and forecast horizon
    fcast_errors_mAd_mat = fill(NaN,(n_fcst,n_fvar));   # houses the forecast error for each evaluated variable and forecast horizon, mean over all draws


    if allequal(eachcol(no_nan_flag_mat))
        locs = [findfirst(==(s), var_list) for s in var_list_true]
        obs_datesLF = datesLF[no_nan_flag_mat[:,1]];                    # dates for which we have obs, everthing else is forecast, we may simply take the first column as we checked they are all equal
        fcast_flags = fdatesLF .∉  Ref(obs_datesLF);                    # indicate which values in store_YYlf of variable i are forecasts based on the data in dataLF_tab
        fcast_datesLF = fdatesLF[fcast_flags];                          # corresponding dates 

        data_true_flags_vec = datesLF_true .∈  Ref(fcast_datesLF)                   # flags saying which rows of the true data correspond to our forecasts (e.g. we might have 12 periods of true data but have only done forecast for 2)
        fcastDatesOverlap = fcast_datesLF[fcast_datesLF .∈  Ref(datesLF_true)]      # which forecasts overlap with our true data (e.g. we might have done 8 forecasts but have only 4 periods of true data)
        fcastDatesOverlap_BitVec = fdatesLF .∈ Ref(fcastDatesOverlap)               # flags saying which rows of the forecast correspond to our true data (e.g. we might have done 8 forecasts but have only 4 periods of true data)

        # this is the true data for our T+1 to T+n_fcst forecasts
        data_true_VecView = data_truef_mat[data_true_flags_vec,:]    # select only the variables that we want to evaluate and only the relevant time periods

        # these are the relevant forecasts
        fcast_YY =  @view store_YY_LF[fcastDatesOverlap_BitVec,locs,:]
        fcast_YY_mat = dropdims(mean(fcast_YY,dims=3),dims=3);
        fcast_errors_mat = data_true_VecView .- fcast_YY

        fcast_errors_mAd_mat[1:size(fcast_errors_mat,1),:] = dropdims(mean(fcast_errors_mat,dims=3),dims=3);   # mean forecast error across draws

        # predictive likelihood as in Carriero et al 2013 JAE
        pred_lik_mat[1:size(fcast_errors_mat,1),:] = BEAVARs.pred_lik_CCM(fcast_YY,data_true_VecView)
        
    else
        # TODO finish this for the case where the dataLF_tab is unbalanced
        # for i_var = 1:n_fvar
        #     i_var_sym = var_list_true[i_var];                               # symbol of the variable for which we will evaluate
        #     i_var_storeYY_loc = i_var_sym.==var_list;                        # location of the variable to evaluate in store_YY_LF

        #     obs_datesLF_i_var = datesLF[no_nan_flag_mat[:,i_var]];    # dates for which we have obs, everthing else is forecast
        #     fcast_flags_i_var = fdatesLF .∉  Ref(obs_datesLF_i_var);        # indicate which values in store_YYlf of variable i are forecasts based on the data in dataLF_tab
        #     fcast_datesLF_i_var = fdatesLF[fcast_flags_i_var];              # corresponding dates 

        #     data_true_flags_i_var_vec = datesLF_true .∈  Ref(fcast_datesLF_i_var)           # flags saying which rows of the true data correspond to our forecasts (e.g. we might have 12 periods of true data but have only done forecast for 2)
        #     fcastDatesOverlap_i_var = fcast_datesLF_i_var[fcast_datesLF_i_var .∈  Ref(datesLF_true)]    # which forecasts overlap with our true data (e.g. we might have done 8 forecasts but have only 4 periods of true data)
        #     fcastDatesOverlap_i_var_BitVec = fdatesLF .∈ Ref(fcastDatesOverlap_i_var)          # flags saying which rows of the forecast correspond to our true data (e.g. we might have done 8 forecasts but have only 4 periods of true data)

        #     # this is the true data for our T+1 to T+n_fcst forecasts
        #     @views data_true_i_var_VecView = data_truef_mat[data_true_flags_i_var_vec,i_var]    

        #     # these are the relevant forecasts
        #     @views fcast_YY =  store_YY_LF[fcastDatesOverlap_i_var_BitVec,i_var_storeYY_loc,:]
        #     fcast_errors_i_var = data_true_i_var_VecView .- fcast_YY


        #     # h_eval = size(fcast_YY,1);     # number of forecasts for i_var (must be less or equal to n_fcst)
        #     # for ii = 1:h_eval
        #     #     logdensKDE_3dmat[ii,i_var] =BEAVARs.mixture_log_score(vec(fcast_YY[ii,:,:]),data_true_i_var_VecView[ii])
        #     # end
        #     fcast_errors_mAd_mat[1:size(fcast_errors_i_var,1),i_var] = dropdims(mean(fcast_errors_i_var,dims=3),dims=3);   # mean forecast error across draws

        #     # predictive likelihood as in Carriero et al 2013 JAE
        #     pred_lik_mat[1:size(fcast_errors_i_var,1),i_var] = BEAVARs.pred_lik_CCM(fcast_YY,data_true_i_var_VecView)
        # end
    end
    
    # TODO this works only for the balanced case (the first one above)
    data_true_dates = datesLF_true[data_true_flags_vec];
    
    eval_vint_CPZ2023_struct = BEAVARs.eval_vint_CPZ2023(pred_lik_mat, fcast_errors_mAd_mat,fcastDatesOverlap,data_true_VecView,data_true_dates)
    # eval_vint_CPZ2023_SA_struct = BEAVARs.eval_vint_CPZ2023_SA(pred_lik_mat, fcast_errors_mAd_mat,data_true_VecView,fcast_YY_mat)
    return eval_vint_CPZ2023_struct
end




"""
    beavars_prep_eval(vint_out_dict::ThreadSafeDict{String,BEAVARs.BVARmodelOutput}, vint_dict::ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup}, dataLF_true_ftab::TimeArray)

    Prepare the forecasts for evaluation. Only the variables that are in dataLF_true_ftab will be evaluated. 
        They need to all be present.
"""
function beavars_prep_eval(vint_out_dict::ThreadSafeDict{String,BEAVARs.BVARmodelOutput}, vint_dict::ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup}, dataLF_true_ftab::TimeArray)
    eval_vint_dict = ThreadSafeDict{String,BEAVARs.BVARmodelEval}()
    ks      = collect(keys(vint_out_dict))
    n_eval  = length(ks)
    n_fcast = vint_dict[ks[1]].set_struct.n_fcst;
    varEval_list = colnames(dataLF_true_ftab);
    n_varToeval = length(varEval_list);
    fe_mat  = fill(NaN,n_fcast,n_eval,n_varToeval)
    pred_lik_mat=fill(NaN,n_fcast,n_eval,n_varToeval)
    list_keys = String[]
    list_dates = Date[]
    
    for index in ks
        println("Evaluating $index")
        eval_vint_dict[index] = BEAVARs.eval_forecast(vint_out_dict[index], vint_dict[index].data_struct, vint_dict[index].set_struct, dataLF_true_ftab)
        fe_mat[:,ks.==index,:] = eval_vint_dict[index].fcast_errors_mAd_mat
        pred_lik_mat[:,ks.==index,:] = eval_vint_dict[index].pred_lik_mat
        push!(list_keys, index)
        push!(list_dates,eval_vint_dict[index].data_true_dates[1])
    end

    sfe_mat = fe_mat.^2;     # squared forecast error matrix
    ae_mat  = abs.(fe_mat);  # absolute forecast error matrix

    sort_ind            = sortperm(list_dates);
    list_dates_sort     = list_dates[sort_ind];
    list_keys_sort      = list_keys[sort_ind];
    fe_mat_sort         = fe_mat[:,sort_ind,:];
    sfe_mat_sort        = sfe_mat[:,sort_ind,:];
    ae_mat_sort        = ae_mat[:,sort_ind,:];
    pred_lik_mat_sort   = pred_lik_mat[:,sort_ind,:]

    toEval_struct = BEAVARs.EvalForecast(fe_mat_sort,sfe_mat_sort,ae_mat_sort,pred_lik_mat_sort,list_keys_sort,list_dates_sort,varEval_list)

    return toEval_struct, eval_vint_dict
end

"""
    eval_forecasts(eval_struct; drop_dates = Date[])



"""
function eval_forecasts(eval_struct; drop_dates = Date[])
    @unpack list_dates, list_keys, fe_mat, sfe_mat, ae_mat, pred_lik_mat, var_eval_list =  eval_struct

    if typeof(drop_dates)<:Date                             # check if the date supplied is a vector of dates (multiple) or only one, then convert it to Vector
        drop_dates_vec = [drop_dates]
    elseif typeof(drop_dates)<:StepRange
        drop_dates_vec = collect(drop_dates)
    elseif typeof(drop_dates)<:Vector{Date}
        drop_dates_vec = drop_dates;
    end
    select_dates = list_dates .∉ Ref(drop_dates_vec)       # which forecast origins to keep

    if iszero(select_dates)
        error("No forecasts left to evaluate. You supplied a dates vector that lead to the dropping of all possible forecast origins.")
    end

    if any(select_dates.==0)
        println("Dropping ", list_dates[select_dates.==0], "when evaluating.")
    end

    # apl_vec = log.(dropdims(BEAVARs.nanfunc(sum, exp.(pred_lik_mat); dims=2),dims=2)); # average predictive likelihood (gemoetric mean)
    apl = dropdims(BEAVARs.nanfunc(mean, pred_lik_mat[:,select_dates,:]; dims=2),dims=2); # average predictive likelihood 

    msfe = dropdims(BEAVARs.nanfunc(mean, sfe_mat[:,select_dates,:]; dims=2),dims=2)
    mafe = dropdims(BEAVARs.nanfunc(mean, ae_mat[:,select_dates,:]; dims=2),dims=2)
    rmsfe = sqrt.(msfe);

    n_fcast = size(fe_mat,1);
    h_values = ["T+$i" for i in 1:n_fcast]
    # fe_tab = (h = h_values, RMSFE = rmsfe, MAFE = mafe, APL = apl)
    # pretty_table(fe_tab)
    fcast_metrics_struct = fcast_metrics(h_values,msfe,rmsfe,mafe,apl,var_eval_list)
    return fcast_metrics_struct
end
