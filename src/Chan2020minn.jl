# ------------------------------------------------------------------
# BVAR with classical  Minnesota prior (homoscedastic fixed variance-covariance matrix) 
# as in Chan, J.C.C. (2020), Large Bayesian Vecotrautoregressions, P. Fuleky (Eds), 
# _Macroeconomic Forecasting in the Era of Big Data_, 95-125, Springer, Cham, 
# [https://doi.org/10.1007/978-3-030-31150-6](https://doi.org/10.1007/978-3-030-31150-6), 
# see also [joshuachan.org](https://joshuachan.org) and his [pdf](https://joshuachan.org/papers/large_BVAR.pdf).



# types for output export
@with_kw struct VAROutput_Chan2020minn{T <: AbstractFloat, N}  <: BVARmodelOutput
    store_β::Array{T,N}      # 
    store_Σ::Array{T,N}      # 
    YY::Array{T,N}            #
    fdatesLF:: Vector{DateTime}
    store_YY::Array{T,3}            #
end

function makeHypSetup(::Chan2020minn_type)
    return hypChan2020()
end



@doc raw"""
    makeDataSetup(::Chan2020minn_type,data_tab::TimeArray; var_list =  colnames(data_tab))

Generate a dataset strcture for use with the single-frequency models
    
# Arguments
    model_type: The custom model type (not a string)
    data_tab:   TimeArray with the data_de
    var_list:   A symbol list with the variable names. Will be used for oredering the variables. Uses by default the names from data_tab if not supplied. note that Symbol lists have a particular synthax.
# Returns
    data_struct: A dataBVAR_TA structure with the data and metadata
"""
function makeDataSetup(::Chan2020minn_type,data_tab::TimeArray; var_list =  colnames(data_tab))
    return data_BVAR(data_tab,values(data_tab), var_list)
end


function makeDataSetup(::Chan2020minn_type,dataHF_tab::TimeArray, dataLF_tab::TimeArray; var_list =  [colnames(dataLF_tab); colnames(dataHF_tab)])
    data_tab = merge(dataLF_tab, dataHF_tab)
    return data_BVAR(data_tab,values(data_tab), var_list)
end


@doc raw"""
    BEAVARs.Chan2020minn(YY,VARSetup,hypSetup)

Implements the classic homoscedastic Minnesota prior with a SUR form following Chan (2020)

# Arguments
    YY:         A T x n matrix with the data
    VARSetup:   A BVARmodelSetup structure with the model setup
    hypSetup:   A BVARmodelHypSetup structure with the hyperparameters
# Returns
    store_β:    A matrix with the posterior draws of the VAR coefficients
    store_Σ:    A matrix with the posterior draws of the variance-covariance matrix

# Description
The function implements the homoscedastic Minnesota prior with a SUR form as in Chan (2020).

# Reference
Chan, J.C.C. (2020), Large Bayesian Vecotrautoregressions, P. Fuleky (Eds), _Macroeconomic Forecasting in the Era of Big Data_, 95-125, Springer, Cham, https://doi.org/10.1007/978-3-030-31150-6 and https://joshuachan.org/papers/large_BVAR.pdf.
"""
function Chan2020minn(YY::Array{Tp},VARSetup::BVARmodelSetup,hypSetup::BVARmodelHypSetup) where Tp <: AbstractFloat
    @unpack p,n_burn,n_save,prior_RW = VARSetup
    
    Y, X, T, n, sigmaP, S_0, Σt_inv, Vβ_inv, Vβ_inv_vecView, Σ_invsp, Σt_LI, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, K_β, beta, intercept, betOLS = BEAVARs.init_Minn(YY,p);

    (idx_kappa1,idx_kappa2, Vβ_vec, βMinn) = BEAVARs.prior_Minn(n,p,sigmaP,hypSetup,prior_RW)

    Vβ_inv_vecView[:] = 1.0./Vβ_vec;                # update the diagonal of Vβ_inv
    Xsur_den[Xsur_CI] = X[X_CI];                    # update Xsur  
    mul!(XtΣ_inv_den,Xsur_den',Σ_invsp);            #  X'*( I(T) ⊗ Σ^{-1} )
    mul!(XtΣ_inv_X,XtΣ_inv_den,Xsur_den);           #  X'*( I(T) ⊗ Σ^{-1} )*X
    K_β[:,:] .= Vβ_inv .+ XtΣ_inv_X;                #  K_β = V^{-1} + X'*( I(T) ⊗ Σ^{-1} )*X
    prior_ = Vβ_inv*βMinn;                          #  V^-1 * βMinn 
    # println(prior_)
    mul!(prior_,XtΣ_inv_den, vec(Y'),1.0,1.0);      # (V^-1_Minn * beta_Minn) + X' ( I(T) ⊗ Σ-1 ) y
    # println(prior_);
    cholK_β = cholesky(Hermitian(K_β));             # Cholesky factor
    beta_hat = ldiv!(cholK_β.U,ldiv!(cholK_β.L,prior_));    # C'\(C*(V^-1_Minn * beta_Minn + X' ( I(T) ⊗ Σ-1 ) y)
    

    ndraws = n_save+n_burn;
    store_β=zeros(n^2*p+n,n_save);
    for ii = 1:ndraws 
        beta = beta_hat + ldiv!(cholK_β.U,randn(k*n,)); # draw for β
        if ii>n_burn
            store_β[:,ii-n_burn] = beta;
        end
    end

    store_Σ = repeat(vec(S_0),1,n_save);
    return store_β, store_Σ
end

#-----------------------
# The den of the beavar
#-----------------------
function beavar(::Chan2020minn_type, set_struct, hyp_str, data_struct)
    println("Hello Minn")
    @unpack data_tab, data_mat, var_list = data_struct;
    @unpack n_fcst, n_save = set_struct;
    freqL_date = BEAVARs.get_data_freq(data_tab);
    datesLF = timestamp(data_tab);
    datesLF_fcast = collect(datesLF[end]+freqL_date:freqL_date:datesLF[end]+freqL_date*(n_fcst));
    fdatesLF = [datesLF;datesLF_fcast];
    YY = data_mat;
    T,n = size(YY);
    store_YY = fill(NaN,(T+n_fcst,n,n_save))
    store_β, store_Σ = Chan2020minn(YY,set_struct,hyp_str);
    out_struct = VAROutput_Chan2020minn(store_β,store_Σ,YY,fdatesLF,store_YY);
    fcast_struct = forecast(out_struct, set_struct, data_struct);
    out_struct.store_YY[:,:,:] =  @view fcast_struct.Yfor3d[:,:,:];
    return out_struct 
end



#------------------------------
# Forecasting block
#------------------------------
function makeForecastOutput(::Chan2020minn_type,Yfor3D)
    return dataBVAR_TA(data_tab, var_list)
end

@doc raw"""
    Yfor3D = BEAVARs.forecast(VAROutput::VAROutput_Chan2020minn,,VARSetup::BVARmodelSetup,data_struct::BVARmodelDataSetup)

Generates forecasts from the Chan2020minn model output

# Arguments
    VAROutput: A VAROutput_Chan2020minn structure with the model output
    VARSetup:  A BVARmodelSetup structure with the model setup  
# Returns
    fcast_struct:    The forecast structure

# Description

The function generates forecasts from the Chan2020minn model output.
"""
function forecast(VAROutput::VAROutput_Chan2020minn,VARSetup::BVARmodelSetup,data_struct::BVARmodelDataSetup)
    @unpack store_β, store_Σ, YY = VAROutput
    @unpack n_fcst, p, n_save = VARSetup
    T, n = size(YY);

    Yfor_3dmat    = fill(NaN,(T+n_fcst,n,n_save))
    Yfor_3dmat[1:T,:,:] .= @views YY[1:T,:];
    
    for i_draw = 1:n_save
        Yfor = @views Yfor_3dmat[T-p+1:end,:,i_draw];
        A_draw = @views reshape(store_β[:,i_draw],n*p+1,n);
        Σ_draw = @views reshape(store_Σ[:,i_draw],n,n);
                
        for i_for = 1:n_fcst
            tclass = @views vec(reverse(Yfor[1+i_for-1:p+i_for-1,:],dims=1)')
            tclass = [1.0;tclass];
            Yfor_3dmat[T+i_for,:,i_draw]=tclass'*A_draw  .+ (cholesky(Σ_draw).U*randn(n,1))';    
        end
    end

    # Yfor3d = Yfor3D[p+1:end,:,:];

    YY_low2, YY_low1, YY_low, YY_med, YY_hih, YY_hih1, YY_hih2 = BEAVARs.get_imp_percentiles(Yfor_3dmat); # TODO add the to the output
 
    fcast_struct = BEAVARs.VARForecast(Yfor_3dmat,data_struct.data_tab,data_struct.var_list,n_fcst)
    return fcast_struct

end # end function fcastChan2020minn()



"""
    forecast(VAROutput::VAROutput_Chan2020minn,VARSetup::BVARmodelSetup,data_struct::BVARmodelDataSetup,data_true_ftab)

    Calculates forecast for the classic Minnesota model given and compares it with the true data provided in data_true_ftab.


"""
function forecast(VAROutput::VAROutput_Chan2020minn,VARSetup::BVARmodelSetup,data_struct::BVARmodelDataSetup,data_true_ftab::TimeArray{T,N,D,A}) where {T <: AbstractFloat, N, D, A <: AbstractArray{T, N}}
    @unpack store_β, store_Σ, YY, fdatesLF = VAROutput
    @unpack n_fcst, p, n_save = VARSetup
    n = size(YY,2);

    Yfor_3dmat    = fill(NaN,(p+n_fcst,n,n_save))
    Yfor_3dmat[1:p,:,:] .= @views YY[end-p+1:end,:];
    
    data_truef_mat = values(data_true_ftab)
    data_true_flags_vec = timestamp(data_true_ftab) .∈  Ref(fdatesLF[end-n_fcst+1:end]) # flags saying which rows of the true data correspond to our forecasts (e.g. we might have 12 periods of tre data but have only done forecast for 2)
    forecast_overlap_vec = fdatesLF[end-n_fcst+1:end] .∈  Ref(timestamp(data_true_ftab)) # flags saying which rows of the forecast correspond to our true data (e.g. we might have done 8 forecasts but have only 4 periods of true data)

    @views data_true_mat = data_truef_mat[data_true_flags_vec,:]    # this is the true data for our T+1 to T+n_fcst forecasts
    
    logdens_3dmat = fill(NaN,(n_fcst,n,n_save))
    joint_logdens_3dmat = fill(NaN,(n_fcst,1,n_save))


    for i_draw = 1:n_save
        Yfor = @views Yfor_3dmat[:,:,i_draw];
        YforExp = @views Yfor_3dmat[p+1:end,:,i_draw];
        A_draw = @views reshape(store_β[:,i_draw],n*p+1,n);
        Σ_draw = @views reshape(store_Σ[:,i_draw],n,n);
        sqSig = sqrt.(Σ_draw);
        Sig_vec = diag(Σ_draw)

                
        for i_for = 1:n_fcst
            tclass = [1.0; @views vec(reverse(Yfor[1+i_for-1:p+i_for-1,:],dims=1)')];
            Yfor[p+i_for,:]=tclass'*A_draw  .+ (cholesky(Σ_draw).U*randn(n,1))';   
            YforExp[i_for,:]=tclass'*A_draw;      
        end

        E_fcast_errors_i = @views data_truef_mat[data_true_flags_vec,:] - YforExp[forecast_overlap_vec,:];       # fcast error compared to the expectation

        res = [sqSig\r for r in eachrow(E_fcast_errors_i)]                                # adjusted for the model variance 
        joint_logdens_3dmat[forecast_overlap_vec,:,i_draw] = (-n/2).*log(2*π) .-sum(log.(diag(sqSig))) .- 0.5.*[r'*r for r in eachrow(res)]
        logdens_vv = [-.5*log.(2*π.*Sig_vec) - 0.5.*(r.^2)./Sig_vec for r in eachrow(E_fcast_errors_i)]
        logdens_3dmat[forecast_overlap_vec,:,i_draw] = reduce(hcat,logdens_vv)';

    end

    lpl_mat = dropdims(log.(BEAVARs.nanfunc(mean,exp.(logdens_3dmat.-BEAVARs.nanfunc(maximum,exp.(logdens_3dmat),dims=3)),dims=3)),dims=3); # log predictive likelihood for this vintage. Chan does a trick with the maximum that I don't know what it does
    
    max_dens = BEAVARs.nanfunc(maximum,joint_logdens_3dmat,dims=3);
    lpl_joint_mat = dropdims(log.(BEAVARs.nanfunc(mean,exp.(joint_logdens_3dmat.-max_dens),dims=3))+max_dens,dims=3);

    Yfor3d = Yfor_3dmat[p+1:end,:,:];
    fcast_ϵ_mat = dropdims(mean(-(Yfor3d[forecast_overlap_vec,:,:] .-  data_truef_mat[data_true_flags_vec,:]),dims=3),dims=3);    # forecast errors

    YY_low2, YY_low1, YY_low, YY_med, YY_hih, YY_hih1, YY_hih2 = BEAVARs.get_imp_percentiles(Yfor3d); # TODO add the to the output
 
    fcast_struct = BEAVARs.VARForecast(Yfor3d,data_struct.data_tab,data_struct.var_list,n_fcst)
    return fcast_struct, fcast_ϵ_mat, lpl_mat, lpl_joint_mat

end # end function fcastChan2020minn()



function eval_forecast(out_struct::VAROutput_Chan2020minn,data_struct::BVARmodelDataSetup,set_struct::BVARmodelSetup,dataLF_true_ftab::TimeArray{T,N,D,A}) where {T <: AbstractFloat, N, D, A <: AbstractArray{T, N}}
    @unpack store_YY = out_struct;  
    @unpack data_tab, var_list = data_struct
    @unpack n_fcst, p, n_save, prior_RW = set_struct

    store_YY_LF = store_YY;
    dataLF_tab = data_tab;
    # check if we are working in levels to transform the true data to growth rates for evaluation
    if prior_RW == 1
        dataLF_tab = percentchange(dataLF_tab);
        dataLF_true_ftab = percentchange(dataLF_true_ftab);
        fdatesLF = out_struct.fdatesLF[2:end];              # we lose the first date when we take growth rates, so we need to adjust the forecast dates accordingly
        store_YY_LF = out_struct.store_YY_LF[2:end,:,:]./out_struct.store_YY_LF[1:end-1,:,:].-1;
    else
        fdatesLF = out_struct.fdatesLF;
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


    
    locs = [findfirst(==(s), var_list) for s in var_list_true]
    obs_datesLF = datesLF[no_nan_flag_mat[:,1]];                    # dates for which we have obs, everthing else is forecast, we may simply take the first column as we checked they are all equal
    fcast_flags = fdatesLF .∉  Ref(obs_datesLF);                    # indicate which values in store_YYlf of variable i are forecasts based on the data in dataLF_tab
    fcast_datesLF = fdatesLF[fcast_flags];                          # corresponding dates 

    data_true_flags_vec = datesLF_true .∈  Ref(fcast_datesLF)                   # flags saying which rows of the true data correspond to our forecasts (e.g. we might have 12 periods of true data but have only done forecast for 2)
    fcastDatesOverlap = fcast_datesLF[fcast_datesLF .∈  Ref(datesLF_true)]      # which forecasts overlap with our true data (e.g. we might have done 8 forecasts but have only 4 periods of true data)
    fcastDatesOverlap_BitVec = fdatesLF .∈ Ref(fcastDatesOverlap)               # flags saying which rows of the forecast correspond to our true data (e.g. we might have done 8 forecasts but have only 4 periods of true data)

    # this is the true data for our T+1 to T+n_fcst forecasts
    data_true_VecView = data_truef_mat[data_true_flags_vec,locs]    # select only the variables that we want to evaluate and only the relevant time periods

    # these are the relevant forecasts
    fcast_YY =  store_YY_LF[fcastDatesOverlap_BitVec,locs,:]
    fcast_YY_mat = dropdims(mean(fcast_YY,dims=3),dims=3);
    fcast_errors_mat = data_true_VecView .- fcast_YY

    fcast_errors_mAd_mat[1:size(fcast_errors_mat,1),:] = dropdims(mean(fcast_errors_mat,dims=3),dims=3);   # mean forecast error across draws

    # predictive likelihood as in Carriero et al 2013 JAE
    pred_lik_mat[1:size(fcast_errors_mat,1),:] = BEAVARs.pred_lik_CCM(fcast_YY,data_true_VecView)
    
    
       
    
    # TODO this works only for the balanced case (the first one above)
    data_true_dates = datesLF_true[data_true_flags_vec];
    
    eval_vint_Chan2020minn_struct = BEAVARs.eval_vint_CPZ2023(pred_lik_mat, fcast_errors_mAd_mat,fcastDatesOverlap,data_true_VecView,data_true_dates)
    return eval_vint_Chan2020minn_struct
    # return pred_lik_mat, fcast_errors_mAd_mat,fcastDatesOverlap,data_true_VecView,data_true_dates
end