# ------------------------------------------------------------------
# Minnesota prior with independent Normal inverse Wishart prior

function makeHypSetup(::Chan2020iniw_type)
    return hypChan2020()
end


@with_kw struct VAROutput_Chan2020iniw{T <: AbstractFloat} <: BVARmodelOutput
    store_β::Matrix{T}      # 
    store_Σ::Array{T,3}      # 
    YY::Matrix{T}           #
    fdatesLF:: Vector{DateTime}
end

@doc raw"""
    makeDataSetup(::Chan2020iniw_type,data_tab::TimeArray; var_list =  colnames(data_tab))

Generate a dataset strcture for use with the single-frequency models.
"""
function makeDataSetup(::Chan2020iniw_type,data_tab::TimeArray; var_list =  colnames(data_tab))
    return data_BVAR(data_tab,values(data_tab), var_list)
end



"""
    Chan2020iniw(YY,VARSetup::BVARmodelSetup,hypSetup::BVARmodelHypSetup)

    Implements BVAR with Independent Normal Inverse Wishart (iniw) prior following Chan (2020)

"""
function Chan2020iniw(YY,VARSetup::BVARmodelSetup,hypSetup::BVARmodelHypSetup)
    @unpack p,n_burn,n_save, prior_RW = VARSetup
    n_draws  = n_save+n_burn;

    Y, X, T, n, sigmaP, S_0, Σt_inv, Vβminn_inv, Vβminn_inv_elview, Σ_invsp, Σt_LI, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, K_β, beta, intercept, betOLS = BEAVARs.init_Minn(YY,p);

    (idx_kappa1,idx_kappa2, Vβminn, βMinn) = prior_Minn(n,p,sigmaP,hypSetup,prior_RW)

    Vβminn_inv_elview[:] = 1.0./Vβminn;             # update the diagonal of Vβminn_inv
    Xsur_den[Xsur_CI] = X[X_CI];                    # update Xsur  

    # allocate output for saving
    store_β = zeros(n^2*p+n,n_save);
    store_Σt = zeros(n,n,n_save);
    for ii = 1:n_draws 
        beta = BEAVARs.Chan2020_drawβ(Σ_invsp,Xsur_den,XtΣ_inv_den,XtΣ_inv_X,Vβminn_inv,βMinn,K_β,Y,n,k);
        Σt, Σt_inv = Chan2020_drawΣt(Y,Xsur_den,beta,n,T,S_0,hypSetup.nu0);

        Σ_invsp.nzval[:] = Σt_inv[Σt_LI];               # update ( I(T) ⊗ Σ^{-1} )

        if ii>n_burn
            store_β[:,ii-n_burn] = beta;
            store_Σt[:,:,ii-n_burn] = Σt;
        end
    end

    return store_β, store_Σt
end



#-------------------------------------
# The Den of the beavar
#-------------------------------------

function beavar(::Chan2020iniw_type, set_struct, hyp_str, data_struct)
    println("Hello Independent Normal Inverse Wishart")
    @unpack data_tab, data_mat, var_list = data_struct;
    freqL_date = BEAVARs.get_data_freq(data_tab);
    datesLF = timestamp(data_tab);
    datesLF_fcast = collect(datesLF[end]+freqL_date:freqL_date:datesLF[end]+freqL_date*(set_struct.n_fcst));
    fdatesLF = [datesLF;datesLF_fcast];
    YY = values(data_struct.data_tab);
    store_β, store_Σ = Chan2020iniw(YY,set_struct,hyp_str);
    out_struct = VAROutput_Chan2020iniw(store_β,store_Σ,YY,fdatesLF)
    return out_struct
end




function forecast(VAROutput::VAROutput_Chan2020iniw,VARSetup::BVARmodelSetup)
    @unpack store_β, store_Σ, YY = VAROutput
    @unpack n_fcst,p,n_save = VARSetup
    n = size(YY,2);

    @unpack store_β, store_Σ, YY = VAROutput
    @unpack n_fcst,p,n_save = VARSetup
    n = size(YY,2);

    Yfor3D    = fill(NaN,(p+n_fcst,n,n_save))
    Yfor3D[1:p,:,:] .= @views YY[end-p+1:end,:];
    
    for i_draw = 1:n_save
        Yfor = @views Yfor3D[:,:,i_draw];
        A_draw = @views reshape(store_β[:,i_draw],n*p+1,n);
        Σ_draw = @views store_Σ[:,:,i_draw];
        # Σ_draw =  dropdims(median(store_Σ,dims=3),dims=3);
                
        for i_for = 1:n_fcst
            tclass = @views vec(reverse(Yfor[1+i_for-1:p+i_for-1,:],dims=1)')
            tclass = [1;tclass];
            Yfor[p+i_for,:]=tclass'*A_draw  .+ (cholesky(Hermitian(Σ_draw)).U*randn(n,1))';    
        end
    end
    return Yfor3D


end # end function forecast Chan2020iniw()



function forecast(VAROutput::VAROutput_Chan2020iniw,VARSetup::BVARmodelSetup,data_struct::BVARmodelDataSetup,data_true_ftab)
    @unpack store_β, store_Σ, YY, fdatesLF = VAROutput
    @unpack n_fcst,p,n_save = VARSetup
    n = size(YY,2);

    Yfor3D    = fill(NaN,(p+n_fcst,n,n_save))
    Yfor3D[1:p,:,:] .= @views YY[end-p+1:end,:];

    data_truef_mat = values(data_true_ftab)
    data_true_flags_vec = timestamp(data_true_ftab) .∈  Ref(fdatesLF[end-n_fcst+1:end]) # flags saying which rows of the true data correspond to our forecasts (e.g. we might have 12 periods of tre data but have only done forecast for 2)
    forecast_flags_vec = fdatesLF[end-n_fcst+1:end] .∈  Ref(timestamp(data_true_ftab)) # flags saying which rows of the forecast correspond to our true data (e.g. we might have done 8 forecasts but have only 4 periods of true data)

    @views data_true_mat = data_truef_mat[data_true_flags_vec,:]    # this is the true data for our T+1 to T+n_fcst forecasts
    
    logdens_3dmat = fill(NaN,(n_fcst,n,n_save))
    joint_logdens_3dmat = fill(NaN,(n_fcst,1,n_save))
    
    for i_draw = 1:n_save
        Yfor = @views Yfor3D[:,:,i_draw];
        YforExp = @views Yfor3D[p+1:end,:,i_draw];
        A_draw = @views reshape(store_β[:,i_draw],n*p+1,n);
        Σ_draw = @views store_Σ[:,:,i_draw];
        cholSig = cholesky(Hermitian(Σ_draw));
        sqSig = cholSig.L;
        Sig_vec = diag(Σ_draw)
                
        for i_for = 1:n_fcst
            tclass = [1.0; @views vec(reverse(Yfor[1+i_for-1:p+i_for-1,:],dims=1)')];
            # tclass = [1.0;tclass];
            Yfor[p+i_for,:]=tclass'*A_draw  .+ (cholSig.U*randn(n,1))';    
            YforExp[i_for,:]=tclass'*A_draw;    
        end

        YY_fcast = @views Yfor3D[p+1:end,:,i_draw]
        fcast_errors_i = @views data_truef_mat[data_true_flags_vec,:] - YY_fcast[forecast_flags_vec,:];       # fcast error for draw i
        E_fcast_errors_i = @views data_truef_mat[data_true_flags_vec,:] - YforExp[forecast_flags_vec,:];       # fcast error compared to the expectation

        res = [sqSig\r for r in eachrow(E_fcast_errors_i)]                                # adjusted for the model variance 
        joint_logdens_3dmat[forecast_flags_vec,:,i_draw] = (-n/2).*log(2*π) .-sum(log.(diag(sqSig))) .- 0.5.*[r'*r for r in eachrow(res)]
        logdens_vv = [-.5*log.(2*π.*Sig_vec) - 0.5.*(r.^2)./Sig_vec for r in eachrow(E_fcast_errors_i)]
        logdens_3dmat[forecast_flags_vec,:,i_draw] = reduce(hcat,logdens_vv)';

    end
    
    lpl_mat = dropdims(log.(BEAVARs.nanfunc(mean,exp.(logdens_3dmat.-BEAVARs.nanfunc(maximum,exp.(logdens_3dmat),dims=3)),dims=3)),dims=3); # log predictive likelihood for this vintage. Chan does a trick with the maximum that I don't know what it does
    
    # lpl_joint_mat = dropdims(log.(nanfunc(mean,exp.(joint_logdens_3dmat),dims=3)),dims=3)
    lpl_joint_mat = dropdims(log.(BEAVARs.nanfunc(mean,exp.(joint_logdens_3dmat.-BEAVARs.nanfunc(maximum,joint_logdens_3dmat,dims=3)),dims=3)),dims=3)

    Yfor3d = Yfor3D[p+1:end,:,:];
    fcast_ϵ_mat = dropdims(mean(-(Yfor3d[forecast_flags_vec,:,:] .-  data_truef_mat[data_true_flags_vec,:]),dims=3),dims=3);    # forecast errors

    YY_low2, YY_low1, YY_low, YY_med, YY_hih, YY_hih1, YY_hih2 = BEAVARs.get_imp_percentiles(Yfor3d); # TODO add the to the output
    fcast_struct = BEAVARs.VARForecast(Yfor3d,data_struct.data_tab,data_struct.var_list,n_fcst)
    return fcast_struct, fcast_ϵ_mat, lpl_mat, lpl_joint_mat


end # end function forecast Chan2020iniw()
