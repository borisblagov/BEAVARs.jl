# ------------------------------------------------------------------
# BVAR with classical  Minnesota prior (homoscedastic fixed variance-covariance matrix) 
# as in Chan, J.C.C. (2020), Large Bayesian Vecotrautoregressions, P. Fuleky (Eds), 
# _Macroeconomic Forecasting in the Era of Big Data_, 95-125, Springer, Cham, 
# [https://doi.org/10.1007/978-3-030-31150-6](https://doi.org/10.1007/978-3-030-31150-6), 
# see also [joshuachan.org](https://joshuachan.org) and his [pdf](https://joshuachan.org/papers/large_BVAR.pdf).

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
    data_strct: A dataBVAR_TA structure with the data and metadata
"""
function makeDataSetup(::Chan2020minn_type,data_tab::TimeArray; var_list =  colnames(data_tab))
    return dataBVAR_TA(data_tab, var_list)
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
function Chan2020minn(YY,VARSetup::BVARmodelSetup,hypSetup::BVARmodelHypSetup)
    @unpack p,nburn,nsave = VARSetup
    
    Y, X, T, n, sigmaP, S_0, Σt_inv, Vβ_inv, Vβ_inv_vecView, Σ_invsp, Σt_LI, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, K_β, βminn, intercept = BEAVARs.initMinn(YY,p);

    (idx_kappa1,idx_kappa2, Vβ_vec) = BEAVARs.prior_Minn(n,p,sigmaP,hypSetup)

    Vβ_inv_vecView[:] = 1.0./Vβ_vec;                # update the diagonal of Vβ_inv
    Xsur_den[Xsur_CI] = X[X_CI];                    # update Xsur  
    mul!(XtΣ_inv_den,Xsur_den',Σ_invsp);            #  X'*( I(T) ⊗ Σ^{-1} )
    mul!(XtΣ_inv_X,XtΣ_inv_den,Xsur_den);           #  X'*( I(T) ⊗ Σ^{-1} )*X
    K_β[:,:] .= Vβ_inv .+ XtΣ_inv_X;                #  K_β = V^{-1} + X'*( I(T) ⊗ Σ^{-1} )*X
    prior_mean = Vβ_inv*βminn;                      #  V^-1 * βminn 
    mul!(prior_mean,XtΣ_inv_den, vec(Y'),1.0,1.0);  # (V^-1_Minn * beta_Minn) + X' ( I(T) ⊗ Σ-1 ) y
    cholK_β = cholesky(Hermitian(K_β));             # Cholesky factor
    beta_hat = ldiv!(cholK_β.U,ldiv!(cholK_β.L,prior_mean));    # C'\(C*(V^-1_Minn * beta_Minn + X' ( I(T) ⊗ Σ-1 ) y)
    

    ndraws = nsave+nburn;
    store_β=zeros(n^2*p+n,nsave)
    for ii = 1:ndraws 
        beta = beta_hat + ldiv!(cholK_β.U,randn(k*n,)); # draw for β
        if ii>nburn
            store_β[:,ii-nburn] = beta;
        end
    end

    store_Σ = repeat(vec(S_0),1,nsave);
    return store_β, store_Σ
end



# types for output export
@with_kw struct VAROutput_Chan2020minn <: BVARmodelOutput
    store_β::Array{}      # 
    store_Σ::Array{}      # 
    YY::Array{}             #
end



#------------------------------
# Forecasting block
#------------------------------
function makeForecastOutput(::Chan2020minn_type,Yfor3D)
    return dataBVAR_TA(data_tab, var_list)
end

@doc raw"""
    Yfor3D = BEAVARs.forecast(VAROutput::VAROutput_Chan2020minn,VARSetup)

Generates forecasts from the Chan2020minn model output

# Arguments
    VAROutput: A VAROutput_Chan2020minn structure with the model output
    VARSetup:  A BVARmodelSetup structure with the model setup  
# Returns
    Yfor3D:    A 3D array with the forecasts. Dimensions are (p+n_fcst) x n x nsave

# Description

The function generates forecasts from the Chan2020minn model output.
"""
function forecast(VAROutput::VAROutput_Chan2020minn,VARSetup::BVARmodelSetup,data_strct::BVARmodelDataSetup)
    @unpack store_β, store_Σ, YY = VAROutput
    @unpack n_fcst,p,nsave = VARSetup
    n = size(YY,2);

    Yfor3D    = fill(NaN,(p+n_fcst,n,nsave))
    Yfor3D[1:p,:,:] .= @views YY[end-p+1:end,:];
    
    for i_draw = 1:nsave
        Yfor = @views Yfor3D[:,:,i_draw];
        A_draw = @views reshape(store_β[:,i_draw],n*p+1,n);
        Σ_draw = @views reshape(store_Σ[:,i_draw],n,n);
                
        for i_for = 1:n_fcst
            tclass = @views vec(reverse(Yfor[1+i_for-1:p+i_for-1,:],dims=1)')
            tclass = [1;tclass];
            Yfor[p+i_for,:]=tclass'*A_draw  .+ (cholesky(Σ_draw).U*randn(n,1))';    
        end
    end
    fcast_strct = BEAVARs.VARForecast(Yfor3D,data_strct.data_tab,data_strct.var_list,n_fcst)
    return fcast_strct

end # end function fcastChan2020minn()


function forecast(VAROutput::VAROutput_Chan2020minn,VARSetup,trueYY)
    @unpack store_β, store_Σ, YY = VAROutput
    @unpack n_fcst,p,nsave = VARSetup
    n = size(YY,2);

    Yfor3D    = fill(NaN,(p+n_fcst,n,nsave))
    Yfor3D[1:p,:,:] .= @views YY[end-p+1:end,:];
    
    for i_draw = 1:nsave
        Yfor = @views Yfor3D[:,:,i_draw];
        A_draw = @views reshape(store_β[:,i_draw],n*p+1,n);
        Σ_draw = @views reshape(store_Σ[:,i_draw],n,n);
                
        for i_for = 1:n_fcst
            tclass = @views vec(reverse(Yfor[1+i_for-1:p+i_for-1,:],dims=1)')
            tclass = [1;tclass];
            Yfor[p+i_for,:]=tclass'*A_draw  .+ (cholesky(Σ_draw).U*randn(n,1))';    
        end
    end

    MSE_t_mat = dropdims(mean((Yfor3D.-trueYY).^2,dims=3),dims=3)
    return Yfor3D, MSE_t_mat

end # end function fcastChan2020minn()

# depreciated after moving to the syntax beavar(strcts)
#------------------------------
# dispatchModel block
#------------------------------
# function dispatchModel(::Chan2020minn_type,YY_tup, hyper_str, p,n_burn,n_save,n_irf,n_fcst)
#     println("Hello Minn")
#     intercept = 1;
#     if isa(YY_tup[1],Array{})
#         YY = YY_tup[1];
#     elseif isa(YY_tup[1],TimeArray{})
#         YY_TA = YY_tup[1];
#         YY = values(YY_TA)
#         varList = colnames(YY_TA)
#     end
#     set_strct = VARSetup(p,n_save,n_burn,n_irf,n_fcst,intercept);
#     store_β, store_Σ = Chan2020minn(YY,set_strct,hyper_str);
#     out_strct = VAROutput_Chan2020minn(store_β,store_Σ,YY)
#     return out_strct, set_strct
# end


