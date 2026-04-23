function makeHypSetup(::Blagov2025_type)
    return hypChan2020()
end



# Structure for the datasets and the frequency mix
@doc raw"""
    dataBlagov2025(data_HF::TimeArray,data_LF::TimeArray,aggMix::Int,var_list::Array{Symbol,1})

generate a dataset strcture for use with Blagov2025 model

# Arguments
    dataHF_tab: TimeArray with your high-frequency variables (monthly or quarterly, respectively)
    dataLF_tab: TimeArray with your low-frequency variables (quarterly or yearly, respectively)
    aggMix:     0 for data in growth rates, 1 for log-levels. Determines the weights how high freq. variables fit with low-frequency ones. Will use averages for log-levels or Mariano and Murasawa (2010) weights for growth rates 
    var_list:   the variable order. Note that the functions that call these variables allow this to be optional.

See also `makeDataSetup`.
"""
@with_kw struct dataBlagov2025 <: BVARmodelDataSetup
    dataHF_tab::TimeArray                                       # data for the high-frequency variables
    dataLF_tab::TimeArray                                       # data for the low-frequency variables
    aggMix::Int                                                 # 0: growth rates, 1: log-levels. indicator for the aggregate weights in the inter-temporal aggregation
    var_list::Array{Symbol,1}                                   # Symbol vector with the variable names, will be used for ordering
end

@doc raw"""
    Prepare the structure containg the data for the mixed-frequency VAR. Uses Time Arrays from the TimeSeries package
"""
function makeDataSetup(::Blagov2025_type,dataHF_tab::TimeArray, dataLF_tab::TimeArray, aggMix::Int; var_list =  [colnames(dataHF_tab); colnames(dataLF_tab)])
    return dataBlagov2025(dataHF_tab, dataLF_tab, aggMix, var_list)
end



@doc raw"""
     BEAVARs.Blagov2025(dataHF_tab,dataLF_tab,varList,varSetup,hypSetup,trans)

Implements the mixed-frequency BVAR model as in Blagov et al. (2025)

# Arguments
    dataHF_tab: TimeArray with your high-frequency variables (monthly or quarterly, respectively)
    dataLF_tab: TimeArray with your low-frequency variables (quarterly or yearly, respectively)
    varList:    A symbol list with the variable names. Will be used for oredering the variables.
    varSetup:   A BVARmodelSetup structure with the model setup
    hypSetup:   A BVARmodelHypSetup structure with the hyperparameters

# Returns
    store_YY:      A 3D array with the posterior draws of the data matrix
    store_β:       A matrix with the posterior draws of the VAR coefficients
    store_Σt_inv:  A 3D array with the posterior draws of the variance-covariance matrix inverse
    M_zsp:         The mapping matrix from high-frequency to low-frequency
    z_vec:         A vector indicating the low-frequency observations in the high-frequency time series
    Sm_bit:        A binary selection matrix for the missing values in YY
    store_Σt:      A 3D array with the posterior draws of the variance-covariance matrix
    store_h:       A matrix with the posterior draws of the stochastic volatility log-variances
    store_s2_h:    A matrix with the posterior draws of the stochastic volatility variances
    store_ρ:       A vector with the posterior draws of the AR(1) coefficient of the stochastic volatility process
    store_σ_h2:    A vector with the posterior draws of the innovation variance of the stochastic volatility process
    store_eh:      A matrix with the posterior draws of the stochastic volatility innovations

# Description
The function implements the mixed-frequency BVAR model as in Blagov et al. (2025).
# Reference
Blagov, S., Giannone, D., Lenza, M., Modugno, M. (2025), Mixed-Frequency Bayesian VARs with Stochastic Volatility: Methodology and Macroeconomic Applications, Journal of Econometrics, forthcoming.
"""
function Blagov2025(dataHF_tab,dataLF_tab,varList,varSetup,hypSetup,trans)
    @unpack ρ, σ_h2, v_h0, S_h0, ρ_0, V_ρ = hypSetup
    @unpack p, nsave, nburn, const_loc = varSetup
    ndraws = nsave+nburn;
    # nmdraws = 10;               # given a draw from the parameters to draw multiple time from the distribution of the missing data for better confidence intervals

    fdataHF_tab, z_tab, freq_mix_tp, datesHF, varNamesLF, fvarNames = BEAVARs.CPZ_prep_TimeArrays(dataLF_tab,dataHF_tab,varList,trans)

    YYwNA = values(fdataHF_tab);
    YY = deepcopy(YYwNA);
    Tf,n = size(YY);
    
    B_draw, structB_draw, Σt_inv, b0 = BEAVARs.initParamMatrices(n,p,const_loc) 
    
    YYt, Y0, longyo, nm, H_B, H_B_CI, strctBdraw_LI, Σ_invsp, Σt_LI, Σp_invsp, Σpt_ind, Xb, cB, cB_b0_LI, Smsp, Sosp, Sm_bit, Gm, Go, GΣ, Kym = BEAVARs.CPZ_initMatrices(YY,structB_draw,b0,Σt_inv,p);
    
    M_zsp, z_vec, T_z, MOiM, MOiz = BEAVARs.CPZ_makeM_inter(z_tab,YYt,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,nm,Tf);
    
    
    # YY has missing values so we need to draw them once to be able to initialize matrices and prior values
    cB,H_B,Σ_invsp  = BEAVARs.Blagov2025_updCPZ!(cB,H_B,Σ_invsp,B_draw,structB_draw,Σt_inv,Y0,cB_b0_LI,p,n,H_B_CI,strctBdraw_LI,Σt_LI);
    YYt             = BEAVARs.Blagov2025_draw_wz!(YYt,longyo,cB,Σ_invsp,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym);
    
    # BEAVARs.CPZ_draw_wz!(YYt,longyo,Y0,cB,B_draw,structB_draw,strctBdraw_LI,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,H_B_CI,nmdraws);
    
    
    # Initialize matrices for updating the parameter draws from CPZ_iniv  
    # ------------------------------------
    Y, X, T, k, sigmaP, S_0, Σ, A_0, V_Ainv, v_0, H_ρ,h,eh,Ωinv, dg_ind_Ωinv, VAinvDA0, AVAinvA, intercept   = BEAVARs.Blagov2025_initcsv(YY,p,hypSetup);
    
    (deltaP, sigmaP, mu_prior)  = trainPriors(YY,p);                         # do OLS to initialize priors
    # for updating the priors
    updP_vec = sum(Sm_bit,dims=2).>size(Sm_bit,2)*0.25;

    # prepare matrices for storage
       
    store_YY    = zeros(Tf,n,nsave);
    store_β = zeros(k*n,nsave);
    store_h = zeros(T,nsave);
    store_Σt = zeros(n,n,nsave);
    store_Σt_inv= zeros(n,n,nsave);
    store_s2_h = zeros(T,nsave);
    store_ρ = zeros(nsave,);
    store_σ_h2 = zeros(nsave,); 
    store_eh = zeros(T,nsave);
    

    @showprogress for ii in 1:ndraws

        Y, X = mlagL!(YY,Y,X,p,n);
        A, cholΣU, Σ, s2_h, U, Ωinv = BEAVARs.Chan2020_drawA(Y,X,n,k,T,v_0,h,Ωinv,dg_ind_Ωinv,V_Ainv,S_0,VAinvDA0,AVAinvA);
        h = BEAVARs.Chan2020_draw_h!(h,s2_h,ρ,σ_h2,n,H_ρ,T);
        ρ, σ_h2, eh = BEAVARs.Chan2020_draw_ρ!(ρ,h,eh,v_h0,S_h0,ρ_0,V_ρ,T);
        
        
        Σt_inv = inv(Σ) 
        B_draw[:,:] = A';
        b0[:] = B_draw[:,1];
        structB_draw[:,n+1:end] = B_draw[:,2:end];
        
        cB,H_B,Σ_invsp  = BEAVARs.Blagov2025_updCPZ!(cB,H_B,Σ_invsp,B_draw,structB_draw,Σt_inv,Y0,cB_b0_LI,p,n,H_B_CI,strctBdraw_LI,Σt_LI);
        Σ_invsp         = BEAVARs.Blagov2025_updCPZcsv!(Σ_invsp,p,n,Tf,Ωinv);
        YYt             = BEAVARs.Blagov2025_draw_wz!(YYt,longyo,cB,Σ_invsp,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym);

        # update priors
        (deltaP, sigmaP, mu_prior) = BEAVARs.updatePriors3!(Y,X,n,mu_prior,deltaP,sigmaP,intercept,updP_vec);
        S_0                         = Diagonal(sigmaP);       
        (idx_kappa1,idx_kappa2, Vβ_Minn, β_Minn) = prior_NonConj(n,p,sigmaP,hypSetup);
        A_0     = reshape(β_Minn,k,n);
        V_Ainv  = sparse(1:k,1:k,1.0./Vβ_Minn);
        VAinvDA0 = V_Ainv\A_0;
        AVAinvA = A_0'*V_Ainv*A_0;   # this will not change unless we update the prior
    
        if ii>nburn
            store_YY[:,:,ii-nburn]  = YY;
            store_Σt_inv[:,:,ii-nburn]    = Σt_inv;
            # store_Σt[:,:,ii-nburn] = Σt;
            store_β[:,ii-nburn] = vec(A);
            store_h[:,ii-nburn] = h;
            store_Σt[:,:,ii-nburn] = Σ;
            store_s2_h[:,ii-nburn] = s2_h;
            store_ρ[ii-nburn,] = ρ;
            store_σ_h2[ii-nburn,] = σ_h2;
            store_eh[:,ii-nburn] = eh;
        end
    end

    return store_β, store_Σt_inv, store_YY, M_zsp, z_vec, Sm_bit, freq_mix_tp, store_Σt, store_h, store_s2_h, store_ρ, store_σ_h2, store_eh
end





#------------------------------
# Output structure
@with_kw struct VAROutput_Blagov2025 <: BVARmodelOutput
    store_β::Array{}        # 
    store_Σt_inv::Array{}        # 
    store_YY::Array{}
    M_zsp::Array{} 
    z_vec::Array{} 
    Sm_bit::Array{}
    freq_mix_tp
    store_Σt::Array{}        # 
    store_h::Array{}        # 
    store_s2_h::Array{}     # 
    store_ρ::Array{}        # 
    store_σ_h2::Array{}     # 
    store_eh::Array{}       #
end
# end of output strcutres
#------------------------------




# function dispatchModel(::Blagov2025_type,YY_tup, hyp_strct, p,n_burn,n_save,n_irf,n_fcst)
#     println("Hello Blagov2025")
#     intercept = 1;
#     dataHF_tab  = YY_tup[1]
#     dataLF_tab  = YY_tup[2]
#     varList     = YY_tup[3]
#     trans       = YY_tup[4] # transformation of the LF variables (0: growth rates or 1: log-levels)
#     set_strct = VARSetup(p,n_save,n_burn,n_irf,n_fcst,intercept);
#     store_YY,store_β, store_Σt_inv, M_zsp, z_vec, Sm_bit, store_Σ, store_h, store_s2_h, store_ρ, store_σ_h2, store_eh = Blagov2025(dataHF_tab,dataLF_tab,varList,set_strct,hyp_strct,trans)    
#     out_strct = VAROutput_Blagov2025(store_β,store_Σt_inv,store_YY,M_zsp, z_vec, Sm_bit,store_Σ, store_h, store_s2_h, store_ρ, store_σ_h2, store_eh)
#     return out_strct, set_strct
# end



"""
    Y, X, T, deltaP, sigmaP, mu_prior, V_Minn_inv, V_Minn_inv_elview, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, intercept, K_β, beta,  = CPZ_initMinn(YY,p)

    Initializes matrices for using the Minnesota prior in the CPZ2023 framework
"""
function Blagov2025_initcsv(YY,p,hypSetup)
    Y, X, T, n, intercept       = mlagL(YY,p);
    k                           = n*p+intercept
    sigmaP                      = ar4!(YY,zeros(n,));  # do OLS to initialize priors
    S_0                         = Diagonal(sigmaP);
    Σ = Matrix(S_0);              
    (idx_kappa1,idx_kappa2, Vβ_Minn, β_Minn) = prior_NonConj(n,p,sigmaP,hypSetup);
    A_0     = reshape(β_Minn,k,n);
    V_Ainv  = sparse(1:k,1:k,1.0./Vβ_Minn);
    VAinvDA0 = V_Ainv\A_0;
    AVAinvA = A_0'*V_Ainv*A_0;   # this will not change unless we update the prior
    v_0     = hypSetup.nu0+n;h = zeros(T,)
    H_ρ     = sparse(Matrix(1.0I, T, T)) - sparse(hypSetup.ρ*diagm(-1=>repeat(1:1,T-1)));
    h       = zeros(T,)
    eh      = similar(h);
    Ωinv    = sparse(1:T,1:T,exp.(-h));
    dg_ind_Ωinv = diagind(Ωinv);

    return Y, X, T, k, sigmaP, S_0, Σ, A_0, V_Ainv, v_0, H_ρ,h,eh,Ωinv, dg_ind_Ωinv, VAinvDA0, AVAinvA, intercept
end



@doc raw"""
    Draw with restrictions
"""
function Blagov2025_draw_wz!(YYt,longyo,cB,Σ_invsp,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym;nmdraws=10)
    
    mul!(Gm,H_B,Smsp);
    mul!(Go,H_B,Sosp);
    mul!(GΣ,Gm',Σ_invsp);
    mul!(Kym,GΣ,Gm);
    CL = cholesky(Hermitian(Kym))
    long_pr = (cB-Go*longyo);
    μ_y = CL.U\(CL.L\(GΣ*long_pr));

    KymBar = MOiM + Kym;
    CLBar = cholesky(Hermitian(KymBar))
    μ_yBar = CLBar.U\(CLBar.L\(MOiz + Kym*μ_y))

    mdraws = zeros(nm,nmdraws)
    for i_draw in 1:nmdraws
        mdraws[:,i_draw] = μ_yBar +  ldiv!(CLBar.U,randn(nm,))
    end    
    YYt[Sm_bit] = dropdims(median(mdraws,dims=2),dims=2);
    return YYt
end


function Blagov2025_updCPZ!(cB,H_B,Σ_invsp,B_draw,structB_draw,Σt_inv,Y0,cB_b0_LI,p,n,H_B_CI,strctBdraw_LI,Σt_LI)
    # updating cB
    BEAVARs.CPZ_update_cB!(cB,B_draw[:,2:end],B_draw[:,1],Y0,cB_b0_LI,p,n)

    # updating H_B
    H_B[H_B_CI] = -structB_draw[strctBdraw_LI];
    
    # updating Σ_invsp
    Σ_invsp.nzval[:] = Σt_inv[Σt_LI];
    return cB,H_B,Σ_invsp
end


function Blagov2025_updCPZcsv!(Σ_invsp,p,n,Tf,Ωinv)    
    # adding common-stoch-vol, since Σ^-1 = kron(Σt_inv,Ωinv)
    # we can just multiply each block with exp.(-h[ij]), because Ωinv = diagm(exp.(-h)) is diagonal 
    
    @views for ij in 1:Tf
        if ij < p+1
            lmul!(Ωinv[1,1], Σ_invsp[ 1 + (ij-1)*n : n + (ij-1)*n,1 + (ij-1)*n : n + (ij-1)*n])
        else 
            lmul!(Ωinv[ij-p,ij-p], Σ_invsp[ 1 + (ij-1)*n : n + (ij-1)*n,1 + (ij-1)*n : n + (ij-1)*n])
        end
    end
    return Σ_invsp
end


@doc raw"""
    Magg = Blagov2025_createMagg(dataHF_tab,freq_mix)

Create an `M` matrix for aggregating a high-frequency time-series to a lower frequency. 
    
The frequency relationship can be either monthly and quarterly or quarterly and yearly for growth rates or levels.
Consider `z=Magg*y` where `y` is a vector of monthly data. `Magg` is created in such a way so that `z` is the quarterly counterpart to `y`.  

# Arguments

    timeHF: a `Date` vector with the high-frequency dates to be matched to low-frequency dates
    freq_mix: a `Tuple` that states the relationship and transformation. (1,3,0) is monthly and quarterly growth rates, (1,3,1) are levels. 

The growth rate approximation follows Mariano and Murasawa (2003, 2010)    

References: Mariano and Murasawa (2010), A Coincident Index, Common Factors, and Monthly Real GDP, https://onlinelibrary.wiley.com/doi/full/10.1111/j.1468-0084.2009.00567.x
"""
function Blagov2025_createMagg(timeHF,freq_mix)
    # We will initialize some temporary objects here to generate an M matrix to convert monthly time series to quarterly (and quarterly to yearly later)
    
    if freq_mix==(1,3,0)||freq_mix==(1,3,1)
        selected_months = [1, 4, 7, 10]     # Jan, April, July, October
        # if we have growth rates we need 5 monthly rates to approximate the quarterly growth rate
        if freq_mix[3] == 0&& rem(month(timeHF[end]),3) !=0        # check whether the high-freq data ends at 3,6,9,12 motnh so that we have a full quarter to approximate
            # drop the last hf observations
            timeHF_sh = timeHF[1:end-rem(month(timeHF[end]),3),]
        else
            timeHF_sh = timeHF;
        end
        
    elseif freq_mix==(3,12,0)||freq_mix==(3,12,1)
        # add here the case for yearly and quarterly data
        #TODO
    end
    timeLF = filter(d -> month(d) in selected_months, timeHF_sh);
    Tpfhor = length(timeHF_sh);
    tempYYt = fill(NaN,(Tpfhor,))';
    tempZ_tab = TimeArray(timeLF,fill(NaN,(length(timeLF))));       # make a temporary z_tab to get an M matrix to convert monthly to quarterly for each series
    tempSm_bit = isnan.(tempYYt);
    nm = sum(tempSm_bit);
    out_tup = BEAVARs.CPZ_makeM_inter(tempZ_tab,tempYYt,tempSm_bit,timeHF_sh,colnames(tempZ_tab),colnames(tempZ_tab),freq_mix,nm,Tpfhor;scVal=10e-8)

    Magg = out_tup[1];
    return Magg, timeLF
end



@doc raw"""
    lf_mat, lf_mat_med = Blagov2025_hf2lf(out_strct,Magg,var_name::Symbol)

Convert high-frequency draws to low-frequency draws.

# Arguments
    out_strct: output structure from the Bayesian VAR model
    Magg:       matrix with the aggregation restrictions, output from Blagov2025_createMagg
    var_name:  Symbol for the variable that should be aggregated to lower frequency

See also Blagov2025_createMagg
"""
function Blagov2025_hf2lf(out_strct,Magg,var_name::Symbol)
 
    var_no = findfirst(==( var_name ), out_strct.var_list);   # here ==( :gdpBG ) is an anonymous function. ==(a, b) is the same as a == b and x -> x == :gdpBG is equiv to ==( :gdpBG )
    hf_mat = out_strct.store_YY[:,var_no,:]; # matrix with high-frequency data x draws to be aggregated to low-frequency
    lf_mat = Magg*hf_mat;                    # aggregated low-frequency x draws
    lf_mat_med = percentile_mat(lf_mat,0.5,dims=2);
    lf_mat_mean =  mean(lf_mat,dims=2);
    return lf_mat, lf_mat_med, lf_mat_mean
end


# #--------------------------------------
# # Forecast Blagov2025
# function forecast(VAROutput::VAROutput_Blagov2025,VARSetup)
#     @unpack store_β, store_Σt, store_h, s2_h_store, store_ρ, store_σ_h2,store_eh, YY = VAROutput
#     @unpack n_fcst,p,nsave = VARSetup

#     YY = median(store_YY,dims=3)

#     n = size(YY,2);

#     Yfor3D    = fill(NaN,(p+n_fcst,n,nsave))
#     Yfor3D[1:p,:,:] .= @views YY[end-p+1:end,:];
    
#     for i_draw = 1:nsave
#         # Yfor3D[1:p,:,i_draw] .= @views store_YY[end-p+1:end,:,i_draw];
#         Yfor3D[1:p,:,i_draw] .= @views YY[end-p+1:end,:];
#         Yfor = @views Yfor3D[:,:,i_draw];
#         A_draw = @views reshape(store_β[:,i_draw],n*p+1,n);
#         Σ_draw = @views store_Σt[:,:,i_draw];
                
#         for i_for = 1:n_fcst
#             tclass = @views vec(reverse(Yfor[1+i_for-1:p+i_for-1,:],dims=1)')
#             tclass = [1;tclass];
#             Yfor[p+i_for,:]=tclass'*A_draw  .+ (cholesky(Hermitian(Σ_draw)).U*randn(n,1))';    
#         end
#     end
#     return Yfor3D

# end # end function fcastCPZ2023()


"""
    THIS FUNCTINO IS ON THE TODO LIST
"""
function forecast(VAROutput::VAROutput_Blagov2025,VARSetup)
    @unpack store_β, store_Σt, store_h, s2_h_store, store_ρ, store_σ_h2,store_eh, store_YY = VAROutput
    @unpack n_fcst,p,nsave = VARSetup
    YY = median(store_YY,dims=3)        # centres forecasts on the median (can be relaxed)
    n = size(YY,2);

    Yfor3D    = fill(NaN,(p+n_fcst,n,nsave))
    hfor3D    = fill(NaN,(p+n_fcst,nsave)); 
    
    
    Yfor3D[1:p,:,:] .= @views YY[end-p+1:end,:];
    hfor3D[1:p,:] = @views store_h[end-p+1:end,:];
    
    for i_draw = 1:nsave
        Yfor3D[1:p,:,i_draw] .= @views store_YY[end-p+1:end,:,i_draw];
        # Yfor3D[1:p,:,i_draw] .= @views YY[end-p+1:end,:];
        hfor = @views hfor3D[:,i_draw];
        Yfor = @views Yfor3D[:,:,i_draw];
        A_draw = @views reshape(store_β[:,i_draw],n*p+1,n);
        ρ_draw = @view store_ρ[i_draw];
        σ_h2_draw = @views store_σ_h2[i_draw];
        Σ_draw = @views store_Σ[:,:,i_draw];
                
        for i_for = 1:n_fcst
            hfor[p+i_for,] = ρ_draw.*hfor[p+i_for-1,] + sqrt(σ_h2_draw).*randn()
            tclass = @views vec(reverse(Yfor[1+i_for-1:p+i_for-1,:],dims=1)')
            tclass = [1;tclass];
            Yfor[p+i_for,:]=tclass'*A_draw  .+ (exp.(hfor[p+i_for,]./2.0)*cholesky(Σ_draw).U*randn(n,1))';    
        end
    end
    return Yfor3D
end
