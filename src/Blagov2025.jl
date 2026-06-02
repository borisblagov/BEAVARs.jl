function makeHypSetup(::Blagov2025_type)
    return hypChan2020()
end



# Structure for the datasets and the frequency mix
@doc raw"""
    dataBlagov2025(data_HF::TimeArray,data_LF::TimeArray,prior_RW::Int,var_list::Array{Symbol,1})

generate a dataset strcture for use with Blagov2025 model

# Arguments
    dataHF_tab: TimeArray with your high-frequency variables (monthly or quarterly, respectively)
    dataLF_tab: TimeArray with your low-frequency variables (quarterly or yearly, respectively)
    var_list:   the variable order. Note that the functions that call these variables allow this to be optional.

See also `makeDataSetup`.
"""
@with_kw struct dataBlagov2025 <: BVARmodelDataSetup
    dataHF_tab::TimeArray                                       # data for the high-frequency variables
    dataLF_tab::TimeArray                                       # data for the low-frequency variables
    var_list::Array{Symbol,1}                                   # Symbol vector with the variable names, will be used for ordering
end

@doc raw"""
    Prepare the structure containg the data for the mixed-frequency VAR. Uses Time Arrays from the TimeSeries package
"""
function makeDataSetup(::Blagov2025_type,dataHF_tab::TimeArray, dataLF_tab::TimeArray; var_list =  [colnames(dataHF_tab); colnames(dataLF_tab)])
    return dataBlagov2025(dataHF_tab, dataLF_tab, var_list)
end



@doc raw"""
     BEAVARs.Blagov2025(dataHF_tab,dataLF_tab,varList,varSetup,hypSetup)

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
function Blagov2025(dataHF_tab,dataLF_tab,varOrder,varSetup,hypSetup)
    @unpack ρ, σ_h2, v_h0, S_h0, ρ_0, V_ρ = hypSetup
    @unpack p, nsave, nburn, n_fcst, const_loc, prior_RW = varSetup
    ndraws = nsave+nburn;
    # nmdraws = 10;               # given a draw from the parameters to draw multiple time from the distribution of the missing data for better confidence intervals

    fdataHF_tab, z_tab, freq_mix_tp, datesHF, varNamesLF, fvarNames = BEAVARs.CPZ_prep_TimeArrays(dataLF_tab,dataHF_tab,varOrder,prior_RW,n_fcst)

    YYwNA = values(fdataHF_tab);
    YY = deepcopy(YYwNA);
    Tf,n = size(YY);
    
    B_draw, structB_draw, Σt_inv, b0 = BEAVARs.initParamMatrices(n,p,const_loc) 
    
    YYt, Y0, longyo, nm, H_B, H_B_CI, strctBdraw_LI, Σ_invsp, Σt_LI, Σp_invsp, Σpt_ind, Xb, cB, cB_b0_LI, Smsp, Sosp, Sm_bit, Gm, Go, GΣ, Kym = BEAVARs.CPZ_initMatrices(YY,structB_draw,b0,Σt_inv,p);
    
    M_zsp, z_vec, T_z, MOiM, MOiz = BEAVARs.CPZ_makeM_inter(z_tab,YYt,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,nm,Tf);
    
    
    
    fdatesHF = timestamp(fdataHF_tab);
    fdatesLF = collect(timestamp(z_tab)[1]:Month(freq_mix_tp[2]):fdatesHF[end]);
    M_inter_agg = BEAVARs.CPZ_makeM_inter_agg(fdatesLF,fdatesHF,freq_mix_tp);
    
    # YY has missing values so we need to draw them once to be able to initialize matrices and prior values
    cB,H_B,Σ_invsp  = BEAVARs.Blagov2025_updCPZ!(cB,H_B,Σ_invsp,B_draw,structB_draw,Σt_inv,Y0,cB_b0_LI,p,n,H_B_CI,strctBdraw_LI,Σt_LI);
    YYt             = BEAVARs.Blagov2025_draw_wz!(YYt,longyo,cB,Σ_invsp,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym);
    
        
    # Initialize matrices for updating the parameter draws from CPZ_iniv  
    # ------------------------------------
    Y, X, T, k, sigmaP, S_0, Σ, A_0, V_Ainv, v_0, H_ρ,h,eh,Ωinv, dg_ind_Ωinv, VAinvDA0, AVAinvA, intercept   = BEAVARs.Blagov2025_initcsv(YY,p,hypSetup,prior_RW);
    
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
        (deltaP, sigmaP, mu_prior) = BEAVARs.updatePriors_bitVec!(Y,X,n,mu_prior,deltaP,sigmaP,intercept,updP_vec);
        S_0                         = Diagonal(sigmaP);       
        (idx_kappa1,idx_kappa2, Vβ_Minn) = prior_NatConj(n,p,sigmaP,hypSetup);
        # A0 doesn't change, it is either zeros(n*p+1,n) or [ones(n,1); eye(n); zeros((n-1)*p,n)]
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

    return store_β, store_Σt_inv, store_YY, M_zsp, z_vec, Sm_bit, freq_mix_tp, store_Σt, store_h, store_s2_h, store_ρ, store_σ_h2, store_eh, M_inter_agg, fdatesHF, fdatesLF
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
    M_inter_agg::Array{}
    fdatesHF::Array{Date, 1}
    fdatesLF::Array{Date, 1}
end
# end of output strcutres
#------------------------------




"""
    Y, X, T, deltaP, sigmaP, mu_prior, V_Minn_inv, V_Minn_inv_elview, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, intercept, K_β, beta,  = CPZ_initMinn(YY,p)

    Initializes matrices for using the Minnesota prior in the CPZ2023 framework
"""
function Blagov2025_initcsv(YY,p,hypSetup,prior_RW)
    Y, X, T, n, intercept       = mlagL(YY,p);
    k                           = n*p+intercept
    sigmaP                      = ar4!(YY,zeros(n,));  # do OLS to initialize priors
    S_0                         = Diagonal(sigmaP);
    Σ = Matrix(S_0);              
    (idx_kappa1,idx_kappa2, Vβ_Minn) = prior_NatConj(n,p,sigmaP,hypSetup);
    A_0 = zeros(n*p+1,n)
    if prior_RW == 1
        A_0[2:n+1,1:n] = Matrix(1.0I, n, n)     # account for the constant on the top row
    end
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
        #TODO: add here the case for yearly and quarterly data
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
    lf_mat, lf_mat_med = Blagov2025_hf2lf(out_struct,Magg,var_name::Symbol)

Convert high-frequency draws to low-frequency draws.

# Arguments
    out_struct: output structure from the Bayesian VAR model
    Magg:       matrix with the aggregation restrictions, output from Blagov2025_createMagg
    var_name:  Symbol for the variable that should be aggregated to lower frequency

See also Blagov2025_createMagg
"""
function Blagov2025_hf2lf(out_struct,Magg,var_name::Symbol)
 
    var_no = findfirst(==( var_name ), out_struct.var_list);   # here ==( :gdpBG ) is an anonymous function. ==(a, b) is the same as a == b and x -> x == :gdpBG is equiv to ==( :gdpBG )
    hf_mat = out_struct.store_YY[:,var_no,:]; # matrix with high-frequency data x draws to be aggregated to low-frequency
    lf_mat = Magg*hf_mat;                    # aggregated low-frequency x draws
    lf_mat_med = percentile_mat(lf_mat,0.5,dims=2);
    lf_mat_mean =  mean(lf_mat,dims=2);
    return lf_mat, lf_mat_med, lf_mat_mean
end




#--------------------------------------
# Forecast Block for Blagov2025
@doc raw"""

"""
function forecast(VAROutput::VAROutput_Blagov2025,VARSetup::BVARmodelSetup,data_struct::BVARmodelDataSetup)

    @unpack store_β, store_Σt, store_YY, M_inter_agg, fdatesHF, fdatesLF = VAROutput
    @unpack n_fcst,p,nsave = VARSetup
    @unpack dataHF_tab, dataLF_tab, var_list = data_struct
    YYforHF3d = store_YY;
    YYforLF3d = mapslices(x->M_inter_agg*x,store_YY,dims=1:2)

    # Calculate the percentiles for the forecast distribution for the low frequency and high-frequency variables
    YforLF_low1 = percentile_mat(YYforLF3d,0.05,dims=3);
    YforLF_low = percentile_mat(YYforLF3d,0.16,dims=3);
    YforLF_med = percentile_mat(YYforLF3d,0.5,dims=3);
    YforLF_hih = percentile_mat(YYforLF3d,0.84,dims=3);
    YforLF_hih1 = percentile_mat(YYforLF3d,0.95,dims=3);

    YYforLF_low05_tab = rename!(TimeArray(fdatesLF,YforLF_low1),var_list);
    YYforLF_low16_tab = rename!(TimeArray(fdatesLF,YforLF_low),var_list);
    YYforLF_med_tab = rename!(TimeArray(fdatesLF,YforLF_med),var_list);
    YYforLF_hih84_tab = rename!(TimeArray(fdatesLF,YforLF_hih),var_list);
    YYforLF_hih95_tab = rename!(TimeArray(fdatesLF,YforLF_hih1),var_list);

    # now for the high-frequency variables
    YforHF_low1 = percentile_mat(YYforHF3d,0.05,dims=3);
    YforHF_low = percentile_mat(YYforHF3d,0.16,dims=3);
    YforHF_med = percentile_mat(YYforHF3d,0.5,dims=3);
    YforHF_hih = percentile_mat(YYforHF3d,0.84,dims=3);
    YforHF_hih1 = percentile_mat(YYforHF3d,0.95,dims=3);

    YYforHF_low05_tab = rename!(TimeArray(fdatesHF,YforHF_low1),var_list);
    YYforHF_low16_tab = rename!(TimeArray(fdatesHF,YforHF_low),var_list);
    YYforHF_med_tab = rename!(TimeArray(fdatesHF,YforHF_med),var_list);
    YYforHF_hih84_tab = rename!(TimeArray(fdatesHF,YforHF_hih),var_list);
    YYforHF_hih95_tab = rename!(TimeArray(fdatesHF,YforHF_hih1),var_list);

    # save them in a structure with [5%, 16%, 50%, 84%, 95%] probability intervals for each variable and each time point
    YYforLF_struct = BEAVARs.data_fcast_PI(YYforLF_low05_tab, YYforLF_low16_tab, YYforLF_med_tab, YYforLF_hih84_tab, YYforLF_hih95_tab);
    YYforHF_struct = BEAVARs.data_fcast_PI(YYforHF_low05_tab, YYforHF_low16_tab, YYforHF_med_tab, YYforHF_hih84_tab, YYforHF_hih95_tab);

    data_flags_vec = fdatesLF .∈  Ref(timestamp(dataLF_tab));       # bit_vector showing the data position
    forecast_flags_vec = .!data_flags_vec;                          # bit_vector showing the forecasted position. Only supports balanced z_tab #TODO: make it work for unbalanced z_tab
    fcast_struct = BEAVARs.VAR_MF_Forecast(YYforHF3d,YYforLF3d,dataHF_tab,dataLF_tab,var_list,n_fcst,YYforHF_struct,YYforLF_struct,data_flags_vec,forecast_flags_vec)    

    return fcast_struct

end # end function fcastCPZ2023()