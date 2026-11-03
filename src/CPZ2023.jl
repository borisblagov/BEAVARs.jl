#-------------------------------------
# The Den: this is where the beavar lives
#-------------------------------------

@doc raw"""
    Main function for CPZ2023
"""
function beavar(::CPZ2023_type, set_struct::BVARmodelSetup, hyp_struct::BVARmodelHypSetup, data_struct::BVARmodelDataSetup)
    println("Hello CPZ2023")
    @unpack dataHF_tab,dataLF_tab, var_list = data_struct
    store_YY,store_YY_LF,store_β, store_Σt_inv, M_zsp, z_vec, Sm_bit,store_Σt, freq_mix_tp, fdatesHF, fdatesLF = CPZ2023(dataHF_tab,dataLF_tab,var_list,set_struct,hyp_struct);
    out_struct = VAROutput_CPZ2023(store_β,store_Σt_inv,store_YY,store_YY_LF, M_zsp, z_vec, Sm_bit,store_Σt,var_list,freq_mix_tp, fdatesHF, fdatesLF);
    return out_struct
end



@doc raw"""
    BEAVARs.CPZ2023(dataHF_tab::TimeArray{Typ,N,D,A},dataLF_tab::TimeArray{Typ,N,D,A},varOrder::Array{Symbol,1},varSetup::BVARmodelSetup,hyp_struct::BVARmodelHypSetup)
    
    Estimate Chan, Zhu, Poon 2023 using a  Minnesota-based independent Normal-Wishart prior
"""
function CPZ2023(dataHF_tab::TimeArray{Typ,N,D,A},dataLF_tab::TimeArray{Typ,N,D,A},varOrder::Array{Symbol,1},varSetup::BVARmodelSetup,hyp_struct::BVARmodelHypSetup) where {Typ <: AbstractFloat, N, D, A <: AbstractArray{Typ, N}}
    @unpack p, n_burn,n_save, const_loc, n_fcst, prior_RW = varSetup
    ndraws = n_save+n_burn;
    nmdraws = 10;               # given a draw from the parameters to draw multiple time from the distribution of the missing data for better confidence intervals

    fdataHF_tab, z_tab, freq_mix_tp, datesHF, varNamesLF, fvarNames = BEAVARs.init_MF_TimeArrays(dataLF_tab,dataHF_tab,varOrder,prior_RW,n_fcst)

    YYwNA = values(fdataHF_tab);
    YY = deepcopy(YYwNA);
    Tf,n = size(YY);
    
    B_draw, structB_draw, Σt_inv, b0 = BEAVARs.initParamMatrices(n,p,const_loc) 

    YYt, Y0, longyo, nm, H_B, H_B_CI, strctBdraw_LI, Σ_invsp, Σt_LI, Σp_invsp, Σpt_ind, Xb, cB, cB_b0_LI, Smsp, Sosp, Sm_bit, Gm, Go, GΣ, Kym,Σt_ns_CI, Σpt_ind_CI = BEAVARs.CPZ_initMatrices_v2(YY,structB_draw,b0,Σt_inv,p);
    
    M_zsp, z_vec, T_z, MOiM, MOiz = BEAVARs.makeM_intertemp(z_tab,YYt,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,nm,Tf);

    
    fdatesHF = timestamp(fdataHF_tab);
    fdatesLF = collect(timestamp(z_tab)[1]:Month(freq_mix_tp[2]):fdatesHF[end]);
    M_inter_agg = BEAVARs.makeMInterAgg(fdatesLF,fdatesHF,freq_mix_tp);
    
    # YY has missing values so we need to draw them once to be able to initialize matrices and prior values
    YYt = BEAVARs.CPZ_draw_wz!(YYt,longyo,Y0,cB,B_draw,structB_draw,strctBdraw_LI,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,H_B_CI,nmdraws,Σt_ns_CI);
    
    # we will be updating the priors for variables with many missing observations (>25%)
    updP_vec = sum(Sm_bit,dims=2).>size(Sm_bit,2)*0.25;
    
    # Initialize matrices for updating the parameter draws from CPZ_iniv  
    # ------------------------------------
    Y, X, T, deltaP, sigmaP, mu_prior, S_0, S_0_diag_view, Vβ_inv, Vβ_inv_diag_view, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, K_β, cholK_β, β_draw, intercept = CPZ_init_Minn(YY,p)

    # Estimate the prior using the first draw
    (idx_kappa1,idx_kappa2, Vβ_vec, βMinn) = BEAVARs.prior_Minn(n,p,sigmaP,hyp_struct,prior_RW);

    # Update the initialized matrices
    Vβ_inv_diag_view[:] = 1.0./Vβ_vec;                # update the diagonal of Vβ_inv

    # prepare matrices for storage
    store_YY    = zeros(Tf,n,n_save);
    store_β     = zeros(n^2*p+n,n_save);
    store_Σt_inv= zeros(n,n,n_save);
    store_Σt    = zeros(n,n,n_save);

    μ_yBar = zeros(nm,)
    KymBar = similar(Kym);
    mdraws = zeros(nm,nmdraws)
    draw_tmp = zeros(nm)
    for ii in 1:ndraws
        # draw of the missing values
        BEAVARs.CPZ_draw_wz!(YYt,longyo,Y0,cB,B_draw,structB_draw,strctBdraw_LI,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,H_B_CI,nmdraws,Σt_ns_CI);
       
        # draw of the parameters
        beta,b0,B_draw,Σt_inv,structB_draw,Σt = BEAVARs.CPZ_iniw!(YY,p,hyp_struct,n,k,b0,B_draw,Σt_inv,structB_draw,Σp_invsp,Σpt_ind,Y,X,T,Xsur_den,Xsur_CI,X_CI,XtΣ_inv_den,XtΣ_inv_X,Vβ_inv,βMinn,K_β,cholK_β,β_draw,S_0, Σpt_ind_CI);

        if ii>n_burn
            store_β[:,ii-n_burn]  = beta;
            store_YY[:,:,ii-n_burn]  = YY;
            store_Σt_inv[:,:,ii-n_burn]    = Σt_inv;
            store_Σt[:,:,ii-n_burn] = Σt;
        end
    end
    store_YY_LF = mapslices(x->M_inter_agg*x,store_YY,dims=1:2);
    return store_YY, store_YY_LF, store_β, store_Σt_inv, M_zsp, z_vec, Sm_bit, store_Σt, freq_mix_tp, fdatesHF, fdatesLF
end


# Data functions
@doc raw"""
    makeDataSetup(::CPZ2023_type,dataHF_tab::TimeArray, dataLF_tab::TimeArray; var_list =  [colnames(dataHF_tab); colnames(dataLF_tab)])

Generate data for a mixed-frequency VAR. Uses Time Arrays from the TimeSeries package
    
# Arguments
    dataHF_tab: TimeArray with your high-frequency variables (monthly or quarterly, respectively)
    dataLF_tab: TimeArray with your low-frequency variables (quarterly or yearly, respectively)
    var_list:   the variable order. Note that the functions that call these variables allow this to be optional.


"""
function makeDataSetup(::CPZ2023_type,dataHF_tab::TimeArray, dataLF_tab::TimeArray; var_list =  [colnames(dataLF_tab); colnames(dataHF_tab)])
    return data_MFVAR_2f(dataHF_tab, dataLF_tab, var_list)
end





@doc raw"""
    CPZ_update_cB!()

    Uses
    B = [B1 B2] =  [a b e f 
                    c d g h]
    and populates the cB vector by taking 
    B1*y_{-1} + B2*y_{-2}

    The indices loop on line 4 can also be made to support
    B = [B0 B1 B2]
    structure, by omitting B0 and starting from B1 (same as above) if you use
    Bmat[:,1+(p-io+kk)*n:(p-io+kk)*n+n]
"""
function CPZ_update_cB!(cB::Vector{Float64},Bmat,b0,Y0,cB_b0_LI::Vector{Int64},p::Int,n::Int)
    for io = 0:p-1
        ytmp = zeros(n,);
        for kk = 0:io
            ytmp1 = Bmat[:,1+(p-io+kk)*n-n:(p-io+kk)*n+n-n]*Y0[p-kk,:];  # This is \sum_1^p B_j y_{t-j}. For t = 0 : cB = b0 + B_p y0
            ytmp = ytmp+ytmp1;
        end
        ytmp = ytmp + b0;
        cB[n*(p-io)-n+1 : n*(p-io)-n+n,] = ytmp;
    end
    cB[n*p-n+1+n : end]=b0[cB_b0_LI]
    return cB
end




@doc raw"""
    
"""
function CPZ_initMatrices_v2(YY,structB_draw,b0,Σt_inv,p)
    (Tf,n) = size(YY); # full time span (with initial conditions)
    k = n*(p+1); kn = k*n
    Tfn = n*Tf;
    
    YYt = (YY');
    vYYt = vec(YYt);
        
    Sm_bit = isnan.(YYt)
    So_bit = .!isnan.(YYt)
    longyo = vYYt[vec(So_bit)];

    Y0 = @views YY[1:p,:]
    if any(isnan.(Y0))
        Y0[isnan.(Y0)]=zeros(size(Y0[isnan.(Y0)],1)); # for the first pass remove NaNs for zeroes
        # print("NaNs found in Y0, replaced with zeros");
    end
    
    indC_nan_wide = findall(Sm_bit) #  Cartesian indices of missing values
    # indC_non_wide = findall(!isnan,YYt)  # Cartesian indices of not missing values
    indC_non_wide = findall(So_bit)  # Cartesian indices of not missing values
    
    # convert between linear and cartesian indices
    indL_all = LinearIndices(YYt);
    indL_nan_wide = indL_all[indC_nan_wide] # are the linear indices of NaN values
    indL_non_wide = indL_all[indC_non_wide] # are the linear indices of non NaN values
       
    
    nm = sum(Sm_bit);       # the number of missing values
    Smsp = sparse(indL_nan_wide, 1:nm, ones(nm), Tfn, nm);
    n_observed = length(indL_non_wide)
    Sosp = sparse(indL_non_wide, 1:n_observed, ones(n_observed), Tfn, n_observed);
    
    # Initialize matrices
    H_B, H_B_CI, strctBdraw_LI = BEAVARs.makeBlkDiag_ns(Tfn,n,p, -structB_draw);
    Σ_inv, Σt_ns_CI, Σt_ns_LI = BEAVARs.makeBlkDiag_ns(Tfn,n,0,Σt_inv);   # make a non-sparse matrix if needed                         # this is ( I(Tf*n) ⊗ Σ-1 )
    Σp_inv, Σpt_ind_CI, Σpt_ind_ns_LI = BEAVARs.makeBlkDiag_ns(Tfn-n*p,n,0,Σt_inv);                # this is ( I(T*n) ⊗ Σ-1 ), the difference is that this includes the 0,-1,...,-p lags
    cB_b0_LI = repeat(1:n,div(Tfn-n*p-n+1+n,n));  # this repeats [1:n] so that we can update cB[indicesAfter Y_0,Y_{-1}, ..., Ymp] = b0[cB_b0_LI]
    Xb = sparse(Matrix(1.0I, Tfn, Tfn))
    cB = repeat(b0,Tf);
    
    Gm = H_B*Smsp; Go = H_B*Sosp; GΣ = Gm'*Σ_inv; Kym = GΣ*Gm; # we can initialize all these and then mutate with mul!()
    
    return YYt, Y0, longyo, nm, H_B, H_B_CI, strctBdraw_LI, Σ_inv, Σt_ns_LI, Σp_inv, Σpt_ind_ns_LI, Xb, cB, cB_b0_LI, Smsp, Sosp, Sm_bit, Gm, Go, GΣ, Kym,Σt_ns_CI, Σpt_ind_CI;
end

@doc raw"""

"""
function CPZ_draw!(YYt,longyo,Y0,cB,B_draw,structB_draw,sBd_ind,Σt_inv,Σt_LI,Xb,cB_b0_LI,H_Bsp,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,nm)
    # updating cB
    BEAVARs.CPZ_update_cB!(cB,B_draw[:,2:end],B_draw[:,1],Y0,cB_b0_LI,p,n)

    # updating H_B
    H_Bsp.nzval[:] = -structB_draw[sBd_ind];

    # updating Σ_invsp
    Σ_invsp.nzval[:] = Σt_inv[Σt_LI];

    Gm = H_Bsp*Smsp
    Go = H_Bsp*Sosp
    Kym     = Gm'*Σ_invsp*Gm
    CL = cholesky(Hermitian(Kym))
    μ_y = CL.UP\(CL.PtL\Gm'*Σ_invsp)*(Xb*cB-Go*longyo)

    YYt[Sm_bit] = μ_y + CL.UP\randn(nm,)
    return YYt
end





@doc raw"""
    Draw with restrictions
"""
function CPZ_draw_wz!(YYt,longyo,Y0,cB,B_draw,structB_draw,sBd_ind,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_inv,p,n,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,H_B_CI,nmdraws,Σt_ns_CI)
    # updating cB
    BEAVARs.CPZ_update_cB!(cB,B_draw[:,2:end],B_draw[:,1],Y0,cB_b0_LI,p,n)

    # updating H_B
    H_B[H_B_CI] = -structB_draw[sBd_ind];
    # updating Σ_invFsp
    #  Σ_invsp.nzval[:] = Σt_inv[Σt_LI];
     Σ_inv[Σt_ns_CI] = Σt_inv[Σt_LI];

    mul!(Gm,H_B,Smsp);
    mul!(Go,H_B,Sosp);
    mul!(GΣ,Gm',Σ_inv);
    mul!(Kym,GΣ,Gm);
    CL = cholesky(Hermitian(Kym))
    long_pr = (cB-Go*longyo);
    μ_y = CL.U\(CL.U'\(GΣ*long_pr));

    KymBar = MOiM + Kym;
    CLBar = cholesky(Hermitian(KymBar))
    μ_yBar = CLBar.U\(CLBar.U'\(MOiz + Kym*μ_y))
    
    mdraws = zeros(nm,nmdraws)
    for i_draw in 1:nmdraws
        mdraws[:,i_draw] = μ_yBar +  ldiv!(CLBar.U,randn(nm,))
    end    
    YYt[Sm_bit] = dropdims(median(mdraws,dims=2),dims=2);
    return YYt
end






@doc raw"""
    Updates parameters using an independennt Normal-Wishart prior

    This is the original function that has been written, but it can be optimized for allocations a lot and it also updates the prior of the VAR. This might be undesirable for some.
"""
function CPZ_iniw_bckp(YY,p,hyp_struct,n,k,b0,B_draw,Σt_inv,structB_draw,Σp_invsp,Σpt_ind,Y,X,T,mu_prior,deltaP,sigmaP,intercept,Xsur_den,Xsur_CI,X_CI,XtΣ_inv_den,XtΣ_inv_X,Vβ_inv,Vβ_inv_diag_view,upd_these_vec,K_β,beta,prior_RW)
    Y, X = mlagL!(YY,Y,X,p,n)
    (deltaP, sigmaP, mu_prior) = BEAVARs.updatePriors_bitVec!(Y,X,n,mu_prior,deltaP,sigmaP,intercept,upd_these_vec);
    S_0 = Diagonal(sigmaP);
    beta_Minn = zeros(n^2*p+n);
    idx_kappa1,idx_kappa2, V_Minn_vec = prior_Minn(n,p,sigmaP,hyp_struct,prior_RW);
    V_Minn_vec_inv = 1.0./V_Minn_vec;
    Σp_invsp.nzval[:] = Σt_inv[Σpt_ind];    
    Xsur_den[Xsur_CI] = X[X_CI]; 
    Vβ_inv_diag_view[:] = V_Minn_vec_inv;  # update the diagonal of Vβ_inv, i.e. V_Minn^-1

    beta[:] = BEAVARs.Chan2020_drawβ(Σp_invsp,Xsur_den,XtΣ_inv_den,XtΣ_inv_X,Vβ_inv,beta_Minn,K_β,Y,n,k);
    

    B_draw[:,:] = reshape(beta,k,n)'
    b0[:] = B_draw[:,1]
    structB_draw[:,n+1:end] = B_draw[:,2:end]


    # errors 
    Σt, Σt_inv = Chan2020_drawΣt(Y,Xsur_den,beta,n,T,S_0,hyp_struct.nu0);

    return beta,b0,B_draw,Σt_inv,structB_draw,Σt
end


@doc raw"""
    CPZ_iniw!(YY,p,hyp_struct,n,k,b0,B_draw,Σt_inv,structB_draw,Σp_invsp,Σpt_ind,Y,X,T,Xsur_den,Xsur_CI,X_CI,XtΣ_inv_den,XtΣ_inv_X,Vβ_inv,βMinn,K_β,cholK_β,β_draw,S_0, Σpt_ind_CI)
    Y, X = mlagL!(YY,Y,X,p,n)

    Draws parameters using an independennt Normal-Wishart prior

    Updates (i.e. overwrites): 
        β_draw, 
        B_draw, 
        structB_draw,
        Σt,
        Σt_inv
"""
function CPZ_iniw!(YY,p,hyp_struct,n,k,b0,B_draw,Σt_inv,structB_draw,Σp_invsp,Σpt_ind,Y,X,T,Xsur_den,Xsur_CI,X_CI,XtΣ_inv_den,XtΣ_inv_X,Vβ_inv,βMinn,K_β,cholK_β,β_draw,S_0, Σpt_ind_CI)
    Y, X = mlagL!(YY,Y,X,p,n)
    # Σp_invsp.nzval[:] = Σt_inv[Σpt_ind];
    Σp_invsp[Σpt_ind_CI] = Σt_inv[Σpt_ind];     
    Xsur_den[Xsur_CI] = X[X_CI]; 

    β_draw[:] = BEAVARs.Chan2020_drawβ(Σp_invsp,Xsur_den,XtΣ_inv_den,XtΣ_inv_X,Vβ_inv,βMinn,K_β,Y,n,k);
    # β_draw[:] = BEAVARs.Chan2020_drawβ_nonsp(Σp_invsp,Xsur_den,XtΣ_inv_den,XtΣ_inv_X,Vβ_inv,βMinn,K_β,cholK_β,Y,n,k);
    

    B_draw[:,:] = reshape(β_draw,k,n)'
    b0[:] = B_draw[:,1]
    structB_draw[:,n+1:end] = B_draw[:,2:end]


    # errors 
    Σt, Σt_inv = Chan2020_drawΣt(Y,Xsur_den,β_draw,n,T,S_0,hyp_struct.nu0);

    return β_draw,b0,B_draw,Σt_inv,structB_draw,Σt
end



"""
    Y, X, T, deltaP, sigmaP, mu_prior, Vβ_inv, Vβ_inv_diag_view, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, intercept, K_β, beta,  = CPZ_init_Minn(YY,p)

    Initializes matrices for using the Minnesota prior in the CPZ2023 framework
"""
function CPZ_init_Minn(YY,p)
    Y, X, T, n, intercept       = mlagL(YY,p);
    k                           = n*p+intercept
    (deltaP, sigmaP, mu_prior)  = trainPriors(YY,p);                         # do OLS to initialize priors
    S_0                         = Diagonal(sigmaP);                         # Initialize S_0       
    S_0_diag_view               = @view(S_0[diagind(S_0)]);                 # view to the diagonal of S_0 to update the prior without allocations
    Vβ_inv                      = 1.0*Matrix(I,n*k,n*k);                    # inverse of the prior variance matrix, Vβ, of the coefficients β
    Vβ_inv_diag_view            = @view(Vβ_inv[diagind(Vβ_inv)]);           # will be used to update the diagonal of Vβ_inv without allocations with 1.0/Vβ_vec (i.e. inv of Vβ_vec)
    XtΣ_inv_den                 = zeros(k*n,T*n);                           # this is X' ( I(T) ⊗ Σ-1 )   from page 6 in Chan 2020 LBA
    XtΣ_inv_X                   = zeros(n*k,n*k);                           # this is X' ( I(T) ⊗ Σ-1 ) X from page 6 in Chan 2020 LBA    
    Xsur_den, Xsur_CI, X_CI     = BEAVARs.SUR_form_dense(X,n);              # prepares the SUR form and the indices of the parameters for updating
    K_β                         = 1.0*Matrix(I,n*k,n*k);                    # Variance covariance matrix of the parameters
    cholK_β                     = cholesky(Hermitian(K_β))                  # pre-allocate a cholesky object for later use with cholesky!
    β_draw                      = zeros(n*k,);                              # will house the Minnesota parameters vector

    return Y, X, T, deltaP, sigmaP, mu_prior, S_0, S_0_diag_view, Vβ_inv, Vβ_inv_diag_view, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, K_β, cholK_β, β_draw, intercept
end



#------------------------------
# Output structure
@with_kw struct VAROutput_CPZ2023{T <: AbstractFloat, N} <: BVARmodelOutput
    store_β::Array{T,N}        # 
    store_Σt_inv::Array{T,3}        # 
    store_YY::Array{T,3}
    store_YY_LF::Array{T,3}
    M_zsp::Array{T,N} 
    z_vec::Array{T,1} 
    Sm_bit::Array{Bool}
    store_Σt::Array{T,3}        # 
    var_list::Array{Symbol,1}
    freq_mix_tp::Tuple{Int,Int,Int}
    # M_inter_agg::Array{T,N}
    fdatesHF::Array{Date, 1}
    fdatesLF::Array{Date, 1}
end
# end of output strcutres
#------------------------------



#--------------------------------------
# Forecast Block for CPZ2023
@doc raw"""
    forecast(VAROutput::VAROutput_CPZ2023,VARSetup::BVARmodelSetup,data_struct::BVARmodelDataSetup)


"""
function makeForecast(VAROutput::VAROutput_CPZ2023,VARSetup::BVARmodelSetup,data_struct::BVARmodelDataSetup)
    @unpack store_β, store_Σt, store_YY, M_inter_agg, fdatesHF, fdatesLF = VAROutput
    @unpack n_fcst,p,n_save = VARSetup
    @unpack dataHF_tab, dataLF_tab, var_list = data_struct
    YYforHF3d = store_YY;
    YYforLF3d = mapslices(x->M_inter_agg*x,store_YY,dims=1:2)

    # Calculate the percentiles for the forecast distribution for the low frequency variables
    YYforLF_low025, YYforLF_low05, YYforLF_low16, YforLF_med, YYforLF_hih84, YYforLF_hih95, YYforLF_hih975 = get_imp_percentiles(YYforLF3d) 

    YYforLF_low025_tab = rename!(TimeArray(fdatesLF,YYforLF_low025),var_list);
    YYforLF_low05_tab = rename!(TimeArray(fdatesLF,YYforLF_low05),var_list);
    YYforLF_low16_tab = rename!(TimeArray(fdatesLF,YYforLF_low16),var_list);
    YYforLF_med_tab = rename!(TimeArray(fdatesLF,YforLF_med),var_list);
    YYforLF_hih84_tab = rename!(TimeArray(fdatesLF,YYforLF_hih84),var_list);
    YYforLF_hih95_tab = rename!(TimeArray(fdatesLF,YYforLF_hih95),var_list);
    YYforLF_hih975_tab = rename!(TimeArray(fdatesLF,YYforLF_hih975),var_list);

    # now for the high-frequency variables
    YYforHF_low025, YYforHF_low05, YYforHF_low16, YforHF_med, YYforHF_hih84, YYforHF_hih95, YYforHF_hih975 = get_imp_percentiles(YYforHF3d) 

    YYforHF_low025_tab = rename!(TimeArray(fdatesHF,YYforHF_low025),var_list);
    YYforHF_low05_tab = rename!(TimeArray(fdatesHF,YYforHF_low05),var_list);
    YYforHF_low16_tab = rename!(TimeArray(fdatesHF,YYforHF_low16),var_list);
    YYforHF_med_tab = rename!(TimeArray(fdatesHF,YforHF_med),var_list);
    YYforHF_hih84_tab = rename!(TimeArray(fdatesHF,YYforHF_hih84),var_list);
    YYforHF_hih95_tab = rename!(TimeArray(fdatesHF,YYforHF_hih95),var_list);
    YYforHF_hih975_tab = rename!(TimeArray(fdatesHF,YYforHF_hih975),var_list);

    # save them in a structure with [5%, 16%, 50%, 84%, 95%] probability intervals for each variable and each time point
    YYforLF_struct = BEAVARs.data_probability_intervals(YYforLF_low025_tab, YYforLF_low05_tab, YYforLF_low16_tab, YYforLF_med_tab, YYforLF_hih84_tab, YYforLF_hih95_tab, YYforLF_hih975_tab);
    YYforHF_struct = BEAVARs.data_probability_intervals(YYforHF_low025_tab,YYforHF_low05_tab, YYforHF_low16_tab, YYforHF_med_tab, YYforHF_hih84_tab, YYforHF_hih95_tab,YYforHF_hih975_tab);

    data_flags_vec = fdatesLF .∈  Ref(timestamp(dataLF_tab));       # bit_vector showing the data position
    forecast_flags_vec = .!data_flags_vec;                          # bit_vector showing the forecasted position. Only supports balanced z_tab #TODO: make it work for unbalanced z_tab
    fcast_struct = BEAVARs.VAR_MF_Forecast(YYforHF3d,YYforLF3d,dataHF_tab,dataLF_tab,var_list,n_fcst,YYforHF_struct,YYforLF_struct,data_flags_vec,forecast_flags_vec)    

    return fcast_struct

end # end function fcastCPZ2023()


## Model fit block

"""
    modelFit(out_struct,varSetup)

     Generates fitted values from the model output and the model setup. Used for in-sample fit and for calculating residuals for the training sample.

     Arguments:
        out_struct: A structure with the model output, including the store_β and YY matrices
        varSetup: A structure with the model setup, including p and const_loc

     Returns:
        Yfit: The fitted values from the model
        Yact: The actual values from the data (YY)


        ## TODO: rewrite the function so that we don't use the median and then add a plotting function to it
"""
function modelFit(out_struct::VAROutput_CPZ2023,varSetup::BEAVARs.VARSetup)
    @unpack p, const_loc = varSetup;
    YY = median(out_struct.store_YY,dims=3);
    if const_loc == 1
        Y,X,T,n = BEAVARs.mlagL(YY,p);
    elseif const_loc == 0
        Y,X,T,n = BEAVARs.mlag(YY,p);
    end
    Amed = reshape(percentile_mat(out_struct.store_β,0.5,dims=2),n*p+1,n)
    Yfit = X*Amed;
    Yact = @views Y
    return Yfit, Yact
end


##--------------------------------------
#
#   CONSTRUCTORS
#
#--------------------------------------

# Hyperparameters structure
"""
    makeHypSetup(::CPZ2023_type)

    Constructs the structure with the hyperparameters for the CPZ2023 model. Calls the function hypChan2020() that initializes the hyperparameters as in Chan 2020, but in the future we might want to add more options for different hyperparameter settings.

    Arguments:
        ::CPZ2023_type: A type that indicates that we want to use the CPZ2023 model. This is a dummy argument that is used to dispatch on the type of model we want to use.

    Returns:
        A structure with the hyperparameters for the CPZ2023 model, currently initialized as in Chan 2020.

    See also:
        - hypChan2020() for initializing the hyperparameters as in Chan 2020.
"""
function makeHypSetup(::CPZ2023_type)
    return hypChan2020()
end

