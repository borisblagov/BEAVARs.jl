#-------------------------------------
# The Den: this is where the beavar lives
#-------------------------------------

@doc raw"""
    Main function for CPZ2023dev
"""
function beavar(::CPZ2023dev_type, set_struct::BVARmodelSetup, hyp_struct::BVARmodelHypSetup, data_struct::BVARmodelDataSetup)
    println("Hello CPZ2023dev")
    @unpack dataHF_tab,dataLF_tab, var_list = data_struct
    store_YY,store_YY_LF,store_β, store_Σt_inv, M_zsp, z_vec, Sm_bit,store_Σt, freq_mix_tp, fdatesHF, fdatesLF = CPZ2023dev(dataHF_tab,dataLF_tab,var_list,set_struct,hyp_struct);
    out_struct = VAROutput_CPZ2023(store_β,store_Σt_inv,store_YY,store_YY_LF, M_zsp, z_vec, Sm_bit,store_Σt,var_list,freq_mix_tp, fdatesHF, fdatesLF);
    return out_struct
end


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
function makeHypSetup(::CPZ2023dev_type)
    return hypChan2020()
end


@doc raw"""
    BEAVARs.CPZ2023(dataHF_tab::TimeArray{Typ,N,D,A},dataLF_tab::TimeArray{Typ,N,D,A},varOrder::Array{Symbol,1},set_struct::BVARmodelSetup,hyp_struct::BVARmodelHypSetup)
    
    Estimate Chan, Zhu, Poon 2023 using a  Minnesota-based independent Normal-Wishart prior
"""
function CPZ2023dev(dataHF_tab::TimeArray{Typ,N,D,A},dataLF_tab::TimeArray{Typ,N,D,A},varOrder::Array{Symbol,1},set_struct::BVARmodelSetup,hyp_struct::BVARmodelHypSetup) where {Typ <: AbstractFloat, N, D, A <: AbstractArray{Typ, N}}
    @unpack p, n_burn,n_save, const_loc, n_fcst, prior_RW = set_struct
    ndraws = n_save+n_burn;
    nmdraws = 10;               # given a draw from the parameters to draw multiple time from the distribution of the missing data for better confidence intervals

    fdataHF_tab, z_tab, freq_mix_tp, datesHF, varNamesLF, fvarNames = BEAVARs.CPZ_prep_TimeArrays(dataLF_tab,dataHF_tab,varOrder,prior_RW,n_fcst)

    YYwNA = values(fdataHF_tab);
    YY = deepcopy(YYwNA);
    Tf,n = size(YY);
    
    B_draw, structB_draw, Σt_inv, b0 = BEAVARs.initParamMatrices(n,p,const_loc) 

    YYt, Y0, longyo, nm, H_B, H_B_CI, strctBdraw_LI, Σ_inv, Σt_LI, Σp_invsp, Σpt_ind, Xb, cB, cB_b0_LI, Smsp, Sosp, Sm_bit, Gm, Go, GΣ, Kym,Σt_ns_CI, Σpt_ind_CI = BEAVARs.CPZ_initMatrices_v2(YY,structB_draw,b0,Σt_inv,p);
    
    M_zsp, z_vec, T_z, MOiM, MOiz = BEAVARs.CPZ_makeM_inter(z_tab,YYt,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,nm,Tf);

    
    fdatesHF = timestamp(fdataHF_tab);
    fdatesLF = collect(timestamp(z_tab)[1]:Month(freq_mix_tp[2]):fdatesHF[end]);
    M_inter_agg = BEAVARs.CPZ_makeM_inter_agg(fdatesLF,fdatesHF,freq_mix_tp);
    
    # YY has missing values so we need to draw them once to be able to initialize matrices and prior values
    YYt = BEAVARs.CPZ_draw_wz!(YYt,longyo,Y0,cB,B_draw,structB_draw,strctBdraw_LI,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_inv,p,n,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,H_B_CI,nmdraws,Σt_ns_CI);
    
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
    μ_y = zeros(nm,);
    long_pr = similar(cB);
    
    pbsi_struct = PrecisionBasedSamplerInputs(structB_draw, H_B, Σt_inv, Σ_inv, cB, long_pr, μ_y, μ_yBar, mdraws, Gm, Go, GΣ,  Kym, KymBar, Smsp,Sosp,  MOiM,  MOiz);

    for ii in 1:ndraws
        # draw of the missing values
        BEAVARs.BEAVARs.CPZdev_draw_wz!(YYt,longyo,Y0,pbsi_struct,B_draw,strctBdraw_LI,Σt_LI,cB_b0_LI,p,n,Sm_bit,nm,H_B_CI,nmdraws,Σt_ns_CI);
        # BEAVARs.CPZ_draw_wz_lessAlloc!(YYt,longyo,Y0,cB,B_draw,structB_draw,strctBdraw_LI,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_inv,p,n,Sm_bit,Smsp,Sosp,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,KymBar,H_B_CI,nmdraws,μ_yBar,mdraws,draw_tmp)
        
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
    makeDataSetup(::CPZ2023dev_type,dataHF_tab::TimeArray, dataLF_tab::TimeArray; var_list =  [colnames(dataHF_tab); colnames(dataLF_tab)])

Generate data for a mixed-frequency VAR. Uses Time Arrays from the TimeSeries package
    
# Arguments
    dataHF_tab: TimeArray with your high-frequency variables (monthly or quarterly, respectively)
    dataLF_tab: TimeArray with your low-frequency variables (quarterly or yearly, respectively)
    var_list:   the variable order. Note that the functions that call these variables allow this to be optional.

See also `dataCPZ2023`.

"""
function makeDataSetup(::CPZ2023dev_type,dataHF_tab::TimeArray, dataLF_tab::TimeArray; var_list =  [colnames(dataLF_tab); colnames(dataHF_tab)])
    return dataCPZ2023(dataHF_tab, dataLF_tab, var_list)
end



@doc raw"""
    Draw with restrictions
"""
function CPZdev_draw_wz!(YYt,longyo,Y0,pbsi_struct,B_draw,strctBdraw_LI,Σt_LI,cB_b0_LI,p,n,Sm_bit,nm,H_B_CI,nmdraws,Σt_ns_CI);
    
    # updating cB
    BEAVARs.CPZdev_update_cB!(pbsi_struct.cB,B_draw[:,2:end],B_draw[:,1],Y0,cB_b0_LI,p,n)

    # updating H_B
    @views pbsi_struct.H_B[H_B_CI] = -pbsi_struct.structB_draw[strctBdraw_LI];
    # updating Σ_invFsp
    #  Σ_inv.nzval[:] = Σt_inv[Σt_LI];
    @views pbsi_struct.Σ_inv[Σt_ns_CI] = pbsi_struct.Σt_inv[Σt_LI];


    mul!(pbsi_struct.Gm,pbsi_struct.H_B,pbsi_struct.Smsp);
    mul!(pbsi_struct.Go,pbsi_struct.H_B,pbsi_struct.Sosp);
    mul!(pbsi_struct.GΣ,pbsi_struct.Gm',pbsi_struct.Σ_inv);
    mul!(pbsi_struct.Kym,pbsi_struct.GΣ,pbsi_struct.Gm);
    CL = cholesky(Hermitian(pbsi_struct.Kym));
    pbsi_struct.long_pr[:,] = (pbsi_struct.cB-pbsi_struct.Go*longyo);
    pbsi_struct.μ_y[:,] = CL.U\(CL.U'\(pbsi_struct.GΣ*pbsi_struct.long_pr));


    pbsi_struct.KymBar[:,:] = pbsi_struct.MOiM + pbsi_struct.Kym;
    CLBar = cholesky(Hermitian(pbsi_struct.KymBar))
    pbsi_struct.μ_yBar[:,] = CLBar.U\(CLBar.U'\(pbsi_struct.MOiz + pbsi_struct.Kym*pbsi_struct.μ_y))

    # mdraws = zeros(nm,nmdraws)
    for i_draw in 1:nmdraws
        pbsi_struct.mdraws[:,i_draw] = pbsi_struct.μ_yBar +  ldiv!(CLBar.U,randn(nm,))
    end    
    YYt[Sm_bit] = dropdims(median(pbsi_struct.mdraws,dims=2),dims=2);
    return YYt
end



struct PrecisionBasedSamplerInputs{T <: AbstractFloat, N} 
    structB_draw::Array{T, N}   
    H_B::Array{T, N}            
    Σt_inv::Array{T, N}         
    Σ_inv::Array{T, N} 
    cB::Array{T, 1}             
    long_pr::Array{T, 1}
    μ_y::Array{T, 1}            
    μ_yBar::Array{T,1}
    mdraws::Array{T,N}
    Gm::Array{T,N}
    Go::Array{T,N}
    GΣ::Array{T,N}
    Kym::Array{T,N}
    KymBar::Array{T,N}
    Smsp::Array{Bool,N}
    Sosp::Array{Bool,N}
    MOiM::Array{T,N}
    MOiz::Array{T,1}
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
function CPZdev_update_cB!(cB,Bmat,b0,Y0,cB_b0_LI,p,n)
    for io = 0:p-1
        ytmp = zeros(n,);
        for kk = 0:io
            @views ytmp1 = Bmat[:,1+(p-io+kk)*n-n:(p-io+kk)*n+n-n]*Y0[p-kk,:];  # This is \sum_1^p B_j y_{t-j}. For t = 0 : cB = b0 + B_p y0
            ytmp = ytmp+ytmp1;
        end
        ytmp = ytmp + b0;
        cB[n*(p-io)-n+1 : n*(p-io)-n+n,] = ytmp;
    end
    @views cB[n*p-n+1+n : end]=b0[cB_b0_LI]
    return cB
end
