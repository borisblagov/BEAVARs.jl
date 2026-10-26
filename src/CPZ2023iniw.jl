#-------------------------------------
# The Den: this is where the beavar lives
#-------------------------------------

@doc raw"""
    beavar(::CPZ2023iniw_type, set_struct::BVARmodelSetup, hyp_struct::BVARmodelHypSetup, data_struct::BVARmodelDataSetup)


"""
function beavar(::CPZ2023iniw_type, set_struct::BVARmodelSetup, hyp_struct::BVARmodelHypSetup, data_struct::BVARmodelDataSetup)
    println("Hello CPZ2023iniw")
    @unpack dataHF_tab,dataLF_tab, var_list = data_struct
    store_YY, store_YY_LF, store_β,  store_Σt, store_Σt_inv, store_h, store_s2_h,store_σ_h2,store_ρ, store_eh, store_c1, store_c2, M_zsp, z_vec, Sm_bit, freq_mix_tp, fdatesHF, fdatesLF,M_inter_agg = CPZ2023iniw(dataHF_tab,dataLF_tab,var_list,set_struct,hyp_struct);
    YYforHF_struct = makeProbabilityIntervalsStruct(store_YY,fdatesHF,var_list);
    YYforLF_struct = makeProbabilityIntervalsStruct(store_YY_LF,fdatesLF,var_list);
    data_flags_vec = fdatesLF .∈  Ref(timestamp(dataLF_tab));       # bit_vector showing the data position of the low-frequency variable
    forecast_flags_vec = .!data_flags_vec;                          # bit_vector showing the forecasted position. Only supports balanced z_tab #TODO: make it work for unbalanced z_tab
    out_struct = VAROutput_Blagov2027(store_YY,store_YY_LF,store_β,store_Σt_inv, M_zsp, z_vec, Sm_bit,freq_mix_tp,store_Σt, store_h, store_s2_h, store_ρ, store_σ_h2, store_eh, store_c1, store_c2, M_inter_agg, fdatesHF, fdatesLF, YYforLF_struct, YYforHF_struct,data_flags_vec,forecast_flags_vec);
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
function makeHypSetup(::CPZ2023iniw_type)
    return hypChan2020()
end


@doc raw"""
    BEAVARs.CPZ2023iniw(dataHF_tab::TimeArray{Typ,N,D,A},dataLF_tab::TimeArray{Typ,N,D,A},varOrder::Array{Symbol,1},set_struct::BVARmodelSetup,hyp_struct::BVARmodelHypSetup)
    
    Estimate a mixed-frequency VAR with a Minnesota-based independent Normal-Wishart prior using the precision based sampler from Chan, Zhu, Poon (2023), JoE
"""
function CPZ2023iniw(dataHF_tab::TimeArray{Typ,N,D,A},dataLF_tab::TimeArray{Typ,N,D,A},varOrder::Array{Symbol,1},set_struct::BVARmodelSetup,hyp_struct::BVARmodelHypSetup) where {Typ <: AbstractFloat, N, D, A <: AbstractArray{Typ, N}}
    @unpack p, n_burn,n_save, const_loc, n_fcst, prior_RW = set_struct
    ndraws = n_save+n_burn;
    nmdraws = 10;               # given a draw from the parameters to draw multiple time from the distribution of the missing data for better confidence intervals

    fdataHF_tab, z_tab, freq_mix_tp, datesHF, varNamesLF, fvarNames, HF_n_fcst = BEAVARs.CPZ_prep_TimeArrays(dataLF_tab,dataHF_tab,varOrder,prior_RW,n_fcst)
    fdatesHF = timestamp(fdataHF_tab);
    fdatesLF = collect(timestamp(z_tab)[1]:Month(freq_mix_tp[2]):fdatesHF[end]);
    M_inter_agg = BEAVARs.makeMInterAgg(fdatesLF,fdatesHF,freq_mix_tp);

    YYwNA   = values(fdataHF_tab);

    YY      = deepcopy(YYwNA);
    Tf,n    = size(YY);
    YYt     = (YY');                # this is just another name for YY, so if we update this, we update YY
    vYYt    = vec(YYt);             # this as well is connected all to YY

    k       = n*p+1;                #
    Sm_bit  = isnan.(YYwNA');       # we should only call this in the beginning
    So_bit  = .!isnan.(YYwNA');     # we should only call this in the beginning
    nm      = sum(Sm_bit);          # the number of missing values
    longyo  = vYYt[vec(So_bit)];    # observed values, should not change ever

    M_zsp, z_vec, T_z, MOiM, MOiz = BEAVARs.makeM_intertemp(z_tab,YYt,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,nm,Tf);

    Y0 = @views YY[1:p,:];     # this is just a view, so if we update YY, we update this
    if any(isnan.(Y0))
        Y0[isnan.(Y0)]=zeros(size(Y0[isnan.(Y0)],1)); # for the first pass only remove NaNs for zeroes
        # consider changing this and drawing them somehow maybe? Note that they are drawn again 
        # print("NaNs found in Y0, replaced with zeros");
    end


    # initialize the precision based sampler structure
    pbsi_struct = BEAVARs.init_pbsi_struct(YYt, Sm_bit, So_bit, n,p,const_loc,Tf, nm, nmdraws, MOiM, MOiz)

    # draw with the precision based sampler with uninformative parameters (zeros) to fill missings for the parameter initialization
    BEAVARs.pbsi_draw_wz!(YYt,longyo,Y0,pbsi_struct,p,n,Sm_bit,nm,nmdraws,Tf);


    # initialize the priors, drop any forecast to not use it for setting priors
    ~, X, T, ~, sigmaP_ar4, betaOLS_ar4, priorMinn_struct, β_hat_bb_struct, csv_Σt_struct, csv_h_struct = BEAVARs.init_Minn_csv(YY[1:end-HF_n_fcst,:],set_struct,hyp_struct);
    update_priorMinn!(priorMinn_struct,hyp_struct)

    # make a view to the actual dependent variables. These will be updated as we update YYt
    YYs     = @view YY[1:end-HF_n_fcst,:];              # drop the forecast to not use it for priors or parameter estimation
    Y       = @view YYs[p+1:end,:];
    vecYt   = vec(Y');

    # prepare matrices for saving
    store_YY    = zeros(Tf,n,n_save);
    store_β = zeros(n^2*p+n,n_save);
    store_Σt = zeros(n,n,n_save);
    store_Σt_inv = zeros(n,n,n_save);
    store_h = zeros(T,n_save);
    store_s2_h = zeros(T,n_save);
    store_ρ = zeros(n_save,);
    store_σ_h2 = zeros(n_save,); 
    store_eh = zeros(T,n_save);
    store_c1 = zeros(n_save,);
    store_c2 = zeros(n_save,);

    for ii = 1:ndraws
        # updating pbsi_struct with Σt_inv.
        BEAVARs.update_Σf_inv_bb!(pbsi_struct.Σf_inv_bb,csv_Σt_struct.Σt_inv,Tf,n);
        
        # update pbsi_struct with β 
        pbsi_struct.B_draw .= reshape(β_hat_bb_struct.β_draw,k,n)';
        @views pbsi_struct.structB_draw[:,n+1:end] .= pbsi_struct.B_draw[:,2:end];

        # Draw YYt using the precision based sampler, updating YYs, Y, vecYt
        BEAVARs.pbsi_draw_wz!(YYt,longyo,Y0,pbsi_struct,p,n,Sm_bit,nm,nmdraws,Tf);

        # update X by mutation, exclude the forecast from the 
        mlagL!(YYs,X,p,n);

        # update β_hat_bb_struct with Σt_inv for drawing the parameters. Xsur is updated in the draw_β func
        # β_hat_bb_struct.Xsur.X .= X;      # Xsur  
        BEAVARs.update_Σf_inv_bb!(β_hat_bb_struct.Σ_inv_bb,csv_Σt_struct.Σt_inv,T,n);  

        # draw β
        BEAVARs.Chan2020_draw_βsur_bb_NoPrUpd!(priorMinn_struct,β_hat_bb_struct,X,vecYt);

        # draw Σt and calculate Σt_inv
        BEAVARs.CCK2016_drawΣtsur!(vecYt,β_hat_bb_struct,csv_Σt_struct,csv_h_struct,n,T);



        if ii>n_burn
            store_YY[:,:,ii-n_burn]  = YY;
            store_β[:,ii-n_burn] = β_hat_bb_struct.β_draw;
            store_Σt_inv[:,:,ii-n_burn] = csv_Σt_struct.Σt_inv;
            store_Σt[:,:,ii-n_burn] = csv_Σt_struct.Σt;
            store_h[:,ii-n_burn] = csv_h_struct.h;
            store_s2_h[:,ii-n_burn] = csv_h_struct.s2_h;
            store_ρ[ii-n_burn,] = csv_h_struct.ρ_vec[1];
            store_σ_h2[ii-n_burn,] = csv_h_struct.σ_h2_vec[1];
            store_eh[:,ii-n_burn] = csv_h_struct.eh;
        end

    end
    store_YY_LF = mapslices(x->M_inter_agg*x,store_YY,dims=1:2);
    return store_YY, store_YY_LF, store_β, store_Σt, store_Σt_inv, store_h, store_s2_h, store_ρ, store_σ_h2, store_eh, store_c1, store_c2, M_zsp, z_vec, Sm_bit, freq_mix_tp, fdatesHF, fdatesLF, M_inter_agg
end


# Data functions
@doc raw"""
    makeDataSetup(::CPZ2023iniw_type,dataHF_tab::TimeArray, dataLF_tab::TimeArray; var_list =  [colnames(dataHF_tab); colnames(dataLF_tab)])

Generate data for a mixed-frequency VAR. Uses Time Arrays from the TimeSeries package
    
# Arguments
    dataHF_tab: TimeArray with your high-frequency variables (monthly or quarterly, respectively)
    dataLF_tab: TimeArray with your low-frequency variables (quarterly or yearly, respectively)
    var_list:   the variable order. Note that the functions that call these variables allow this to be optional.

See also `data_MFVAR_2f`.

"""
function makeDataSetup(::CPZ2023iniw_type,dataHF_tab::TimeArray, dataLF_tab::TimeArray; var_list =  [colnames(dataLF_tab); colnames(dataHF_tab)])
    return data_MFVAR_2f(dataHF_tab, dataLF_tab, var_list)
end



@doc raw"""
    Draw with restrictions
"""
function pbsi_draw_wz!(YYt,longyo,Y0,pbsiNew_struct,p,n,Sm_bit,nm,nmdraws,Tf);
    @unpack structB_draw, B_draw, HB,Σf_inv_bb, cB, long_pr, μ_y, μ_yBar, mdraws, Gm, Go, GΣ, Kym, KymBar, Smsp, Sosp, MOiM, MOiz = pbsiNew_struct
    
    # updating cB
    update_cB!(cB,B_draw,Y0,Tf-p,n,p);

    # updating H_B
    HB.mBI .= -structB_draw;

    mul!(Gm,HB,Smsp);
    mul!(Go,HB,Sosp);
    mul!(GΣ,Gm',Σf_inv_bb);
    mul!(Kym,GΣ,Gm);   
    
    CL = cholesky(Hermitian(Kym));
    mul!(long_pr,Go,longyo);
    long_pr .= cB .- long_pr;
    μ_y .= ldiv!(CL.U,ldiv!(CL.U',mul!(μ_y,GΣ,long_pr)));   # eq. 12
  

    KymBar .= MOiM .+ Kym;      
    CLBar = cholesky!(Hermitian(KymBar))
    μ_yBar .= ldiv!(CLBar.U,ldiv!(CLBar.U',MOiz .+ mul!(μ_yBar,Kym,μ_y)));   # eq. 12

    for i_draw in 1:nmdraws
        mdraws[:,i_draw] = μ_yBar +  ldiv!(CLBar.U,randn(nm,))
    end    
    YYt[Sm_bit] = dropdims(median(mdraws,dims=2),dims=2);
end



struct PrecisionBasedSamplerInputs_dev{T <: AbstractFloat, N} 
    structB_draw::Array{T, N}   
    B_draw::Array{T,N}
    HB::HBMatrix{T, Array{T,N}}    
    Σf_inv_bb::BlockBandedMatrix{T}  
    cB::Array{T, 1}             
    long_pr::Array{T, 1}        # temporary array
    μ_y::Array{T, 1}            
    μ_yBar::Array{T,1}
    mdraws::Array{T,N}
    Gm::SparseMatrixCSC{T,Int}
    Go::SparseMatrixCSC{T,Int}
    GΣ::BlockedArray{T}
    Kym::Array{T,N}
    KymBar::Array{T,N}
    Smsp::SparseMatrixCSC{Bool,Int}
    Sosp::SparseMatrixCSC{Bool,Int}
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




@doc raw"""
    init_pbsi_struct(n,p,const_loc,Tf)

    We will use this function to compare to the newer iteration whether we can get performance gains as well as for correctness
"""
function init_pbsi_struct(YYt, Sm_bit, So_bit, n, p, const_loc, Tf, nm, nmdraws, MOiM, MOiz)
    
    # Initialize parameter matrices
    B_draw, structB_draw, Σt_inv, b0 = BEAVARs.initUninformParameters(n,p,const_loc) 
    mBI = -structB_draw;
    HB = BEAVARs.HBMatrix(mBI,Tf);
    Σf_inv_bb = BlockBandedMatrix{Float64}(undef,ones(Int,Tf,).*n, ones(Int,Tf,).*n, (0, 0));
    expmhf = ones(Tf,);                     # expmh and correspondingly h in csv_h_struct are only of size T, whereas for the PBS we need T+p+hf_n_fcast
    BEAVARs.update_Σf_inv_bb!(Σf_inv_bb,Σt_inv,expmhf,Tf,n);

    # initialize matrices for the precision based sampler

    indC_nan_wide = findall(Sm_bit) #  Cartesian indices of missing values
    # indC_non_wide = findall(!isnan,YYt)  # Cartesian indices of not missing values
    indC_non_wide = findall(So_bit)  # Cartesian indices of not missing values

    # convert between linear and cartesian indices
    indL_all = LinearIndices(YYt);
    indL_nan_wide = indL_all[indC_nan_wide] # are the linear indices of NaN values
    indL_non_wide = indL_all[indC_non_wide] # are the linear indices of non NaN values
        

    S_full = I(Tf*n);
    Sm = S_full[:,indL_nan_wide]; # Sm, selection matrix selecting the missing values
    So = S_full[:,indL_non_wide]; # So, selection matrix selecting hte observed values
    Smsp = sparse(Sm);            # sparse Sm
    Sosp = sparse(So);            # sparse So
    cB      = repeat(b0,Tf);
    Gm      = HB*Smsp; 
    Go      = HB*Sosp; 
    GΣ      = Gm'*Σf_inv_bb; 
    Kym     = GΣ*Gm; # we can initialize all these and then mutate with mul!()
    long_pr = similar(cB);
    μ_y     = zeros(nm,);
    μ_yBar  = zeros(nm,);
    KymBar  = similar(Kym);
    mdraws  = zeros(nm,nmdraws)

    pbsi_struct = BEAVARs.PrecisionBasedSamplerInputs_dev(structB_draw, B_draw, HB, Σf_inv_bb, cB, long_pr, μ_y, μ_yBar, mdraws, Gm, Go, GΣ,  Kym, KymBar, Smsp,Sosp,  MOiM,  MOiz);

    return pbsi_struct
end



"""
    update_Σf_inv_bb!(Σf_inv_bb,Σt_inv,Tf,n)

    Updates a large block-banded diagonal matrix with new identical blocks across the diagonal with minimal allocations.

    Σf_inv_bb = [Σtinv    0    0   ;
                   0   Σtinv   0   ;
                   0    0    Σtinv ;
                ]
"""
function update_Σf_inv_bb!(Σf_inv_bb,Σt_inv,Tf,n)
    diagonal_block = vec(Σt_inv)
    for ii in 1:Tf
        copyto!(Σf_inv_bb.data, (ii - 1)*n*n + 1, diagonal_block, 1, n*n)
    end
end



"""
    update_Σf_inv_bb!(Σf_inv_bb,Σt_inv,expmhf,Tf,n)

    Updates a large block-banded diagonal matrix with new blocks across the diagonal, multiplied by exp.(-h[block#]). 
    Used for common stochastic volatility.

    Σf_inv_bb = [Σtinv.*h[1]    0          0   ;
                   0        Σtinv.*h[2]    0   ;
                   0            0      Σtinv.*h[3] ;
                ]
"""
function update_Σf_inv_bb!(Σf_inv_bb,Σt_inv,expmhf,Tf,n)
    diagonal_block = vec(Σt_inv)
    for ii in 1:Tf
        copyto!(Σf_inv_bb.data, (ii - 1)*n*n + 1, diagonal_block.*expmhf[ii], 1, n*n)
    end
end




@doc raw"""
    update_cB!()

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
function update_cB!(cB,B_draw,Y0,T,n,p)
    b0 = @view B_draw[:,1];
    Bmat = @view  B_draw[:,2:end];
    for io = 0:p-1
        ytmp = zeros(n,);
        for kk = 0:io
            @views ytmp1 = Bmat[:,1+(p-io+kk)*n-n:(p-io+kk)*n+n-n]*Y0[p-kk,:];  # This is \sum_1^p B_j y_{t-j}. For t = 0 : cB = b0 + B_p y0
            ytmp = ytmp+ytmp1;
        end
        ytmp = ytmp + b0;
        cB[n*(p-io)-n+1 : n*(p-io)-n+n,] = ytmp;
    end

    for ii in 1:T
        copyto!(cB, n*p + 1 + n*(ii - 1), b0, 1, n)
    end
    return cB
end




"""
    B_draw, structB_draw, Σt_inv, b0 = initUninformParameters(n,p,intercept) 
    
    Initializes paramater matrices for a VAR of the form
        B_0 y_t = b0 + B_1 y_{t-1} + ... + B_p y_{t-1} + varepsilon_t,     varepsilon_t sim N(0,Σt)
        
        B_0 is an identity matrix
        B_1 is an identity matrix
        b0, B_2, ..., B_p are zeroes

    Returns:    
        B_draw = [b0, B1, ..., Bp]; if intercept == 0, b0 is at the and and if set to -1, there is no intercept
        structB_draw = [B_0, B1, ..., Bp]
        Σt_inv = Σt^{-1}

"""
function initUninformParameters(n::Int,p::Int,intercept::Int) 
    if intercept == 1
        B_draw = [zeros(n,)  1.0*I(n) zeros(n,n*(p-1))]; b0 = B_draw[:,1];
        structB_draw = [-1.0*I(n) B_draw[:,2:end]]
    elseif intercept == 0
        B_draw = [1.0*I(n) zeros(n,n*(p-1)) zeros(n,) ]; b0 = B_draw[:,end];
        structB_draw = [-1.0*I(n) B_draw[:,1:end-1]]
    elseif intercept == -1
        B_draw = [1.0*I(n) zeros(n,n*(p-1))]; b0 = ones(n,).*NaN;
        structB_draw = [-1.0*I(n) B_draw]
    end
    Σt  = .001*Matrix(I,n,n); Σt_inv = inv(Σt)
    return B_draw, structB_draw, Σt_inv, b0
end