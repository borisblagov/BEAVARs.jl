function makeHypSetup(::CPZ2023_type)
    return hypChan2020()
end

# Structure for the datasets and the frequency mix
@doc raw"""
    dataCPZ2023(data_HF::TimeArray,data_LF::TimeArray,aggMix::Int,var_list::Array{Symbol,1})

Generate a dataset strcture for use with CPZ2023 model

# Arguments
    dataHF_tab: TimeArray with your high-frequency variables (monthly or quarterly, respectively)
    dataLF_tab: TimeArray with your low-frequency variables (quarterly or yearly, respectively)
    aggMix:     0 for data in growth rates, 1 for log-levels. Determines the weights how high freq. variables fit with low-frequency ones. Will use averages for log-levels or Mariano and Murasawa (2010) weights for growth rates 
    var_list:   the variable order. Note that the functions that call these variables allow this to be optional.

See also `makeDataSetup`.
"""
@with_kw struct dataCPZ2023 <: BVARmodelDataSetup
    dataHF_tab::TimeArray                                       # data for the high-frequency variables
    dataLF_tab::TimeArray                                       # data for the low-frequency variables
    aggMix::Int                                                 # 0: growth rates, 1: log-levels. indicator for the aggregate weights in the inter-temporal aggregation
    var_list::Array{Symbol,1}                                   # Symbol vector with the variable names, will be used for ordering
end

@doc raw"""
    makeDataSetup(::CPZ2023_type,dataHF_tab::TimeArray, dataLF_tab::TimeArray, aggMix::Int; var_list =  [colnames(dataHF_tab); colnames(dataLF_tab)])

Generate data for a mixed-frequency VAR. Uses Time Arrays from the TimeSeries package
    
# Arguments
    dataHF_tab: TimeArray with your high-frequency variables (monthly or quarterly, respectively)
    dataLF_tab: TimeArray with your low-frequency variables (quarterly or yearly, respectively)
    aggMix:     0 for data in growth rates, 1 for log-levels. Determines the weights how high freq. variables fit with low-frequency ones. Will use averages for log-levels or Mariano and Murasawa (2010) weights for growth rates 
    var_list:   the variable order. Note that the functions that call these variables allow this to be optional.

See also `dataCPZ2023`.

"""
function makeDataSetup(::CPZ2023_type,dataHF_tab::TimeArray, dataLF_tab::TimeArray, aggMix::Int; var_list =  [colnames(dataHF_tab); colnames(dataLF_tab)])
    return dataCPZ2023(dataHF_tab, dataLF_tab, aggMix, var_list)
end



@doc raw"""
    varOrder must be a `Vector{Symbol}` and not `Vector{Vector{Symbol}}`
    e.g. [varNamesLF; varNamesHF] and not [varNamesLF, varNamesHF]
    aggMix = 0: growth rates, 1: log-levels. indicator for the aggregate weights in the inter-temporal aggregation
"""
function CPZ_prep_TimeArrays(dataLF_tab,dataHF_tab,varOrder,aggMix)
    varNamesLF = colnames(dataLF_tab)
    # z_tab = dataLF_tab[.!isnan.(dataLF_tab)];
    z_tab = dataLF_tab;
    # add the z_tab as NaN values in the high-frequency tab
    fdataHF_tab = merge(dataHF_tab,map((timestamp, values) -> (timestamp, values.*NaN), z_tab[varNamesLF]),method=:outer)
    fdataHF_tab = fdataHF_tab[varOrder]              # ordering the variables as the user wants them
    fvarNames = colnames(fdataHF_tab)                # full list of the variable names
    datesHF = timestamp(fdataHF_tab)
    datesLF = timestamp(z_tab)
    freqL_date = Month(datesLF[2])-Month(datesLF[1]) # looks whether the data is quarterly or yearly
    freqH_date = Month(datesHF[2])-Month(datesHF[1])

    if freqL_date==Month(0)
        freqL_date = Month(12)
    end 
    # tuple showing the specification: 1, 3, 12 are monthly quarterly, annually and 0,1 is growth rates or log-levels
    freq_mix_tp = (convert(Int,freqH_date/Month(1)), convert(Int,freqL_date/Month(1)),aggMix) # tuple with the high and low frequencies. 1 is monthly, 3 is quarterly, 12 is annually
    return fdataHF_tab, z_tab, freq_mix_tp, datesHF, varNamesLF, fvarNames
end


@doc raw"""
    CPZ_makeM_inter(z_tab,YYt,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,nm,Tf;scVal=10e-8)

Generates an M matrix such that `z = My + eps`. `y` is a vector of unobserved high-frequency values and the observations in `z` relate by a linear combination to a subset of the values in `y`

# Arguments
    z_tab:      A `TimeArray` of low-frequncy observations, either yearly or quarterly
    YYt:        A transposed vector of data. Only its dimensions matter, not what is in it. It is assumed it has some unobserved values which are observed only at the lower frequency (`z_tab`)
    Sm_bit:     A boolean matrix indicating which indices in YYt are to be matched, i.e. which are the missing high-frequency observations
    datesHF:    A TimeArray accompanying `YYt`
    varNamesLF: Vector with variable names from `z_tab` to be matched with a the high-frequency data. Code will err if varNamesLF is not a subset of fvarNames
    fvarNames:  Vector with variable names from `y` to be matched with a the low-frequency data. Code will err if varNamesLF is not a subset of fvarNames
    freq_mix_tp:Tuple indicating what is the frequency mix: monthly, quarterly, yearly, growth rates, log-levels
    nm:         Total number of missing values  (consider dropping this, because we have Sm_bit)
    Tf:         Total number of high-frequency observations (Tf = length(YY), Tf = size(YYt,2), consider dropping this as input, as we have YYt in the function)
    scVal:      Optional parameter, value of how "hard" the constraint is. Setting it to very small values enforces z = My, while setting it higher allows for some discrepancies

!!! note "Important"
    The function currently accepts only one variable in z_tab, has to be extended. If you have both GDP and Consumption which are observed in LF and Z_tab has 2 columns, the O-matrix will be wrong, see below

#TODO        
"""
function CPZ_makeM_inter(z_tab,YYt,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,nm,Tf;scVal=10e-8)
    
    z_var_pos  = indexin(varNamesLF,fvarNames); # positions of the variables in z
    if length(size(z_tab)) == 1
        T_z = size(z_tab,1)
        n_z = 1
    else
        T_z, n_z = size(z_tab);  
    end
    M_z = zeros(T_z*n_z,nm)
    z_vec = zeros(T_z*n_z,)
    iter = CartesianIndices(YYt)
    flagFirstRow = zeros(n_z,);                              # if we don't have a full quarter/year we will not be able to have a hard constraint in the beginning, set the error to a higher value

    for ii_z = 1:n_z # iterator going through each variable in z_tab (along the columns)
        
        datesLF_ii = timestamp(z_tab[varNamesLF[ii_z]])
        ym_ci = iter[Sm_bit]                                # a vector of y_m with cartesian indices of the missing values in YYt
        z_ci = CartesianIndices((z_var_pos[ii_z]:z_var_pos[ii_z],1:Tf))
        
        z_Mind_vec_ii = vec(sum(ym_ci.==z_ci,dims=2))       # alternative z_Mind_vec=vec(indexin(ym_ci,z_ci)).!==nothing
        M_inter_ii = zeros(T_z,nm)
        M_z_ii = @views M_inter_ii[:,z_Mind_vec_ii.==1]
    
        if size(datesHF,1)!==size(M_z_ii,2)
            # error("The size of M does not match the number of dates available in z_tab. Maybe the low-frequency data is longer? The problem is with variable number ", z_var_pos[ii_z])
        end
    
        # we need to watch out with the dates due to how the intertemporal constraint works Take for example growth rates Q and M
        # y_t = 1/3 y_t - 2/3 y_{t-1} \dots - - 2/3 y_{t-3} - 1/3 y_{t-5}
        # Intuitively, Q1 quarterly GDP (e.g. 01.01.2000) is the weighted sum of the monthly March, February, January, December, November, and October
        # if y_t^Q is 01.01.2000, we need +2 and -2 months for the weights
        if freq_mix_tp==(1,3,0)
            hfWeights = [1/3; 2/3; 3/3; 2/3; 1/3]; n_hfw = size(hfWeights,1); #number of weights, depends on the variable transformation and frequency
            hf_num1 = 1; hf_num2 = 1;  # this solves the range below ii_M-div((n_hfw-hf_num1),2): ii_M+div((n_hfw-hf_num2),2). This should give the indices -2, -1, 0, +1, +2
        elseif freq_mix_tp==(3,12,0)
            # quarterly and yearly data with growth rates
            hfWeights = [1/4; 2/4; 3/4; 1; 3/4; 2/4; 1/4]; n_hfw = size(hfWeights,1); #number of weights, depends on the variable transformation and frequency
            hf_num1 = 1; hf_num2 = 1;  # this solves the range below ii_M-div((n_hfw-hf_num1),2): ii_M+div((n_hfw-hf_num2),2). This should give the indices -3, -2, -1, 0, +1, +2, +3
        elseif freq_mix_tp==(1,3,1)
            hfWeights = [1/3; 1/3; 1/3]; n_hfw = size(hfWeights,1); #number of weights, depends on the variable transformation and frequency
            hf_num1 = 3; hf_num2 = -1;  # this solves the range below ii_M-div((n_hfw-hf_num),2): ii_M+div((n_hfw-hf_num),2). This should give the indices -0, +1, +2
        elseif freq_mix_tp==(3,12,1)
            hfWeights = [1/4; 1/4; 1/4; 1/4]; n_hfw = size(hfWeights,1); #number of weights, depends on the variable transformation and frequency
            hf_num1 = 4; hf_num2 = -3;  # this solves the range below ii_M-div((n_hfw-hf_num),2): ii_M+div((n_hfw-hf_num),2). This should give the indices -0, +1, +2
        else
            error("This combination of frequencies and transformation has not been implemented")
        end
    
        for ii_zi in eachindex(datesLF_ii) # iterator going through each time point in datesHF
            if ii_zi == 1 # check if we have a full quarter/year in the beginning, otherwise we will try to acces negative indices in the matrix M
                ii_M = findall(datesHF.==datesLF_ii[ii_zi])[1]       # find the low-frequency index that corresponds to the high-frequency missing value
                MrowRange = ii_M-div((n_hfw-hf_num1),2): ii_M+div((n_hfw-hf_num2),2);
                start_value = max(1, MrowRange[1]); stop_value = MrowRange[end]; positive_range = start_value:stop_value
                M_z_ii[ii_zi,start_value:stop_value]=hfWeights[MrowRange.>0];
                flagFirstRow[ii_z] = 1;
            else
                ii_M = findall(datesHF.==datesLF_ii[ii_zi])[1]       # find the low-frequency index that corresponds to the high-frequency missing value
                # M_z_ii[ii_zi, findall(datesHF.==datesLF_ii[ii_zi])[1]-n_hfw+1:findall(datesHF.==datesLF_ii[ii_zi])[1]] = hfWeights # if shifted above
                M_z_ii[ii_zi,ii_M-div((n_hfw-hf_num1),2): ii_M+div((n_hfw-hf_num2),2)]=hfWeights; # +2 and - 2 months for the weights or +3 and -3
            end
        end
        M_z[(ii_z-1)*T_z + 1:T_z + (ii_z-1)*T_z,:] = M_inter_ii;    # , should be fixed by stacking the M_inter_ii
        z_vec[(ii_z-1)*T_z + 1:T_z + (ii_z-1)*T_z,]  = values(z_tab[varNamesLF[ii_z]]);
    end
    M_zsp = M_z;                        # Leftover from when trying sparse matrices, should be fixed by stacking the M_inter_ii
    O_zsp = Matrix(I,T_z,T_z).*scVal;   # this works only if we have one z variable (with T_z length)
    if flagFirstRow[1] == 1             # this here is only half-baked. We need to iterate over n_z variables and find their corresponding T_z lenghts
        O_zsp[1,1] = scVal*1000
    end
    MOiM = M_zsp'*(O_zsp\M_zsp);
    MOiz = M_zsp'*(O_zsp\z_vec);
    return M_zsp, z_vec, T_z, MOiM, MOiz
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
function CPZ_initMatrices(YY,structB_draw,b0,Σt_inv,p)
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
    S_full = I(Tf*n);
    Sm = S_full[:,indL_nan_wide]; # Sm, selection matrix selecting the missing values
    So = S_full[:,indL_non_wide]; # So, selection matrix selecting hte observed values
    Smsp = sparse(Sm);            # sparse Sm
    Sosp = sparse(So);            # sparse So
    
    # Initialize matrices
    H_Bsp, strctBdraw_LI = BEAVARs.makeBlkDiag(Tfn,n,p, -structB_draw);
    H_B, H_B_CI, strB2HB_ind = BEAVARs.makeBlkDiag_ns(Tfn,n,p, -structB_draw);
    Σ_invsp, Σt_LI = BEAVARs.makeBlkDiag(Tfn,n,0,Σt_inv);
    Σ_inv, Σt_ns_LI = BEAVARs.makeBlkDiag_ns(Tfn,n,0,Σt_inv);   # make a non-sparse matrix if needed                         # this is ( I(Tf*n) ⊗ Σ-1 )
    Σp_invsp, Σpt_ind = BEAVARs.makeBlkDiag(Tfn-n*p,n,0,Σt_inv);                # this is ( I(T*n) ⊗ Σ-1 ), the difference is that this includes the 0,-1,...,-p lags
    cB_b0_LI = repeat(1:n,div(Tfn-n*p-n+1+n,n));  # this repeats [1:n] so that we can update cB[indicesAfter Y_0,Y_{-1}, ..., Ymp] = b0[cB_b0_LI]
    Xb = sparse(Matrix(1.0I, Tfn, Tfn))
    cB = repeat(b0,Tf);
    
    Gm = H_B*Smsp; Go = H_B*Sosp; GΣ = Gm'*Σ_invsp; Kym = GΣ*Gm; # we can initialize all these and then mutate with mul!()
    
    return YYt, Y0, longyo, nm, H_B, H_B_CI, strctBdraw_LI, Σ_invsp, Σt_LI, Σp_invsp, Σpt_ind, Xb, cB, cB_b0_LI, Smsp, Sosp, Sm_bit, Gm, Go, GΣ, Kym;
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
function CPZ_draw_wz!(YYt,longyo,Y0,cB,B_draw,structB_draw,sBd_ind,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,H_B_CI,nmdraws)
    # updating cB
    BEAVARs.CPZ_update_cB!(cB,B_draw[:,2:end],B_draw[:,1],Y0,cB_b0_LI,p,n)

    # updating H_B
    H_B[H_B_CI] = -structB_draw[sBd_ind];
    # updating Σ_invFsp
    Σ_invsp.nzval[:] = Σt_inv[Σt_LI];

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




@doc raw"""
    Updates parameters using an independennt Normal-Wishart prior
"""
function CPZ_iniw!(YY,p,hypSetup,n,k,b0,B_draw,Σt_inv,structB_draw,Σp_invsp,Σpt_ind,Y,X,T,mu_prior,deltaP,sigmaP,intercept,Xsur_den,Xsur_CI,X_CI,XtΣ_inv_den,XtΣ_inv_X,V_Minn_inv,V_Minn_inv_elview,upd_these_vec,K_β,beta)
    Y, X = mlagL!(YY,Y,X,p,n)
    (deltaP, sigmaP, mu_prior) = BEAVARs.updatePriors3!(Y,X,n,mu_prior,deltaP,sigmaP,intercept,upd_these_vec);
    S_0 = Diagonal(sigmaP);
    beta_Minn = zeros(n^2*p+n);
    idx_kappa1,idx_kappa2, V_Minn_vec = prior_Minn(n,p,sigmaP,hypSetup);
    V_Minn_vec_inv = 1.0./V_Minn_vec;
    Σp_invsp.nzval[:] = Σt_inv[Σpt_ind];    
    Xsur_den[Xsur_CI] = X[X_CI]; 
    V_Minn_inv_elview[:] = V_Minn_vec_inv;  # update the diagonal of V_Minn_inv, i.e. V_Minn^-1

    beta = BEAVARs.Chan2020_drawβ(Σp_invsp,Xsur_den,XtΣ_inv_den,XtΣ_inv_X,V_Minn_inv,beta_Minn,K_β,Y,n,k);
    

    B_draw[:,:] = reshape(beta,k,n)'
    b0[:] = B_draw[:,1]
    structB_draw[:,n+1:end] = B_draw[:,2:end]


    # errors 
    Σt, Σt_inv = Chan2020_drawΣt(Y,Xsur_den,beta,n,T,S_0,hypSetup.nu0);

    return beta,b0,B_draw,Σt_inv,structB_draw,Σt
end




@doc raw"""
    Estimate Chan, Zhu, Poon 2024 using a  Minnesota-based independent Normal-Wishart prior
"""
function CPZ2023(dataHF_tab,dataLF_tab,varList,varSetup,hypSetup,aggMix)
    @unpack p, nburn,nsave, const_loc = varSetup
    ndraws = nsave+nburn;
    nmdraws = 10;               # given a draw from the parameters to draw multiple time from the distribution of the missing data for better confidence intervals

    fdataHF_tab, z_tab, freq_mix_tp, datesHF, varNamesLF, fvarNames = BEAVARs.CPZ_prep_TimeArrays(dataLF_tab,dataHF_tab,varList,aggMix)

    YYwNA = values(fdataHF_tab);
    YY = deepcopy(YYwNA);
    Tf,n = size(YY);
    
    B_draw, structB_draw, Σt_inv, b0 = BEAVARs.initParamMatrices(n,p,const_loc) 

    YYt, Y0, longyo, nm, H_B, H_B_CI, strctBdraw_LI, Σ_invsp, Σt_LI, Σp_invsp, Σpt_ind, Xb, cB, cB_b0_LI, Smsp, Sosp, Sm_bit, Gm, Go, GΣ, Kym = BEAVARs.CPZ_initMatrices(YY,structB_draw,b0,Σt_inv,p);
    
    M_zsp, z_vec, T_z, MOiM, MOiz = BEAVARs.CPZ_makeM_inter(z_tab,YYt,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,nm,Tf);

    # YY has missing values so we need to draw them once to be able to initialize matrices and prior values
    YYt = BEAVARs.CPZ_draw_wz!(YYt,longyo,Y0,cB,B_draw,structB_draw,strctBdraw_LI,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,H_B_CI,nmdraws);
    
    # we will be updating the priors for variables with many missing observations (>25%)
    updP_vec = sum(Sm_bit,dims=2).>size(Sm_bit,2)*0.25;
    
    # Initialize matrices for updating the parameter draws from CPZ_iniv  
    # ------------------------------------
    Y, X, T, deltaP, sigmaP, mu_prior, V_Minn_inv, V_Minn_inv_elview, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, K_β, beta, intercept = CPZ_initMinn(YY,p)
    

    # prepare matrices for storage
    store_YY    = zeros(Tf,n,nsave);
    store_β     = zeros(n^2*p+n,nsave);
    store_Σt_inv= zeros(n,n,nsave);
    store_Σt    = zeros(n,n,nsave);

    for ii in 1:ndraws
        # draw of the missing values
        BEAVARs.CPZ_draw_wz!(YYt,longyo,Y0,cB,B_draw,structB_draw,strctBdraw_LI,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,H_B_CI,nmdraws);
        
        # draw of the parameters
        beta,b0,B_draw,Σt_inv,structB_draw,Σt = BEAVARs.CPZ_iniw!(YY,p,hypSetup,n,k,b0,B_draw,Σt_inv,structB_draw,Σp_invsp,Σpt_ind,Y,X,T,mu_prior,deltaP,sigmaP,const_loc,Xsur_den,Xsur_CI,X_CI,XtΣ_inv_den,XtΣ_inv_X,V_Minn_inv,V_Minn_inv_elview,updP_vec,K_β,beta);

        if ii>nburn
            store_β[:,ii-nburn]  = beta;
            store_YY[:,:,ii-nburn]  = YY;
            store_Σt_inv[:,:,ii-nburn]    = Σt_inv;
            store_Σt[:,:,ii-nburn] = Σt;
        end
    end

    return store_YY,store_β, store_Σt_inv, M_zsp, z_vec, Sm_bit, store_Σt, freq_mix_tp
end


function CPZ2023n(YYwNA, z_tab, freq_mix_tp, datesHF, varNamesLF, fvarNames,varSetup,hypSetup)
    @unpack p, nburn,nsave, const_loc = varSetup
    ndraws = nsave+nburn;
    nmdraws = 10;               # given a draw from the parameters to draw multiple time from the distribution of the missing data for better confidence intervals


    YY = deepcopy(YYwNA);
    Tf,n = size(YY);
    
    B_draw, structB_draw, Σt_inv, b0 = BEAVARs.initParamMatrices(n,p,const_loc) 

    YYt, Y0, longyo, nm, H_B, H_B_CI, strctBdraw_LI, Σ_invsp, Σt_LI, Σp_invsp, Σpt_ind, Xb, cB, cB_b0_LI, Smsp, Sosp, Sm_bit, Gm, Go, GΣ, Kym = BEAVARs.CPZ_initMatrices(YY,structB_draw,b0,Σt_inv,p);
    
    M_zsp, z_vec, T_z, MOiM, MOiz = BEAVARs.CPZ_makeM_inter(z_tab,YYt,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,nm,Tf);

    # YY has missing values so we need to draw them once to be able to initialize matrices and prior values
    YYt = BEAVARs.CPZ_draw_wz!(YYt,longyo,Y0,cB,B_draw,structB_draw,strctBdraw_LI,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,H_B_CI,nmdraws);
    
    # we will be updating the priors for variables with many missing observations (>25%)
    updP_vec = sum(Sm_bit,dims=2).>size(Sm_bit,2)*0.25;
    
    # Initialize matrices for updating the parameter draws from CPZ_iniv  
    # ------------------------------------
    Y, X, T, deltaP, sigmaP, mu_prior, V_Minn_inv, V_Minn_inv_elview, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, K_β, beta, intercept = CPZ_initMinn(YY,p)
    

    # prepare matrices for storage
    store_YY    = zeros(Tf,n,nsave);
    store_β     = zeros(n^2*p+n,nsave);
    store_Σt_inv= zeros(n,n,nsave);
    store_Σt    = zeros(n,n,nsave);

    @showprogress for ii in 1:ndraws
        # draw of the missing values
        BEAVARs.CPZ_draw_wz!(YYt,longyo,Y0,cB,B_draw,structB_draw,strctBdraw_LI,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,H_B_CI,nmdraws);
        
        # draw of the parameters
        beta,b0,B_draw,Σt_inv,structB_draw,Σt = BEAVARs.CPZ_iniw!(YY,p,hypSetup,n,k,b0,B_draw,Σt_inv,structB_draw,Σp_invsp,Σpt_ind,Y,X,T,mu_prior,deltaP,sigmaP,const_loc,Xsur_den,Xsur_CI,X_CI,XtΣ_inv_den,XtΣ_inv_X,V_Minn_inv,V_Minn_inv_elview,updP_vec,K_β,beta);

        if ii>nburn
            store_β[:,ii-nburn]  = beta;
            store_YY[:,:,ii-nburn]  = YY;
            store_Σt_inv[:,:,ii-nburn]    = Σt_inv;
            store_Σt[:,:,ii-nburn] = Σt;
        end
    end

    return store_YY,store_β, store_Σt_inv, M_zsp, z_vec, Sm_bit, store_Σt, freq_mix_tp
end




"""
    Y, X, T, deltaP, sigmaP, mu_prior, V_Minn_inv, V_Minn_inv_elview, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, intercept, K_β, beta,  = CPZ_initMinn(YY,p)

    Initializes matrices for using the Minnesota prior in the CPZ2023 framework
"""
function CPZ_initMinn(YY,p)
    Y, X, T, n, intercept       = mlagL(YY,p);
    k                           = n*p+intercept
    (deltaP, sigmaP, mu_prior)  = trainPriors(YY,p);                         # do OLS to initialize priors
    V_Minn_inv                  = 1.0*Matrix(I,n*k,n*k);                    # prior matrix
    V_Minn_inv_elview           = @view(V_Minn_inv[diagind(V_Minn_inv)]);   # will be used to update the diagonal    
    XtΣ_inv_den                 = zeros(k*n,T*n);                           # this is X' ( I(T) ⊗ Σ-1 )   from page 6 in Chan 2020 LBA
    XtΣ_inv_X                   = zeros(n*k,n*k);                           # this is X' ( I(T) ⊗ Σ-1 ) X from page 6 in Chan 2020 LBA    
    Xsur_den, Xsur_CI, X_CI     = BEAVARs.SUR_form_dense(X,n);              # prepares the SUR form and the indices of the parameters for updating
    K_β                         = zeros(n*k,n*k);                           # Variance covariance matrix of the parameters
    beta                        = zeros(n*k,);                              # the parameters in a vector

    return Y, X, T, deltaP, sigmaP, mu_prior, V_Minn_inv, V_Minn_inv_elview, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, K_β, beta, intercept
end



#------------------------------
# Output structure
@with_kw struct VAROutput_CPZ2023 <: BVARmodelOutput
    store_β::Array{}        # 
    store_Σt_inv::Array{}        # 
    store_YY::Array{}
    M_zsp::Array{} 
    z_vec::Array{} 
    Sm_bit::Array{}
    store_Σt::Array{}        # 
    var_list::Array{}
    freq_mix_tp::Tuple{Int,Int,Int}
end
# end of output strcutres
#------------------------------

#--------------------------------------
# Forecast CPZ2023
function forecast(VAROutput::VAROutput_CPZ2023,VARSetup::BVARmodelSetup,data_strct::BVARmodelDataSetup)
    @unpack store_β, store_Σt, store_YY = VAROutput
    @unpack n_fcst,p,nsave = VARSetup

    YY = median(store_YY,dims=3)

    n = size(YY,2);

    Yfor3D    = fill(NaN,(p+n_fcst,n,nsave))
    Yfor3D[1:p,:,:] .= @views YY[end-p+1:end,:];
    
    for i_draw = 1:nsave
        # Yfor3D[1:p,:,i_draw] .= @views store_YY[end-p+1:end,:,i_draw];
        Yfor3D[1:p,:,i_draw] .= @views YY[end-p+1:end,:];
        Yfor = @views Yfor3D[:,:,i_draw];
        A_draw = @views reshape(store_β[:,i_draw],n*p+1,n);
        Σ_draw = @views store_Σt[:,:,i_draw];
                
        for i_for = 1:n_fcst
            tclass = @views vec(reverse(Yfor[1+i_for-1:p+i_for-1,:],dims=1)')
            tclass = [1;tclass];
            Yfor[p+i_for,:]=tclass'*A_draw  .+ (cholesky(Hermitian(Σ_draw)).U*randn(n,1))';    
        end
    end

    fcast_strct = BEAVARs.VARForecast(Yfor3D,data_strct.dataHF_tab,data_strct.var_list,n_fcst)
    
    return fcast_strct

end # end function fcastCPZ2023()



