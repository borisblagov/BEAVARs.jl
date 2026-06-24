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
    
    Estimate Chan, Zhu, Poon 2024 using a  Minnesota-based independent Normal-Wishart prior
"""
function CPZ2023(dataHF_tab::TimeArray{Typ,N,D,A},dataLF_tab::TimeArray{Typ,N,D,A},varOrder::Array{Symbol,1},varSetup::BVARmodelSetup,hyp_struct::BVARmodelHypSetup) where {Typ <: AbstractFloat, N, D, A <: AbstractArray{Typ, N}}
    @unpack p, n_burn,n_save, const_loc, n_fcst, prior_RW = varSetup
    ndraws = n_save+n_burn;
    nmdraws = 10;               # given a draw from the parameters to draw multiple time from the distribution of the missing data for better confidence intervals

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
    YYt = BEAVARs.CPZ_draw_wz!(YYt,longyo,Y0,cB,B_draw,structB_draw,strctBdraw_LI,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,H_B_CI,nmdraws);
    
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
        BEAVARs.CPZ_draw_wz!(YYt,longyo,Y0,cB,B_draw,structB_draw,strctBdraw_LI,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,nm,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,H_B_CI,nmdraws);
        # BEAVARs.CPZ_draw_wz_lessAlloc!(YYt,longyo,Y0,cB,B_draw,structB_draw,strctBdraw_LI,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,KymBar,H_B_CI,nmdraws,μ_yBar,mdraws,draw_tmp)
        
        # draw of the parameters
        beta,b0,B_draw,Σt_inv,structB_draw,Σt = BEAVARs.CPZ_iniw!(YY,p,hyp_struct,n,k,b0,B_draw,Σt_inv,structB_draw,Σp_invsp,Σpt_ind,Y,X,T,Xsur_den,Xsur_CI,X_CI,XtΣ_inv_den,XtΣ_inv_X,Vβ_inv,βMinn,K_β,cholK_β,β_draw,S_0);

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

See also `dataCPZ2023`.

"""
function makeDataSetup(::CPZ2023_type,dataHF_tab::TimeArray, dataLF_tab::TimeArray; var_list =  [colnames(dataLF_tab); colnames(dataHF_tab)])
    return dataCPZ2023(dataHF_tab, dataLF_tab, var_list)
end



@doc raw"""
    CPZ_prep_TimeArrays(dataLF_tab::TimeArray{T,N,D,A},dataHF_tab::TimeArray{T,N,D,A},varOrder::Array{Symbol,1},prior_RW::Int,n_fcst::Int)

    Prepare one large table with the full dataset  `fataHF_tab` with both low and high-frequency variables with low-freq having `NaNs`. 
    It will extend the table with n_fcst, which is the amoung of *low frequency* time periods.

    varOrder must be a `Vector{Symbol}` and not `Vector{Vector{Symbol}}`
    e.g. [varNamesLF; varNamesHF] and not [varNamesLF, varNamesHF]
    prior_RW = 0: growth rates, 1: log-levels. indicator for the aggregate weights in the inter-temporal aggregation
"""
function CPZ_prep_TimeArrays(dataLF_tab::TimeArray{T,N,D,A},dataHF_tab::TimeArray{T,N,D,A},varOrder::Array{Symbol,1},prior_RW::Int,n_fcst::Int)  where {T <: AbstractFloat, N, D, A <: AbstractArray{T, N}}
    varNamesLF::Vector{Symbol} = Vector{Symbol}(colnames(dataLF_tab))
    z_tab::TimeArray{T,N,D,A} = dataLF_tab

    fdataHF_tab::TimeArray{T,N,D,A} = merge(
        dataHF_tab,
        map((timestamp, values) -> (timestamp, values .* NaN), z_tab[varNamesLF]),
        method = :outer,
    )
    fdataHF_tab = fdataHF_tab[varOrder]
    fvarNames::Vector{Symbol} = Vector{Symbol}(colnames(fdataHF_tab))
    datesHF::Vector{Date} = timestamp(fdataHF_tab)
    datesLF::Vector{Date} = timestamp(z_tab)
    # freqL_date::Month = Month(datesLF[2]) - Month(datesLF[1])
    # freqH_date::Month = Month(datesHF[2]) - Month(datesHF[1])
    freqL_date = get_data_freq(dataLF_tab);
    freqH_date = get_data_freq(dataHF_tab);

    if freqL_date == Month(0)
        freqL_date = Month(12)
    end 
    # tuple showing the specification: 1, 3, 12 are monthly quarterly, annually and 0,1 is growth rates or log-levels
    freq_mix_tp = (convert(Int,freqH_date/Month(1)), convert(Int,freqL_date/Month(1)),prior_RW) # tuple with the high and low frequencies. 1 is monthly, 3 is quarterly, 12 is annually

        
    # add the forecast periods to the data
    datesLF_fcast = collect(datesLF[end]+freqL_date:freqL_date:datesLF[end]+freqL_date*(n_fcst+1)) # the added quarters for the forecast. We add one more which will be deleted just to make sure that we have enough months to cover the whole quarter (e.g. Q2 is 01.04. but monthly it is up to 01.06.)
    datesHF_fcast = (datesHF[end]+freqH_date:freqH_date:datesLF_fcast[end]-freqH_date);      # the added high frequency data for the forecast

    ta = TimeArray(datesHF_fcast, fill(NaN,size(datesHF_fcast,1),length(fvarNames)))
    rename!(ta, varOrder)
    fdataHF_tab = [fdataHF_tab; ta];
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

# TODO: add support for z_tab variables with various lengths
"""
function CPZ_makeM_inter(z_tab,YYt,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,nm,Tf;scVal=10e-8)
    z_var_pos  = indexin(varNamesLF,fvarNames); # positions of the variables in z
    if length(size(z_tab)) == 1
        T_z = size(z_tab,1)
        n_z = 1
    else
        T_z, n_z = size(z_tab);  # TODO this currently supports only a balanced panel of low frequency observed variables. E.g. you cannot have GDP but not have consumption
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
            # monthly and quarterly data with growth rates
            hfWeights = [1/3; 2/3; 3/3; 2/3; 1/3]; n_hfw = size(hfWeights,1); #number of weights, depends on the variable transformation and frequency
            hf_num1 = 1; hf_num2 = 1;  # this solves the range below ii_M-div((n_hfw-hf_num1),2): ii_M+div((n_hfw-hf_num2),2). This should give the indices -2, -1, 0, +1, +2
        elseif freq_mix_tp==(3,12,0)
            # quarterly and yearly data with growth rates
            hfWeights = [1/4; 2/4; 3/4; 1; 3/4; 2/4; 1/4]; n_hfw = size(hfWeights,1); #number of weights, depends on the variable transformation and frequency
            hf_num1 = 1; hf_num2 = 1;  # this solves the range below ii_M-div((n_hfw-hf_num1),2): ii_M+div((n_hfw-hf_num2),2). This should give the indices -3, -2, -1, 0, +1, +2, +3
        elseif freq_mix_tp==(1,3,1)
            # monthly and quarterly data in levels
            hfWeights = [1/3; 1/3; 1/3]; n_hfw = size(hfWeights,1); #number of weights, depends on the variable transformation and frequency
            hf_num1 = 3; hf_num2 = -1;  # this solves the range below ii_M-div((n_hfw-hf_num),2): ii_M+div((n_hfw-hf_num),2). This should give the indices -0, +1, +2
        elseif freq_mix_tp==(3,12,1)
            # quarterly and yearly data in levels
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
    # M_zsp = M_z;                        # Leftover from when trying sparse matrices, should be fixed by stacking the M_inter_ii
    O_zsp = Matrix(I,T_z*n_z,T_z*n_z).*scVal;   # this works only if we have one z variable (with T_z length)
    for ii = 1:n_z
        if flagFirstRow[ii] == 1             
            O_zsp[1+(ii-1)*T_z,1+(ii-1)*T_z] = scVal*1000   # add a higher value to the error for the first observation if we don't have a full quarter/year in the beginning, as we will not be able to have a hard constraint in the beginning
        end
    end
    MOiM = M_z'*(O_zsp\M_z);
    MOiz = M_z'*(O_zsp\z_vec);
    return M_z, z_vec, T_z, MOiM, MOiz
end



@doc raw"""
    M_inter_agg = CPZ_makeM_inter_agg(fdatesLF,fdatesHF,freq_mix_tp)

    Generate an M matrix for aggregating vectors of low frequency to high frequency data such that M*high_freq_tab = low_freq_tab
"""
function CPZ_makeM_inter_agg(fdatesLF,fdatesHF,freq_mix_tp)
    T_z = length(fdatesLF);
    Tf = length(fdatesHF);
    M_inter_agg = zeros(T_z,Tf)
    flagFirstRow = zeros(1,);                              # if we don't have a full quarter/year we will not be able to have a hard constraint in the beginning, set the error to a higher value

    M_inter_agg = zeros(T_z,Tf)

    if size(fdatesHF,1)!==size(M_inter_agg,2)
        # error("The size of M does not match the number of dates available in z_tab. Maybe the low-frequency data is longer? The problem is with variable number ", z_var_pos[ii_z])
    end

    # we need to watch out with the dates due to how the intertemporal constraint works Take for example growth rates Q and M
    # y_t = 1/3 y_t - 2/3 y_{t-1} \dots - - 2/3 y_{t-3} - 1/3 y_{t-5}
    # Intuitively, Q1 quarterly GDP (e.g. 01.01.2000) is the weighted sum of the monthly March, February, January, December, November, and October
    # if y_t^Q is 01.01.2000, we need +2 and -2 months for the weights
    if freq_mix_tp==(1,3,0)
        # monthly and quarterly data with growth rates
        hfWeights = [1/3; 2/3; 3/3; 2/3; 1/3]; n_hfw = size(hfWeights,1); #number of weights, depends on the variable transformation and frequency
        hf_num1 = 1; hf_num2 = 1;  # this solves the range below ii_M-div((n_hfw-hf_num1),2): ii_M+div((n_hfw-hf_num2),2). This should give the indices -2, -1, 0, +1, +2
    elseif freq_mix_tp==(3,12,0)
        # quarterly and yearly data with growth rates
        hfWeights = [1/4; 2/4; 3/4; 1; 3/4; 2/4; 1/4]; n_hfw = size(hfWeights,1); #number of weights, depends on the variable transformation and frequency
        hf_num1 = 1; hf_num2 = 1;  # this solves the range below ii_M-div((n_hfw-hf_num1),2): ii_M+div((n_hfw-hf_num2),2). This should give the indices -3, -2, -1, 0, +1, +2, +3
    elseif freq_mix_tp==(1,3,1)
        # monthly and quarterly data in levels
        hfWeights = [1/3; 1/3; 1/3]; n_hfw = size(hfWeights,1); #number of weights, depends on the variable transformation and frequency
        hf_num1 = 3; hf_num2 = -1;  # this solves the range below ii_M-div((n_hfw-hf_num),2): ii_M+div((n_hfw-hf_num),2). This should give the indices -0, +1, +2
    elseif freq_mix_tp==(3,12,1)
        # quarterly and yearly data in levels
        hfWeights = [1/4; 1/4; 1/4; 1/4]; n_hfw = size(hfWeights,1); #number of weights, depends on the variable transformation and frequency
        hf_num1 = 4; hf_num2 = -3;  # this solves the range below ii_M-div((n_hfw-hf_num),2): ii_M+div((n_hfw-hf_num),2). This should give the indices -0, +1, +2
    else
        error("This combination of frequencies and transformation has not been implemented")
    end

    for ii_zi in eachindex(fdatesLF) # iterator going through each time point in datesHF
        if ii_zi == 1 # check if we have a full quarter/year in the beginning, otherwise we will try to acces negative indices in the matrix M
            ii_M = findall(fdatesHF.==fdatesLF[ii_zi])[1]       # find the low-frequency index that corresponds to the high-frequency missing value
            MrowRange = ii_M-div((n_hfw-hf_num1),2): ii_M+div((n_hfw-hf_num2),2);
            start_value = max(1, MrowRange[1]); stop_value = MrowRange[end]; positive_range = start_value:stop_value
            M_inter_agg[ii_zi,start_value:stop_value]=hfWeights[MrowRange.>0];
            flagFirstRow[1] = 1;
        else
            ii_M = findall(fdatesHF.==fdatesLF[ii_zi])[1]       # find the low-frequency index that corresponds to the high-frequency missing value
            # M_inter_agg[ii_zi, findall(datesHF.==fdatesLF[ii_zi])[1]-n_hfw+1:findall(datesHF.==fdatesLF[ii_zi])[1]] = hfWeights # if shifted above
            M_inter_agg[ii_zi,ii_M-div((n_hfw-hf_num1),2): ii_M+div((n_hfw-hf_num2),2)]=hfWeights; # +2 and - 2 months for the weights or +3 and -3
        end
    end

    return M_inter_agg
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
    Smsp = Sm;            # sparse Sm
    Sosp = So;            # sparse So
    
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
    Draw with restrictions
"""
function CPZ_draw_wz_lessAlloc!(YYt,longyo,Y0,cB,B_draw,structB_draw,sBd_ind,Σt_inv,Σt_LI,Xb,cB_b0_LI,Σ_invsp,p,n,Sm_bit,Smsp,Sosp,MOiM,MOiz,Gm,Go,H_B,GΣ,Kym,KymBar,H_B_CI,nmdraws,μ_yBar,mdraws,draw_tmp)
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
    CL = cholesky(sparse(Hermitian(Kym)))
    long_pr = (cB-Go*longyo);
    μ_y = CL.U\(CL.L\(GΣ*long_pr));

    KymBar[:,:] = MOiM + Kym;
    CLBar = cholesky(Hermitian(KymBar))
    # we want to calculate  μ_yBar = CLBar.U\(CLBar.L\(MOiz + Kym*μ_y))
    mul!(μ_yBar,Kym,μ_y)                                        # do the inner most first, overwriting μ_y to save memory
    μ_yBar[:] = MOiz + μ_yBar                                   # now add MOiz, still overwriting μ_yBar
    ldiv!(CLBar.U,ldiv!(CLBar.L,μ_yBar))
    # μ_yBar[:] = CLBar.U\(CLBar.L\(MOiz + Kym*μ_y))

    for i_draw in 1:nmdraws
        mdraws[:,i_draw] .= μ_yBar .+  ldiv!(CLBar.U,Random.randn!(draw_tmp))
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
    Updates parameters using an independennt Normal-Wishart prior

    Updates (i.e. overwrites): 
        β_draw, 
        B_draw, 
        structB_draw,
        Σt,
        Σt_inv
"""
function CPZ_iniw!(YY,p,hyp_struct,n,k,b0,B_draw,Σt_inv,structB_draw,Σp_invsp,Σpt_ind,Y,X,T,Xsur_den,Xsur_CI,X_CI,XtΣ_inv_den,XtΣ_inv_X,Vβ_inv,βMinn,K_β,cholK_β,β_draw,S_0)
    Y, X = mlagL!(YY,Y,X,p,n)
    Σp_invsp.nzval[:] = Σt_inv[Σpt_ind];    
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
function forecast(VAROutput::VAROutput_CPZ2023,VARSetup::BVARmodelSetup,data_struct::BVARmodelDataSetup)
    # TODO change these lines to using the function get_imp_percentiles
    @unpack store_β, store_Σt, store_YY, store_YY_LF, fdatesHF, fdatesLF = VAROutput
    @unpack n_fcst,p,n_save = VARSetup
    @unpack dataHF_tab, dataLF_tab, var_list = data_struct
    YYforHF3d = store_YY;
    YYforLF3d = store_YY_LF

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

#--------------------------------------
# forecast evaluation 
#--------------------------------------

"""
    eval_forecast(out_struct::VAROutput_CPZ2023, data_struct::BVARmodelDataSetup, set_struct::BVARmodelSetup, dataLF_true_ftab::TimeArray)

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
function eval_forecast(out_struct::VAROutput_CPZ2023, data_struct::BVARmodelDataSetup, set_struct::BVARmodelSetup, dataLF_true_ftab::TimeArray)  
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
        data_true_VecView = data_truef_mat[data_true_flags_vec,locs]    # select only the variables that we want to evaluate and only the relevant time periods

        # these are the relevant forecasts
        fcast_YY =  store_YY_LF[fcastDatesOverlap_BitVec,locs,:]
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


# Structure for the datasets and the frequency mix
@doc raw"""
    dataCPZ2023(data_HF::TimeArray,data_LF::TimeArray,var_list::Array{Symbol,1})

Generate a dataset strcture for use with CPZ2023 model

# Arguments
    dataHF_tab: TimeArray with your high-frequency variables (monthly or quarterly, respectively)
    dataLF_tab: TimeArray with your low-frequency variables (quarterly or yearly, respectively)
    var_list:   the variable order. Note that the functions that call these variables allow this to be optional.

See also `makeDataSetup`.
"""
@with_kw struct dataCPZ2023{T <: AbstractFloat, N, D, A <: AbstractArray{T, N}} <: BVARmodelDataSetup
    dataHF_tab::TimeArray{T,N,D,A}                                       # data for the high-frequency variables
    dataLF_tab::TimeArray{T,N,D,A}                                       # data for the low-frequency variables
    var_list::Array{Symbol,1}                                   # Symbol vector with the variable names, will be used for ordering
end


"""
    Houses the forecast errors, averaged across draws and predictive likelihood for a specific vintage
"""
struct eval_vint_CPZ2023{T <: AbstractFloat, N} <: BVARmodelEval
    pred_lik_mat::Array{T,N}  
    fcast_errors_mAd_mat::Array{T,N}
    fcastDatesOverlap::Array{Date,1}
    data_true_VecView::Array{T,N}
    data_true_dates::Array{Date,1}
end


# """
#     Houses the forecast errors, averaged across draws and predictive likelihood for a specific vintage
# """
# struct eval_vint_CPZ2023_SA{T <: AbstractFloat, N} <: BVARmodelEval
#     pred_lik_mat::Array{T,N}  
#     fcast_errors_mAd_mat::Array{T,N}
#     # fcastDatesOverlap::Array{Date,1}
#     data_true_VecView::Array{T,N}
#     fcast_YY::Array{T,N}
#     # data_true_dates::Array{Date,1}
# end