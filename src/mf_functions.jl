# A collection of functions for working with mixed frequencies


@doc raw"""
    makeM_intertemp(z_tab,YYt,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,nm,Tf;scVal=10e-8)

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
function makeM_intertemp(z_tab,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,Tf;scVal=10e-8)
    nm = sum(Sm_bit);
    z_var_pos  = indexin(varNamesLF,fvarNames); # positions of the variables in z
    if length(size(z_tab)) == 1
        T_z = size(z_tab,1)
        n_z = 1
    else
        T_z, n_z = size(z_tab);  # TODO this currently supports only a balanced panel of low frequency observed variables. E.g. you cannot have more obs. in GDP but not in consumption if you have both as low-frequency variables
    end
    M_z = zeros(T_z*n_z,nm)
    z_vec = zeros(T_z*n_z,)
    iter = CartesianIndices(Sm_bit)
    flagFirstRow = falses(n_z);                              # if we don't have a full quarter/year we will not be able to have a hard constraint in the beginning, set the error to a higher value

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
                flagFirstRow[ii_z] = true;
            else
                ii_M = findall(datesHF.==datesLF_ii[ii_zi])[1]       # find the low-frequency index that corresponds to the high-frequency missing value
                # M_z_ii[ii_zi, findall(datesHF.==datesLF_ii[ii_zi])[1]-n_hfw+1:findall(datesHF.==datesLF_ii[ii_zi])[1]] = hfWeights # if shifted above
                M_z_ii[ii_zi,ii_M-div((n_hfw-hf_num1),2): ii_M+div((n_hfw-hf_num2),2)]=hfWeights; # +2 and - 2 months for the weights or +3 and -3
            end
        end
        M_z[(ii_z-1)*T_z + 1:T_z + (ii_z-1)*T_z,:] = M_inter_ii;    # , should be fixed by stacking the M_inter_ii
        z_vec[(ii_z-1)*T_z + 1:T_z + (ii_z-1)*T_z,]  = values(z_tab[varNamesLF[ii_z]]);
    end

    O_z_int = Matrix(I,T_z*n_z,T_z*n_z).*scVal;   # this works only if we have one z variable (with T_z length)
    for ii = 1:n_z
        if flagFirstRow[ii] == true             
            O_z_int[1+(ii-1)*T_z,1+(ii-1)*T_z] = scVal*1000   # add a higher value to the error for the first observation if we don't have a full quarter/year in the beginning, as we will not be able to have a hard constraint in the beginning
        end
    end

    T_z_inter = T_z*n_z;

    return M_z, z_vec, T_z_inter, O_z_int
end


@doc raw"""
    makeM_crossSect(z_tabCS, var_listLF, dataCS_tab, fdataHF_tab, Sm_bit,
                    prior_RW, nm; scVal=10e-8)
              
    Construct the mapping matrix for a cross-sectional constraint on missing
    high-frequency observations. 
    
    Given observations $z$, and a weighting matrix weightsCS_tab the the function returns
        $z = M_cs * y^m$
    for a vector of missing data $y^m$. 
        
    It matches the entries in $z$ with the entries in  $y^m$ using dates in dataCS_tab.

    The constraint series must contain exactly one variable. The dates in
    `z_tabCS` must be present in the weight table.

# Arguments
    - `z_tabCS`: A one-column `TimeArray` containing the cross-sectional
    constraint observations.
    - `var_listLF`: Names of the high-frequency variables included in the
    constraint. These names select and order the columns from `dataCS_tab`.
    - `dataCS_tab`: A `TimeArray` of cross-sectional weights, indexed by date and
    variable.
    - `fdataHF_tab`: The high-frequency data `TimeArray`; its dates and column
    names are used to map variables and observations.
    - `Sm_bit`: Boolean mask identifying missing high-frequency observations in the full dataset.
    Its shape and ordering should correspond to the complete datset in its high-frequency format.
    - `prior_RW`: 0: Whether the data in the MF-VAR is in growth rates or in log-levels
    - `nm`: Number of missing observations in the y^m vector, the sum of `Sm_bit`.
    - `scVal`: Diagonal value used to initialize the constraint error matrix
    `O_z_cs`. Defaults to `10e-8`.

# Returns
    A tuple `(M_cs, z_cs_vec, T_z_cs, O_z_cs)`:
    - `M_cs`: Constraint matrix, with one row per missing observation.
    - `z_cs_vec`: The observed values that are some function of the missing data.
    - `T_z_cs`: Number of constraint rows.
    - `O_z_cs`: Diagonal constraint error matrix.

# Errors
    Throws an error if `z_tabCS` has more than one column, the number of missing
    entries in `Sm_bit` does not equal `nm`, or a constraint date is missing from
    the weight table.
"""
function makeM_crossSect(z_tabCS,var_listLF,dataCS_tab,fdataHF_tab,Sm_bit,prior_RW,nm;scVal=10e-8)
    if size(z_tabCS,2) ≠ 1
        error("z_tabCS should have only one variable in it. If you want multiple variabels consider adding another constraint")
    end

    YY_dates    = TimeSeries.timestamp(fdataHF_tab);
    fvarNames   = TimeSeries.colnames(fdataHF_tab);
    weightsCS_tab = dataCS_tab[var_listLF];         # weighting matrix ordered by the order in fdataHF_tab. Should contain the variable to be constraint
    weights_dates = TimeSeries.timestamp(weightsCS_tab);
    varNamesCS  = var_listLF                        # variable names of the unobserved variables that will be constrained
    T_z_cs      = size(z_tabCS,1)                   # Observations of the cross-sectional constraint
    n_cs_var    = size(weightsCS_tab,2)             # number of variables in the HF to be constrained
    z_dates     = TimeSeries.timestamp(z_tabCS);

    z_varCS_pos  = indexin(varNamesCS,fvarNames);   # positions of the variables in YYt and correspondingly in Sm_bit
    iter = CartesianIndices(Sm_bit)
    ym_ci_all = iter[Sm_bit]                        # The full vector of y^m, should be nm × 1
    if length(ym_ci_all) ≠ nm
        error("The length of the y^m vector does not correspond to the missing values in Sm_bit")
    end

    if issubset(Set(z_dates), Set(weights_dates)) ≠ true
        error("All weight entries in the supplied z_tabCS must be present in the weights table to properly assign weights to them.")
    end
    ym_rownum= getindex.(ym_ci_all, 1);             # all row numbers of the missing entries in y^m. Every row, that correspond to the variables in z_posCS is one that we need
    ym_colnum= getindex.(ym_ci_all, 2);             # all row numbers of the missing entries in y^m. Every row, that correspond to the variables in z_posCS is one that we need
    ym_dates = YY_dates[ym_colnum]                  # The vector of dates that correspond to each missing observation


    # initialize M_cs
    M_cs = zeros(T_z_cs,nm);                        # initialize M_cs
    O_z_cs = diagm(ones(T_z_cs).*scVal);             # initialize the error matrix
    
    if prior_RW == 0
        ## Intertemporal constraints with variables in log-growth rates

        # we now want to loop over each entry of the date in z_tabCS and then fix that year to the cartesian indices of the unobserved variables
        for it_z = 1:T_z_cs
            it_z_date = TimeSeries.timestamp(z_tabCS[it_z])[1]

            bit_rownum = ym_rownum.==z_varCS_pos[1];        # = 1 for every entry in y^m that corresponds to a variable from weightsCS_tab but in the high-frequency
            for io = 2:n_cs_var
                bit_io = ym_rownum.==z_varCS_pos[io]
                bit_rownum .= bit_rownum .| bit_io
            end
            bit_colnum_it_z = ym_dates.==it_z_date;
            idx_it_z = bit_rownum .& bit_colnum_it_z;       # a bit_vector specifying which position in y^m correspond to the entries of it_z

            M_cs[it_z,idx_it_z] = @view values(weightsCS_tab[weights_dates.==it_z_date])[:]
        end

        z_cs_vec = TimeSeries.values(z_tabCS)

    elseif prior_RW == 1
        ## Intertemporal constraints with variables in log-levels. Note that we lose one observation

        # we now want to loop over each entry of the date in z_tabCS and then fix that year to the cartesian indices of the unobserved variables
        for it_z = 2:T_z_cs
            itm1_z_date = TimeSeries.timestamp(z_tabCS[it_z-1])[1]
            it_z_date = TimeSeries.timestamp(z_tabCS[it_z])[1]


            bit_rownum = ym_rownum.==z_varCS_pos[1];        # = 1 for every entry in y^m that corresponds to a variable from weightsCS_tab but in the high-frequency
            for io = 2:n_cs_var
                bit_io = ym_rownum.==z_varCS_pos[io]
                bit_rownum .= bit_rownum .| bit_io
            end
            bit_colnum_itm1_z = ym_dates.==itm1_z_date;     # the minus iz_t-1 position
            bit_colnum_it_z = ym_dates.==it_z_date;         # the iz_t position
            idx_itm1_z = bit_rownum .& bit_colnum_itm1_z;   # a bit_vector specifying which position in y^m correspond to the entries of it_z
            idx_it_z = bit_rownum .& bit_colnum_it_z;       # a bit_vector specifying which position in y^m correspond to the entries of it_z
            
            M_cs[it_z,idx_it_z] = @view values(weightsCS_tab[weight_dates.==it_z_date])[:]
            M_cs[it_z,idx_itm1_z] = -values(weightsCS_tab[weight_dates.==it_z_date])[:]
        end
        
        # drop now the first rows
        M_cs = M_cs[2:end,:]
        z_cs_vec = z_cs_vec[2:end,]
        T_z_cs = T_z_cs-1;
        O_z_cs = O_z_cs[2:end,2:end]
    end


    return M_cs, z_cs_vec,T_z_cs, O_z_cs
end



@doc raw"""
    M_inter_agg = makeMInterAgg(fdatesLF,fdatesHF,freq_mix_tp)

    Generate an M matrix for aggregating vectors of low frequency to high frequency data such that M*high_freq_tab = low_freq_tab
"""
function makeMInterAgg(fdatesLF,fdatesHF,freq_mix_tp)
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