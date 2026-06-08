
@doc raw"""
    hf_ta = BEAVARs.TAtrans(dataHF_TA,varList_HF,trans_dictA)

Applies transformations to a TimeArray based on a transformation dictionary
# Arguments
    dataHF_TA: TimeArray of the high-frequency data
    varList_HF: list of variable names (Symbols) in the high-frequency data
    trans_dictA: dictionary of transformations for each variable

# Returns  
    hf_ta: TimeArray with transformed high-frequency data

# Description

This function applies specified transformations to each variable in a TimeArray based on a provided transformation dictionary.
The transformations are defined as follows:
- If the transformation value is 1, the variable is left unchanged.
- If the transformation value is 2, the variable is divided by 100.
- If the transformation value is 3, the natural logarithm of the variable is taken.
- If the transformation value is 7, the percentage change of the variable is calculated.
- If the transformation value is 8, the variable is multiplied by 100.
- If the transformation value is 9, the exponential of the variable is taken.

# Example

Consider a TimeArray with variables A, B, and C. If the transformation dictionary specifies that A should be left unchanged (1), B should be divided by 100 (2), and C should be logged (3),
the function will apply these transformations accordingly and return a new TimeArray with the transformed variables.

```julia
dataHF_TA = TimeArray(Date(2020,1,1):Month(1):Date(2020,6,1), rand(6,3), [:A, :B, :C])
trans_dictA = Dict(:A => 1, :B => 2, :C => 3)
varList_HF = [:A, :B, :C]
hf_ta = BEAVARs.TAtrans(dataHF_TA, varList_HF, trans_dictA)
```

```
julia> dataHF_TA = TimeArray(Date(2020,1,1):Month(1):Date(2020,6,1), rand(6,3), [:A, :B, :C])
6×3 TimeArray{Float64, 2, Date, Matrix{Float64}} 2020-01-01 to 2020-06-01

┌────────────┬──────────┬──────────┬──────────┐
│            │ A        │ B        │ C        │
├────────────┼──────────┼──────────┼──────────┤
│ 2020-01-01 │   0.3502 │  0.38563 │ 0.626134 │
│ 2020-02-01 │ 0.854189 │ 0.458228 │ 0.457359 │
│ 2020-03-01 │ 0.404973 │ 0.388638 │ 0.605352 │
│ 2020-04-01 │ 0.575055 │ 0.858383 │ 0.389653 │
│ 2020-05-01 │ 0.445191 │  0.54553 │ 0.354785 │
│ 2020-06-01 │ 0.174958 │ 0.672888 │ 0.844214 │
└────────────┴──────────┴──────────┴──────────┘

julia> trans_dictA = Dict(:A => 1, :B => 2, :C => 3)
Dict{Symbol, Int64} with 3 entries:
  :A => 1
  :B => 2
  :C => 3

julia> varList_HF = [:A, :B, :C]
3-element Vector{Symbol}:
 :A
 :B
 :C

julia> hf_ta = BEAVARs.TAtrans(dataHF_TA, varList_HF, trans_dictA)
5×3 TimeArray{Float64, 2, Date, Matrix{Float64}} 2020-02-01 to 2020-06-01
┌────────────┬──────────┬────────────┬───────────┐
│            │ A        │ B          │ C         │
├────────────┼──────────┼────────────┼───────────┤
│ 2020-02-01 │ 0.854189 │ 0.00458228 │ -0.782287 │
│ 2020-03-01 │ 0.404973 │ 0.00388638 │ -0.501945 │
│ 2020-04-01 │ 0.575055 │ 0.00858383 │ -0.942498 │
│ 2020-05-01 │ 0.445191 │  0.0054553 │  -1.03624 │
│ 2020-06-01 │ 0.174958 │ 0.00672888 │ -0.169349 │
└────────────┴──────────┴────────────┴───────────┘
```
"""
function TAtrans(dataHF_TA,varList_HF,trans_dictA)
transHF_vec = getindex.(Ref(trans_dictA),varList_HF);       # vector of transformations for the vars

    # transformations
# if 1, leave it be, initialize TA
hf_ta = map((timestamp, values) -> (timestamp, (values)), dataHF_TA[varList_HF[transHF_vec.==1]])
#if 2, divide by 100
hf_ta_temp = map((timestamp, values) -> (timestamp, (values)./100.0), dataHF_TA[varList_HF[transHF_vec.==2]])
hf_ta = merge(hf_ta,hf_ta_temp)
# if 3, take logs
hf_ta_temp = map((timestamp, values) -> (timestamp, log.(values)), dataHF_TA[varList_HF[transHF_vec.==3]])
hf_ta = merge(hf_ta,hf_ta_temp)
# if 4, take differences
hf_ta_temp = diff(dataHF_TA[varList_HF[transHF_vec.==4]]);
hf_ta = merge(hf_ta,hf_ta_temp)
# if 5 caclulate log changes in decimals (e.g. 0.01 is 1%)
hf_ta_temp = percentchange(dataHF_TA[varList_HF[transHF_vec.==5]], :log);
hf_ta = merge(hf_ta,hf_ta_temp)
# if 7 caclulate percentage changes in decimals (e.g. 0.01 is 1%)
hf_ta_temp = percentchange(dataHF_TA[varList_HF[transHF_vec.==7]]);
hf_ta = merge(hf_ta,hf_ta_temp)
# if 8, multiply by 100
hf_ta_temp = map((timestamp, values) -> (timestamp, (values).*100.0), dataHF_TA[varList_HF[transHF_vec.==6]])
hf_ta = merge(hf_ta,hf_ta_temp)
# if 9, take exp
hf_ta_temp = map((timestamp, values) -> (timestamp, exp.(values)), dataHF_TA[varList_HF[transHF_vec.==9]])
hf_ta = merge(hf_ta,hf_ta_temp)

hf_ta = hf_ta[varList_HF];  # reorder back the variables

return hf_ta
end


function xlsx2ta(data_mat)
    data_mat = data_mat[.!ismissing.(data_mat[:,1]),:];             # sometimes XLSX reads empty rows below the last one (especially conditional formatting), here we remove them

    nan_mat =  fill(NaN, size(data_mat,1),size(data_mat,2));
    data_mat[ismissing.(data_mat)]=nan_mat[ismissing.(data_mat)];   # replace all missing with NaN
    values_mat = convert(Array{Float64},data_mat[2:end,2:end])      # convert values to numbers
    date_vec = Date.(data_mat[2:end,1])                         # convert first column to DateTime
    # date_vecstr = Dates.DateFormat.(date_vec,"yyyy-mm-dd");

    dataf_TA = TimeArray(date_vec,values_mat,Symbol.(data_mat[1,2:end]))
    return dataf_TA
end


function readSpec(modelstr,data_path)
    xf = XLSX.readxlsx(data_path);
    sh_names = XLSX.sheetnames(xf)

    sh_ref = xf["setup"];
    sh_mat = sh_ref[:]

    # Reads the setup sheet
    model_ind = findall(sh_mat[1,:].==modelstr)[1]
    vb_ind = findall(sh_mat[:,1].=="lastRow")[1] + 1;       # where the variables start
    varListA_str = sh_mat[vb_ind:end,1];                    # strings of variables
    varListA_sym = Symbol.(varListA_str[:]);                # symbols of variables

    vm_bit = .!iszero.(sh_mat[vb_ind:end,model_ind]);       # boolean list of ALL variables
    vm_trans = sh_mat[vb_ind:end,model_ind];                # list of variables with transformations for ALL vars

    varListF = varListA_sym[vm_bit]
    trans_dictA = Dict(varListA_sym .=> vm_trans);          # transformation dictionary vector for ALL vars. 



    # Reads the high-frequency
    datasheetHF_str = "datasheet" * string(sh_mat[sh_mat[:,1].=="datasheet",model_ind][1]) * "_HF";    # the high-frequency datasheet
    data_mat = xf[datasheetHF_str][:];
    dataf_HF_TA = BEAVARs.xlsx2ta(data_mat)
    varList_HF = intersect(varListF,colnames(dataf_HF_TA));      # looks for which variables are required and which are found
    dataHF_TA = dataf_HF_TA[varList_HF];                         # selects the variables found in this TA

    hf_ta = BEAVARs.TAtrans(dataHF_TA,varList_HF,trans_dictA)


    # Reads the low-frequency
    datasheetLF_str = "datasheet" * string(sh_mat[sh_mat[:,1].=="datasheet",model_ind][1]) * "_LF";    # the low-frequency datasheet
    data_mat = xf[datasheetLF_str][:];
    dataf_LF_TA = BEAVARs.xlsx2ta(data_mat)
    varList_LF = intersect(varListF,colnames(dataf_LF_TA));      # looks for which variables are required and which are found
    dataLF_TA = dataf_LF_TA[varList_LF];                         # selects the variables found in this TA

    # transHF_vec = Vector{Int}();                                # vector of transformations for the found high-freq. vars
    # push!(transHF_vec,(trans_dictA[i] for i in varList_HF)...);  # fill it
    # transHF_vec = getindex.(Ref(trans_dictA),varList_HF);       # vector of transformations for the vars

    lf_ta = BEAVARs.TAtrans(dataLF_TA,varList_LF,trans_dictA)
    
    return hf_ta, lf_ta, varListF

end




@doc raw"""
    BEAVARs.pseudo_oos(fdataHF_tab,fdataLF_tab,pseudoHF_beg_date,pseudoHF_end_date,ragged_beg_date,ragged_end_date,pubDelay,aggMix,model_type, set_struct, hyp_struct)

    Generate a dictionary of LoopSetup structures for pseudo out-of-sample forecasting with ragged-edge data

# Arguments
    fdataHF_tab: TimeArray of the full high-frequency data
    fdataLF_tab: TimeArray of the full low-frequency data
    pseudoHF_beg_date: Date of the beginning of the pseudo out-of-sample forecasting (high-frequency)
    pseudoHF_end_date: Date of the end of the pseudo out-of-sample forecasting (high-frequency)
    ragged_beg_date: Date of the beginning of the ragged edge (high-frequency)
    ragged_end_date: Date of the end of the ragged edge (high-frequency)
    pubDelay: publication delay in months between high-frequency and low-frequency data
    aggMix: aggregation mix parameter (0=growth rates, 1=levels)
    model_type: model type, output of makeSetup function
    set_struct: BVARmodelSetup structure
    hyp_struct: BVARmodelHypers structure

# Returns
    vint_dict: dictionary of LoopSetup structures for each pseudo out-of-sample date

# Description

This function creates a dictionary of LoopSetup structures for pseudo out-of-sample forecasting with ragged-edge data.
For each date in the pseudo out-of-sample range, it creates a balanced dataset up to that date, applies the ragged edge by setting 
the appropriate high-frequency data points to NaN, and constructs the corresponding LoopSetup structure.

## High and low-frequency datasets

The full high- and low-frequency datasets, `fdataHF_tab` and `fdataLF_tab` are needed to be cut incrementally. 
'pseudoHF_beg_date' and 'pseudoHF_end_date' give the beginning and end dates for which the new datasets will be created.

## Ragged edge
The user defines the ragged edge in the high-frequency dataset by submitting the dates in which the ragged edge starts and ends, `ragged_beg_date` and `ragged_end_date`, respectively.
These dates are defined as follows. `ragged_beg_date` is the first date after the end of the balanced dataset. `ragged_end_date` is the last date before the first row with `NaN`s.

## Example

Consider the following array:
```julia
┌────────────┬───────────┬───────────┬──────────┐
│            │ A         │ B         │ C        │
├────────────┼───────────┼───────────┼──────────┤
│ 2018-01-01 │  0.113132 │  0.643878 │ 0.712192 │
│ 2018-02-01 │  0.508802 │  0.187696 │ 0.926146 │
│ 2018-03-01 │ 0.0694366 │  0.995718 │  0.92898 │
│ 2018-04-01 │  0.683238 │  0.069832 │ 0.354071 │
│ 2018-05-01 │  0.740803 │  0.843847 │ 0.695057 │
│ 2018-06-01 │  0.295119 │  0.802446 │ 0.992353 │
│ 2018-07-01 │  0.163745 │  0.331479 │ 0.743982 │
│ 2018-08-01 │  0.541091 │  0.184965 │ 0.375754 │
│ 2018-09-01 │  0.254278 │  0.327575 │ 0.420596 │
│ 2018-10-01 │  0.948557 │    NaN    │ 0.823303 │
│ 2018-11-01 │  0.256987 │    NaN    │    NaN   │
│ 2018-12-01 │    NaN    │    NaN    │    NaN   │
└────────────┴───────────┴───────────┴──────────┘
```

The ragged edge pattern starts at `2018-10-01` and ends at `2018-11-01`.

Suppose that we want to estimate the model for in a pesudo-out-of-sample fashion for the dates `2018-05-01`, `2018-06-01` and `2018-07-01`,
the function will create the following three arrays:
```julia
For `2018-05-01`:                                   For `2018-06-01`:                                   For `2018-07-01`:
┌────────────┬───────────┬───────────┬──────────┐   ┌────────────┬───────────┬───────────┬──────────┐   ┌────────────┬───────────┬───────────┬──────────┐   
│            │ A         │ B         │ C        │   │            │ A         │ B         │ C        │   │            │ A         │ B         │ C        │
├────────────┼───────────┼───────────┼──────────┤   ├────────────┼───────────┼───────────┼──────────┤   ├────────────┼───────────┼───────────┼──────────┤
│ 2018-01-01 │  0.113132 │  0.643878 │ 0.712192 │   │ 2018-01-01 │  0.113132 │  0.643878 │ 0.712192 │   │ 2018-01-01 │  0.113132 │  0.643878 │ 0.712192 │
│ 2018-02-01 │  0.508802 │  0.187696 │ 0.926146 │   │ 2018-02-01 │  0.508802 │  0.187696 │ 0.926146 │   │ 2018-02-01 │  0.508802 │  0.187696 │ 0.926146 │
│ 2018-03-01 │ 0.0694366 │  0.995718 │  0.92898 │   │ 2018-03-01 │ 0.0694366 │  0.995718 │  0.92898 │   │ 2018-03-01 │ 0.0694366 │  0.995718 │  0.92898 │
│ 2018-04-01 │  0.683238 │  0.069832 │ 0.354071 │   │ 2018-04-01 │  0.683238 │  0.069832 │ 0.354071 │   │ 2018-04-01 │  0.683238 │  0.069832 │ 0.354071 │
│ 2018-05-01 │  0.740803 │    NaN    │ 0.695057 │   │ 2018-05-01 │  0.740803 │  0.843847 │ 0.695057 │   │ 2018-05-01 │  0.740803 │  0.843847 │ 0.695057 │
│ 2018-06-01 │  0.295119 │    NaN    │    NaN   │   │ 2018-06-01 │  0.295119 │    NaN    │ 0.992353 │   │ 2018-06-01 │  0.295119 │  0.802446 │ 0.992353 │
└────────────┴───────────┴───────────┴──────────┘   │ 2018-07-01 │  0.163745 │    NaN    │    NaN   │   │ 2018-07-01 │  0.163745 │    NaN    │ 0.743982 │   
                                                    └────────────┴───────────┴───────────┴──────────┘   │ 2018-08-01 │  0.541091 │    NaN    │    NaN   │
                                                                                                        └────────────┴───────────┴───────────┴──────────┘
```
preserving the ragged edge pattern of the data.
"""
function pseudo_oos(fdataHF_tab,fdataLF_tab,pseudoHF_beg_date,pseudoHF_end_date,ragged_beg_date,ragged_end_date,pubDelay,aggMix,model_type, set_struct, hyp_struct)
    dataHF_beg_date = timestamp(fdataHF_tab)[1];
    fraggedHF = ragged_beg_date:Month(1):ragged_end_date;         # range of ragged edge of the high-frequency data (range between the last row of the balanced sample and the first row of NaNs for every variable)
    fraggednan = isnan.(values(fdataHF_tab[fraggedHF]));               # logical array indicating which variables are NaN in the ragged edge    


    HF_beg_date = pseudoHF_beg_date;
    LF_beg_date = pseudoHF_beg_date-Month(pubDelay)
    n_pseu = length(pseudoHF_beg_date:Month(1):pseudoHF_end_date)
    vint_dict = ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup}()

    for ii in 0:n_pseu-1
        HF_beg_date = pseudoHF_beg_date+Month(ii);
        LF_beg_date = HF_beg_date-Month(pubDelay)
        rangeHF = dataHF_beg_date:Month(1):HF_beg_date;          # range for the estimation (pseudo out of sample)
        dataHF_tab = copy(fdataHF_tab[rangeHF]);                        # balanced copy of the high-frequency data in the estimation range
        # change the last values to NaN according to the ragged edge
        values(dataHF_tab)[end-size(fraggednan,1)+1:end,:] = values(dataHF_tab)[end-size(fraggednan,1)+1:end,:].*(fraggednan.*NaN.+1.0);
        rangeLF = Date(2005, 1, 1):Quarter(1):LF_beg_date;         # range for the low frequency
        dataLF_tab = copy(fdataLF_tab[rangeLF]);

        data_struct = makeDataSetup(model_type,dataHF_tab, dataLF_tab,aggMix)
        loop_struct = LoopSetup(model_type,set_struct,hyp_struct,data_struct)

        vint_dict["v"*Dates.format(HF_beg_date,"yyyy-mm-dd")] = loop_struct
    end
    return vint_dict
end



function make_vintages_dict_loop(file_path::String,model_cols_list::Array{String, 1},vintages_names_list::Array{String, 1},model_type::BVARmodelType,set_struct::BVARmodelSetup,hyp_struct::BVARmodelHypSetup)
    n_list = length(model_cols_list);
    vint_dict = ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup}()
    for ii=1:n_list 
        dataHF_tab, dataLF_tab, varOrder = BEAVARs.readSpec(model_cols_list[ii],file_path);
        data_struct = makeDataSetup(model_type,dataHF_tab, dataLF_tab)
        vint_dict[vintages_names_list[ii]] = LoopSetup(model_type,set_struct,hyp_struct,data_struct)     
    end
    return vint_dict
end