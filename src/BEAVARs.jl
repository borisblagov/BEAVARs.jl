module BEAVARs
using   LinearAlgebra, 
        Distributions, 
        SparseArrays,
        BlockBandedMatrices,
        BlockArrays,
        TimeSeries, 
        Parameters,
        ProgressMeter,
        XLSX,
        ThreadSafeDicts,
        Plots,
        Random,
        KernelDensity,
        Tables,
        PrettyTables,
        LaTeXStrings


export hypChan2020, hypBGR2010

export beavar, beavars, beavars_weave, makeOutput, makeSetup, makeHypSetup, makeDataSetup, LoopSetup


# Types for multiple dispatch across models
abstract type BVARmodelType end         # types for models
abstract type BVARmodelMFType <: BVARmodelType end # a type cathing all mixed frequency models
abstract type BVARmodelSFType <: BVARmodelType end # a type cathing all single frequency models
abstract type BVARmodelSetup end        # type for VAR setup parameters
abstract type BVARmodelHypSetup end     # types for hyperparameters
abstract type BVARmodelDataSetup end    # types for storing the data unputs to the models
abstract type BVARmodelDataSetup_MF <: BVARmodelDataSetup end # a type cathing all mixed frequency models
abstract type BVARmodelDataSetup_MFcs <: BVARmodelDataSetup end # a type cathing all mixed frequency models
abstract type BVARmodelLoopSetup end    # type for VAR setup parameters
abstract type BVARmodelOutput end       # type for output storage
abstract type BVARmodelEval end       # type for forecast evaluation storage
abstract type BVARforecastOutput end    # type for forecast output storage


# Structures for multiple dispatch across models
struct Chan2020minn_type <: BVARmodelSFType end
struct Chan2020iniw_type <: BVARmodelSFType end
struct Chan2020csv_type <: BVARmodelSFType end
struct CCK2016csv_type <: BVARmodelSFType end
struct BGR2010_type <: BVARmodelSFType end
struct CPZ2023_type <: BVARmodelMFType end
struct CPZ2023iniw_type <: BVARmodelMFType end
struct Blagov2025_type <: BVARmodelMFType end
struct Blagov2027_type <: BVARmodelMFType end

# Default structure for hyperparameters
struct hypDefault_struct <: BVARmodelHypSetup end    # empty structure for initialising the hyperparameters


function selectModel(model_str::String)
    if model_str == "Chan2020minn"
        model_type = Chan2020minn_type()
    elseif model_str == "Chan2020iniw"
        model_type = Chan2020iniw_type()        
    elseif model_str == "Chan2020csv"
        model_type = Chan2020csv_type()
    elseif model_str == "CCK2016csv"
        model_type = CCK2016csv_type()
    elseif model_str == "BGR2010"
        model_type = BGR2010_type()
    elseif model_str == "CPZ2023"
        model_type = CPZ2023_type()
    elseif model_str == "CPZ2023iniw"
        model_type = CPZ2023iniw_type()
    elseif model_str == "Blagov2025"
        model_type = Blagov2025_type()
    elseif model_str == "Blagov2027"
        model_type = Blagov2027_type()
    else
        error("Model not found, make sure the spelling is completely correct, upper and lowercase matters!\n Possible models are: \n    BGR2010 \n    Chan2020minn\n    Chan2020csv\n    Chan2020iniw\n CPZ2023\n")
    end
    return model_type
end



# structure initializing the VAR
# Constructor
@doc raw"""
    
"""
@with_kw struct LoopSetup{MT<:BVARmodelType,SS<:BVARmodelSetup,HS<:BVARmodelHypSetup,DS<:BVARmodelDataSetup} <: BVARmodelLoopSetup
    model_type::MT
    set_struct::SS
    hyp_struct::HS
    data_struct::DS
end


# Structure for the datasets for the standard BVARs
@with_kw struct dataBVAR_TA <: BVARmodelDataSetup
    data_tab::TimeArray                                             # data for the high-frequency variables
    var_list::Array{Symbol,1}                                          # Symbol vector with the variable names, will be used for ordering
end

# Structure for the datasets for the standard BVARs
@with_kw struct data_BVAR{T <: AbstractFloat, N, D, A <: AbstractArray{T, N}} <: BVARmodelDataSetup
    data_tab::TimeArray{T,N,D,A}                                             # data for the high-frequency variables
    data_mat::Array{T,N}
    var_list::Array{Symbol,1}                                          # Symbol vector with the variable names, will be used for ordering
end


# Structure for the datasets of a mixed-frequency BVAR
@doc raw"""
    data_MFVAR_crossSect()

See also `makeDataSetup`.
"""
@with_kw struct data_MFVAR_crossSect{T <: AbstractFloat, D, N1, A1 <: AbstractArray{T, N1}, N2, A2 <: AbstractArray{T, N2}, N3, A3 <: AbstractArray{T, N3}, N4, A4 <: AbstractArray{T, N4}} <: BVARmodelDataSetup_MFcs
    dataHF_tab::TimeArray{T,N1,D,A1}                        # data for the high-frequency variables
    dataLF_tab::TimeArray{T,N2,D,A2}                        # data for the low-frequency variables
    z_cs_tab::TimeArray{T,N3,D,A3}                          # observations for cross-sectional constraint
    dataCSweights_tab::TimeArray{T,N4,D,A4}                      # weights for the low-frequency variables that sum on each row to 1
    var_list::Vector{Symbol}                                # Symbol vector with the variable names, will be used for ordering
    var_listHF::Vector{Symbol}                              # Symbol vector with the high-frequency variables
    var_listLF::Vector{Symbol}                              # Symbol vector with the high-frequency variables
    cs_scVal;                                               # error tolerance for the cross-sectional constraint
    endog_flag::Bool    
end


# Structure for the datasets of a mixed-frequency BVAR with cross-sectional restriction
@doc raw"""
    data_MFVAR_2f(data_HF::TimeArray,data_LF::TimeArray,var_list::Array{Symbol,1})

Generate a dataset strcture for use with CPZ2023 model

# Arguments
    dataHF_tab: TimeArray with your high-frequency variables (monthly or quarterly, respectively)
    dataLF_tab: TimeArray with your low-frequency variables (quarterly or yearly, respectively)
    var_list:   the variable order. Note that the functions that call these variables allow this to be optional.

See also `makeDataSetup`.
"""
@with_kw struct data_MFVAR_2f{T <: AbstractFloat, N1, D, A1 <: AbstractArray{T, N1}, N2, A2 <: AbstractArray{T, N2}} <: BVARmodelDataSetup_MF
    dataHF_tab::TimeArray{T,N1,D,A1}                        # data for the high-frequency variables
    dataLF_tab::TimeArray{T,N2,D,A2}                        # data for the low-frequency variables
    var_list::Vector{Symbol}                                # Symbol vector with the variable names, will be used for ordering}
    var_listHF::Vector{Symbol}                              # Symbol vector with the high-frequency variables
    var_listLF::Vector{Symbol}                              # Symbol vector with the high-frequency variables
end



# Structure to hold the median, 68% and 90% percentiles of the forecasts for a VAR
@with_kw struct data_fcast_PI{T <: AbstractFloat, N, D, A <: AbstractArray{T, N}} <: BVARmodelDataSetup
    YYfor_low05_tab::TimeArray{T,N,D,A}         
    YYfor_low16_tab::TimeArray{T,N,D,A}         
    YYfor_med_tab::TimeArray{T,N,D,A}         
    YYfor_hih84_tab::TimeArray{T,N,D,A}
    YYfor_hih95_tab::TimeArray{T,N,D,A}
end

# Structure to hold the median, 68%, 90%, and 95% percentiles of the forecasts for a VAR
struct probability_intervals_struct{T <: AbstractFloat, N, D, A <: AbstractArray{T, N}} <: BVARmodelDataSetup
    YYfor_low025_tab::TimeArray{T,N,D,A}         
    YYfor_low05_tab::TimeArray{T,N,D,A}         
    YYfor_low16_tab::TimeArray{T,N,D,A}         
    YYfor_med_tab::TimeArray{T,N,D,A}         
    YYfor_hih84_tab::TimeArray{T,N,D,A}
    YYfor_hih95_tab::TimeArray{T,N,D,A}
    YYfor_hih975_tab::TimeArray{T,N,D,A}
end

# main setup structure for a BEAVAR model
@with_kw struct VARSetup <: BVARmodelSetup
    p::Int          # number of lags
    n_save::Int      # gibbs to save
    n_burn::Int      # gibbs to burn
    n_irf::Int      # number of impulse responses
    n_fcst::Int     # number of forecast periods
    const_loc::Int  # location of the constant
    prior_RW::Int   # 1 for random walk prior (Y = B1*Y_tm1 + .... with B1 being identity matrix), or 0 for B1 = zeros(n,n), #TODO add for OLS (persistent), #TODO add for mix
    est_hyp::Bool   # set to true to estimate hyperparameters
    show_prog::Bool # set to true to show the progressbar (default)
end

# types for forecast output export
@with_kw struct VARForecast{T <: AbstractFloat, N, D, A <: AbstractArray{T, N}}  <: BVARforecastOutput
    Yfor3d::Array{T,3}            # 3D array with the data and forecasts. Dimensions are (T+n_fcst) x n x n_save
    data_tab::TimeArray{T,N,D,A}  # dataset in a TimeArray format   
    var_list::Array{Symbol,1}  # variable names for the forecasts
    n_fcst::Int                # number of forecast periods
end


# types for forecast output export
# this structure copies the things from store_YY. This is suboptimal for MF models because those are large matrices.
# Should be depreciated
@with_kw struct VAR_MF_Forecast <: BVARforecastOutput
    YforHF3d::Array{}          # 3D array with the high frequency forecasts. Dimensions are (p+n_fcst*) x n x n_save
    YforLF3d::Array{}          # 3D array with the low frequency forecasts. Dimensions are (p+n_fcst) x n x n_save
    dataHF_tab::TimeArray      # High frequency dataset in a TimeArray format   
    dataLF_tab::TimeArray      # Low frequency dataset in a TimeArray format 
    var_list::Array{Symbol,1}  # variable names for the forecasts
    n_fcst::Int                # number of forecast periods of the low-frequency variable
    YYforHF_struct::BVARmodelDataSetup     # structure with the percentiles of the forecasts for the high-frequency variables
    YYforLF_struct::BVARmodelDataSetup     # structure with the percentiles of the forecasts for the low-frequency variables
    data_flags_vec::BitVector              # bit vector showing the position of the data in low-frequency in the output. Only supports balanced z_tab 
    forecast_flags_vec::BitVector          # bit vector showing the position of the low-frequency forecasts in the output
end





@with_kw struct EvalForecast{T<:AbstractFloat,N} <: BVARmodelEval
    fe_mat::Array{T,N}          # forecast errors matrix
    sfe_mat::Array{T,N}         # forecast errors matrix
    ae_mat::Array{T,N}          # forecast errors matrix
    pred_lik_mat::Array{T,N}          
    list_keys::Vector{String,}      
    list_dates::Vector{Date,}  
    var_eval_list::Vector{Symbol}
end


@with_kw struct fcast_metrics{T<:AbstractFloat,N} <: BVARmodelEval
    h_string::Vector{String}       # String with forecast horizons ("T+...")
    msfe_mat::Array{T,N}          # mean squarte foreacst error
    rmsfe_mat::Array{T,N}         # root mean squarte foreacst error
    mafe_mat::Array{T,N}          # mean absolute error
    apl_mat::Array{T,N}          # average predictive likelihood
    var_eval_list::Vector{Symbol}
end


@doc raw"""
    model_type, set_struct, hyp_struct = makeSetup(model_str::String; p::Int=4,n_burn::Int=1000,n_save::Int=1000,n_irf::Int=16,n_fcst::Int = 8,hyp::BVARmodelHypSetup=hypDefault_struct())
    
Specify a model and generate structures for the Bayesian VAR and the hyperparameters.

Only the first argument is mandatory, rest is optional with default values.

# Arguments
    model_str: String, currently supported are "CPZ2023", "Chan2020minn", "Chan2020csv", "Chan2020iniw", "BGR2010"
    p:         number of lags, default is 4
    n_burn:    number of burn-in draws that will be discarded, default is 2000
    n_save:    number of retained draws (total is then n_burn + n_save), default is 1000
    n_irf:     horizon of impulse responses, default is 16
    n_fcst:    horizon of forecasting periods of the lowest frequency variable, default is 8
    hyp:       hyperparameter structure populated with default values for each model. See the relevant papers/documentation for details. To generate your own see the relevant structures below.

See also [`hypChan2020`](@ref), [`hypBGR2010`](@ref).
"""
function makeSetup(model_str::String;p::Int=4,n_burn::Int=1000,n_save::Int=1000,n_irf::Int=16,n_fcst::Int = 8,hyp::BVARmodelHypSetup=hypDefault_struct(),prior_RW::Int = 0,est_hyp=false,show_prog=true)
    model_type = BEAVARs.selectModel(model_str)
    # checking if user supplied the hyperparameter structure
    if isa(hyp,hypDefault_struct)                        # if not supplied, make a default one
        hyp_struct = BEAVARs.makeHypSetup(model_type);   # println("using the default hyperparameters")
    else                                                # else use supplied    
        hyp_struct = hyp; # println("using the supplied parameters")
    end

    intercept = BEAVARs.selectConstLoc(model_str);
    
    set_struct = BEAVARs.VARSetup(p,n_burn,n_save,n_irf,n_fcst,intercept,prior_RW,est_hyp,show_prog);
    return model_type, set_struct, hyp_struct
end

function unpackLoopSetup(loop_struct::BVARmodelLoopSetup)
    @unpack model_type, set_struct, hyp_struct, data_struct,  = loop_struct 
    return model_type, set_struct, hyp_struct, data_struct
end


function selectConstLoc(model_str::String)
    if model_str == "Chan2020csv"
        const_loc = 1
    elseif model_str == "Chan2020minn"
        const_loc = 1
    elseif model_str == "Chan2020iniw"
        const_loc = 1
    elseif model_str == "BGR2010"
        const_loc = 0
    elseif model_str == "CPZ2023"
        const_loc = 1
    elseif model_str == "CPZ2023dev"
        const_loc = 1
    elseif model_str == "CCK2016csv"
        const_loc = 1
    elseif model_str == "Blagov2025"
        const_loc = 1
    elseif model_str == "Blagov2027"
        const_loc = 1
    elseif model_str == "CPZ2023iniw"
        const_loc = 1
    else
        error("Constant location (right or left of X) is not defined for this model.\n Either define it in selectConstLoc function or report the bug")
    end
    return const_loc
end




include("funcs_dataPrep.jl")
include("funcs_init.jl")
include("gigrnd.jl")
include("irfs.jl")
include("funcs_plotting.jl")
include("funcs_mf.jl")
include("funcs_forecasting.jl")
include("Chan2020base.jl")
include("Chan2020new_funcs.jl")
include("Chan2020minn.jl")
include("Chan2020iniw.jl")
include("Chan2020csv.jl")
include("CCK2016csv.jl")
include("CPZ2023.jl")
include("CPZ2023iniw.jl")
include("Blagov2027.jl")
include("BGR2010.jl")


#-------------------------------------
# The Den: this is where the beavars live
#-------------------------------------


"""
    beavars(vint_in_dict::ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup})

Serial version of a function to estimate VAR models across different specifications/vintages.
Useful when one of your vintages leads to errors and you have no idea which one fails.

See also `beavars_weave`
"""
function beavars(vint_in_dict::ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup})
    vint_out_dict = ThreadSafeDict{String,BEAVARs.BVARmodelOutput}()
    # fcast_out_dict = ThreadSafeDict{String,BEAVARs.BVARforecastOutput}()
    prog_bar = Progress(length(vint_in_dict); dt=1.0, desc="Parallel execution of the vintages estimation...",barglyphs=BarGlyphs("[=> ]"), color=:blue)   # minimum update interval: 1 second
    
    @showprogress for (index, value) in pairs(vint_in_dict)
        # println("$index $value")
        println("Estimating data vintage $index")
        model_type, set_struct, hyp_struct, data_struct = BEAVARs.unpackLoopSetup(value);
        out_struct = beavar(model_type, set_struct, hyp_struct, data_struct);
        # fcast_struct = BEAVARs.forecast(out_struct,set_struct,data_struct);
        vint_out_dict[index] = out_struct;
        # fcast_out_dict[index] = fcast_struct;
    end
    
    # return vint_out_dict, fcast_out_dict
    finish!(prog_bar)
    return vint_out_dict
end



"""
        beavars_weave_dep(vint_in_dict::ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup})


        The parallel alternative to beavars function. Should be depreciated as another one is implemented
        that does not write into the same ThreadSafeDict.
"""
function beavars_weave_dep(vint_in_dict::ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup})
    vint_out_dict = ThreadSafeDict{String,BEAVARs.BVARmodelOutput}()
    fcast_out_dict = ThreadSafeDict{String,BEAVARs.BVARforecastOutput}()
    ks = collect(keys(vint_in_dict))
    # Threads.@threads for (index, value) in pairs(vint_in_dict)
    Threads.@threads for index in ks
        # println("$index $value")
        println("Estimating data vintage $index")
        # value=vint_in_dict[index];
        @unpack model_type, set_struct, hyp_struct, data_struct,  = vint_in_dict[index];
        # model_type, set_struct, hyp_struct, data_struct = BEAVARs.unpackLoopSetup(value);
        out_struct = beavar(model_type, set_struct, hyp_struct, data_struct);
        # fcast_struct = BEAVARs.forecast(out_struct,set_struct,data_struct);
        vint_out_dict[index] = out_struct;
        # fcast_out_dict[index] = fcast_struct;
    end
    
    # return vint_out_dict, fcast_out_dict
    return vint_out_dict
end

"""
    beavars_weave(vint_in_dict::ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup})

Parallel function to estimate VAR models across different specifications/vintages.
There is a serial version called beavers that can also be useful if one of your model fails but you don't know which one.

See also `beavars`

"""
function beavars_weave(vint_in_dict::ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup})
    jobs = [(key, vint_in_dict[key]) for key in keys(vint_in_dict)]
    njobs = length(jobs);
    results_vec = Vector{Tuple{String, BEAVARs.BVARmodelOutput}}(undef, njobs)

    prog_bar_weave = Progress(njobs; dt=1.0, desc="Parallel execution of the vintages estimation...",barglyphs=BarGlyphs("[=> ]"), color=:blue)   # minimum update interval: 1 second
    Threads.@threads :greedy for i in eachindex(jobs)
        key, loop_setup = jobs[i]
        model_type, set_struct, hyp_struct, data_struct =
            BEAVARs.unpackLoopSetup(loop_setup)
        output = BEAVARs.beavar(model_type, set_struct, hyp_struct, data_struct)
        update!(prog_bar_weave,i)
        results_vec[i] = (key, output)
    end

    vint_out_dict = ThreadSafeDict{String, BEAVARs.BVARmodelOutput}()
    for (key, output) in results_vec
        vint_out_dict[key] = output
    end
    finish!(prog_bar_weave)
    # return vint_out_dict, fcast_out_dict
    return vint_out_dict
end



#-------------------------------------
end # END OF MODULE
#-------------------------------------
