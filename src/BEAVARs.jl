module BEAVARs
using   LinearAlgebra, 
        Distributions, 
        SparseArrays,
        TimeSeries, 
        Parameters,
        ProgressMeter,
        XLSX,
        ThreadSafeDicts,
        Plots

# from init_functions.jl
export mlag, mlagL, mlagL!, percentile_mat

# from Banbura, Giannone, Reichling 2010
#export makeDummiesMinn!, makeDummiesSumOfCoeff!, getBeta!, getSigma!, gibbs_beta_sigma,trainPriors, BGR2010, hypBGR2010

# from irfs.jl
export irf_chol, irf_chol_overDraws, irf_chol_overDraws_csv

export hypChan2020, hypBGR2010

export beavar, beavars, beavars_weave, makeOutput, makeSetup, makeHypSetup, makeDataSetup, LoopSetup

# Structures, to be uncommented later
# export BVARmodelSetup, BVARmodelOutput, Chan2020csv_type, Chan2020minn_type, BVARmodelHypSetup, hypDefault_struct, outChan2020csv, BVARModelType, VARSetup




# Types for multiple dispatch across models
abstract type BVARmodelType end         # types for models
abstract type BVARmodelSetup end        # type for VAR setup parameters
abstract type BVARmodelHypSetup end     # types for hyperparameters
abstract type BVARmodelDataSetup end    # types for storing the data unputs to the models
abstract type BVARmodelLoopSetup end    # type for VAR setup parameters
abstract type BVARmodelOutput end       # type for output storage
abstract type BVARforecastOutput end    # type for forecast output storage


# Structures for multiple dispatch across models
struct Chan2020minn_type <: BVARmodelType end
struct Chan2020iniw_type <: BVARmodelType end
struct Chan2020iniw_type2 <: BVARmodelType end
struct Chan2020csv_type <: BVARmodelType end
# struct Chan2020csv_type2 <: BVARmodelType end
struct BGR2010_type <: BVARmodelType end
struct CPZ2023_type <: BVARmodelType end
struct Blagov2025_type <: BVARmodelType end

# Default structure for hyperparameters
struct hypDefault_struct <: BVARmodelHypSetup end    # empty structure for initialising the hyperparameters


function selectModel(model_str::String)
    if model_str == "Chan2020csv"
        model_type = Chan2020csv_type()
    elseif model_str == "Chan2020minn"
        model_type = Chan2020minn_type()
    elseif model_str == "Chan2020iniw"
        model_type = Chan2020iniw_type()
    elseif model_str == "BGR2010"
        model_type = BGR2010_type()
    elseif model_str == "CPZ2023"
        model_type = CPZ2023_type()
    elseif model_str == "Blagov2025"
        model_type = Blagov2025_type()
    else
        error("Model not found, make sure the spelling is completely correct, upper and lowercase matters!\n Possible models are: \n    BGR2010 \n    Chan2020minn\n    Chan2020csv\n    Chan2020iniw\n CPZ2023\n")
    end
    return model_type
end



# structure initializing the VAR
# Constructor
@doc raw"""
    
"""
@with_kw struct LoopSetup <: BVARmodelLoopSetup
    model::BVARmodelType
    set::BVARmodelSetup
    hyp::BVARmodelHypSetup
    data::BVARmodelDataSetup
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

# Structure to hold the median, 68% and 95% percentiles of the forecasts for a VAR
@with_kw struct data_fcast_PI <: BVARmodelDataSetup
    YYfor_low05_tab::TimeArray         
    YYfor_low16_tab::TimeArray         
    YYfor_med_tab::TimeArray         
    YYfor_hih84_tab::TimeArray
    YYfor_hih95_tab::TimeArray
end

@with_kw struct VARSetup <: BVARmodelSetup
    p::Int          # number of lags
    n_save::Int      # gibbs to save
    n_burn::Int      # gibbs to burn
    n_irf::Int      # number of impulse responses
    n_fcst::Int     # number of forecast periods
    const_loc::Int  # location of the constant
    prior_RW::Int   # 1 for random walk prior (Y = B1*Y_tm1 + .... with B1 being identity matrix), or 0 for B1 = zeros(n,n), #TODO add for OLS (persistent), #TODO add for mix
end

# types for forecast output export
@with_kw struct VARForecast{T <: AbstractFloat, N, D, A <: AbstractArray{T, N}}  <: BVARforecastOutput
    Yfor3d::Array{T,3}            # 3D array with the forecasts. Dimensions are (p+n_fcst) x n x n_save
    data_tab::TimeArray{T,N,D,A}  # dataset in a TimeArray format   
    var_list::Array{Symbol,1}  # variable names for the forecasts
    n_fcst::Int                # number of forecast periods
end


# types for forecast output export
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
function makeSetup(model_str::String;p::Int=4,n_burn::Int=1000,n_save::Int=1000,n_irf::Int=16,n_fcst::Int = 8,hyp::BVARmodelHypSetup=hypDefault_struct(),prior_RW::Int = 0)
    model_type = BEAVARs.selectModel(model_str)
    # checking if user supplied the hyperparameter structure
    if isa(hyp,hypDefault_struct)                        # if not supplied, make a default one
        hyp_struct = BEAVARs.makeHypSetup(model_type);   # println("using the default hyperparameters")
    else                                                # else use supplied    
        hyp_struct = hyp; # println("using the supplied parameters")
    end

    intercept = BEAVARs.selectConstLoc(model_str);
    
    set_struct = BEAVARs.VARSetup(p,n_burn,n_save,n_irf,n_fcst,intercept,prior_RW);
    return model_type, set_struct, hyp_struct
end

function unpackLoopSetup(loop_struct::BVARmodelLoopSetup)
    @unpack model, set, hyp, data,  = loop_struct 
    model_type = model;
    set_struct = set;
    hyp_struct = hyp;
    data_struct = data;
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
    elseif model_str == "Blagov2025"
        const_loc = 1
    else
        error("Constant location (right or left of X) is not defined for this model.\n Either define it in selectConstLoc function or report the bug")
    end
    return const_loc
end




include("dataPrep.jl")
include("init_functions.jl")
include("BGR2010.jl")
include("irfs.jl")
include("Chan2020base.jl")
include("Chan2020minn.jl")
include("Chan2020iniw.jl")
include("Chan2020csv.jl")
include("CPZ2023.jl")
include("Blagov2025.jl")
include("plot_functions.jl")


#-------------------------------------
# The Den: this is where the beavars live
#-------------------------------------




@doc raw"""
    Main function for Chan2020csv
"""
function beavar(::Chan2020csv_type, set_struct, hyp_str, data_struct)
    println("Hello Minn CSV")
    YY = values(data_struct.data_tab);
    store_β, store_h, store_Σ, s2_h_store, store_ρ, store_σ_h2, store_eh = Chan2020csv(YY,set_struct,hyp_str);
    out_struct = VAROutput_Chan2020csv(store_β,store_Σ,store_h,s2_h_store, store_ρ, store_σ_h2, store_eh,YY)
    return out_struct
end


function beavar(::CPZ2023_type, set_struct, hyp_struct, data_struct)
    println("Hello CPZ2023")
    @unpack dataHF_tab,dataLF_tab, var_list = data_struct
    store_YY,store_β, store_Σt_inv, M_zsp, z_vec, Sm_bit,store_Σt, freq_mix_tp,M_inter_agg, fdatesHF, fdatesLF = CPZ2023(dataHF_tab,dataLF_tab,var_list,set_struct,hyp_struct);
    out_struct = VAROutput_CPZ2023(store_β,store_Σt_inv,store_YY, M_zsp, z_vec, Sm_bit,store_Σt,var_list,freq_mix_tp,M_inter_agg, fdatesHF, fdatesLF);
    return out_struct
end


@doc raw"""
    Main function for Blagov2025
"""
function beavar(::Blagov2025_type, set_struct, hyp_struct, data_struct)
    println("Hello Blagov2025")
    @unpack dataHF_tab,dataLF_tab, var_list = data_struct
    store_β, store_Σt_inv, store_YY, M_zsp, z_vec, Sm_bit, freq_mix_tp, store_Σt, store_h, store_s2_h, store_ρ, store_σ_h2, store_eh, M_inter_agg, fdatesHF, fdatesLF = Blagov2025(dataHF_tab,dataLF_tab,var_list,set_struct,hyp_struct)    
    out_struct = VAROutput_Blagov2025(store_β, store_Σt_inv, store_YY, M_zsp, z_vec, Sm_bit, freq_mix_tp, store_Σt, store_h, store_s2_h, store_ρ, store_σ_h2, store_eh, M_inter_agg, fdatesHF, fdatesLF)
    return out_struct
end


function beavar(::BGR2010_type, set_struct, hyp_struct, data_struct)
    println("Hello BGR2010")
    YY = values(data_struct.data_tab);
    store_β, store_Σ = BGR2010(YY,set_struct,hyp_struct);
    out_struct = VAROutput_BGR2010(store_β,store_Σ,YY);
    return out_struct
end

function beavars(vint_in_dict::ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup})
    vint_out_dict = ThreadSafeDict{String,BEAVARs.BVARmodelOutput}()
    fcast_out_dict = ThreadSafeDict{String,BEAVARs.BVARforecastOutput}()
    for (index, value) in pairs(vint_in_dict)
        # println("$index $value")
        println("Estimating data vintage $index")
        model_type, set_struct, hyp_struct, data_struct = BEAVARs.unpackLoopSetup(value);
        out_struct = beavar(model_type, set_struct, hyp_struct, data_struct);
        fcast_struct = BEAVARs.forecast(out_struct,set_struct,data_struct);
        vint_out_dict[index] = out_struct;
        fcast_out_dict[index] = fcast_struct;
    end
    
    return vint_out_dict, fcast_out_dict
end

function beavars_weave(vint_in_dict::ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup})
    vint_out_dict = ThreadSafeDict{String,BEAVARs.BVARmodelOutput}()
    fcast_out_dict = ThreadSafeDict{String,BEAVARs.BVARforecastOutput}()
    ks = collect(keys(vint_in_dict))
    # Threads.@threads for (index, value) in pairs(vint_in_dict)
    Threads.@threads for index in ks
        # println("$index $value")
        println("Estimating data vintage $index")
        value=vint_in_dict[index];
        model_type, set_struct, hyp_struct, data_struct = BEAVARs.unpackLoopSetup(value);
        out_struct = beavar(model_type, set_struct, hyp_struct, data_struct);
        fcast_struct = BEAVARs.forecast(out_struct,set_struct,data_struct);
        vint_out_dict[index] = out_struct;
        fcast_out_dict[index] = fcast_struct;
    end
    
    return vint_out_dict, fcast_out_dict
end


#-------------------------------------
end # END OF MODULE
#-------------------------------------

