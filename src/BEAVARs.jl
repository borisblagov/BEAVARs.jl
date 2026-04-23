module BEAVARs
using   LinearAlgebra, 
        Distributions, 
        SparseArrays,
        TimeSeries, 
        Parameters,
        ProgressMeter,
        XLSX,
        ThreadSafeDicts

# from init_functions.jl
export mlag, mlagL, mlagL!, percentile_mat

# from Banbura, Giannone, Reichling 2010
#export makeDummiesMinn!, makeDummiesSumOfCoeff!, getBeta!, getSigma!, gibbs_beta_sigma,trainPriors, BGR2010, hypBGR2010

# from irfs.jl
export irf_chol, irf_chol_overDraws, irf_chol_overDraws_csv

export hypChan2020, hypBGR2010

export beavar, beavars, makeOutput, makeSetup, makeHypSetup, makeDataSetup, LoopSetup

# Structures, to be uncommented later
# export BVARmodelSetup, BVARmodelOutput, Chan2020csv_type, Chan2020minn_type, BVARmodelHypSetup, hypDefault_strct, outChan2020csv, BVARModelType, VARSetup




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
struct Chan2020csv_type2 <: BVARmodelType end
struct BGR2010_type <: BVARmodelType end
struct CPZ2023_type <: BVARmodelType end
struct Blagov2025_type <: BVARmodelType end

# Default structure for hyperparameters
struct hypDefault_strct <: BVARmodelHypSetup end    # empty structure for initialising the hyperparameters


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

@with_kw struct VARSetup <: BVARmodelSetup
    p::Int          # number of lags
    nsave::Int      # gibbs to save
    nburn::Int      # gibbs to burn
    n_irf::Int      # number of impulse responses
    n_fcst::Int     # number of forecast periods
    const_loc::Int  # location of the constant
end

# types for forecast output export
@with_kw struct VARForecast <: BVARforecastOutput
    Yfor3D::Array{}            # 3D array with the forecasts. Dimensions are (p+n_fcst) x n x nsave
    data_tab::TimeArray        # dataset in a TimeArray format   
    var_list::Array{Symbol,1}  # variable names for the forecasts
    n_fcst::Int                # number of forecast periods
end


# ------------------------
# MAIN FUNCTION 
# depreciated after moving to the syntax beavar(strcts)
# function beavar(model_str=model_name::String,YY_tup... ;p::Int=4,n_burn::Int=1000,n_save::Int=1000,n_irf::Int=16,n_fcst::Int = 8,hyp::BVARmodelHypSetup=hypDefault_strct())
#     model_type = BEAVARs.selectModel(model_str)
    
#     # checking if user supplied the hyperparameter structure
#     if isa(hyp,hypDefault_strct)                    # if not supplied, make a default one
#         hyp_strct = BEAVARs.makeHypSetup(model_type); # println("using the default hyperparameters")
#     else                                            # else use supplied    
#         hyp_strct = hyp; # println("using the supplied parameters")
#     end
        
#     out_strct, set_strct = dispatchModel(model_type,YY_tup, hyp_strct,p,n_burn,n_save,n_irf,n_fcst);
#     return out_strct, set_strct, hyp_strct
# end

@doc raw"""
    model_type, set_strct, hyp_strct = makeSetup(model_str::String; p::Int=4,n_burn::Int=1000,n_save::Int=1000,n_irf::Int=16,n_fcst::Int = 8,hyp::BVARmodelHypSetup=hypDefault_strct())
    
Specify a model and generate structures for the Bayesian VAR and the hyperparameters.

Only the first argument is mandatory, rest is optional with default values.

# Arguments
    model_str: String, currently supported are "CPZ2023", "Chan2020minn", "Chan2020csv", "Chan2020iniw", "BGR2010"
    p:         number of lags, default is 4
    n_burn:    number of burn-in draws that will be discarded, default is 2000
    n_save:    number of retained draws (total is then nburn + nsave), default is 1000
    n_irf:     horizon of impulse responses, default is 16
    n_fcst:    horizon of forecasting periods, default is 8
    hyp:       hyperparameter structure populated with default values for each model. See the relevant papers/documentation for details. To generate your own see the relevant structures below.

See also [`hypChan2020`](@ref), [`hypBGR2010`](@ref).
"""
function makeSetup(model_str::String;p::Int=4,n_burn::Int=1000,n_save::Int=1000,n_irf::Int=16,n_fcst::Int = 8,hyp::BVARmodelHypSetup=hypDefault_strct())
    model_type = BEAVARs.selectModel(model_str)
    # checking if user supplied the hyperparameter structure
    if isa(hyp,hypDefault_strct)                        # if not supplied, make a default one
        hyp_strct = BEAVARs.makeHypSetup(model_type);   # println("using the default hyperparameters")
    else                                                # else use supplied    
        hyp_strct = hyp; # println("using the supplied parameters")
    end

    intercept = BEAVARs.selectConstLoc(model_str);
    
    set_strct = BEAVARs.VARSetup(p,n_burn,n_save,n_irf,n_fcst,intercept);
    return model_type, set_strct, hyp_strct
end

function unpackLoopSetup(loop_strct::BVARmodelLoopSetup)
    @unpack model, set, hyp, data,  = loop_strct 
    model_type = model;
    set_strct = set;
    hyp_strct = hyp;
    data_strct = data;
    return model_type, set_strct, hyp_strct, data_strct
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

function beavar(::Chan2020minn_type, set_strct, hyper_str, data_strct)
    println("Hello Minn")
    YY = values(data_strct.data_tab);
    store_β, store_Σ = Chan2020minn(YY,set_strct,hyper_str);
    out_strct = VAROutput_Chan2020minn(store_β,store_Σ,YY)
    return out_strct
end



@doc raw"""
    Main function for Chan2020csv
"""
function beavar(::Chan2020csv_type, set_strct, hyper_str, data_strct)
    YY = values(data_strct.data_tab);
    store_β, store_h, store_Σ, s2_h_store, store_ρ, store_σ_h2, store_eh = Chan2020csv(YY,set_strct,hyper_str);
    out_strct = VAROutput_Chan2020csv(store_β,store_Σ,store_h,s2_h_store, store_ρ, store_σ_h2, store_eh,YY)
    return out_strct
end

function beavar(::Chan2020iniw_type, set_strct, hyper_str, data_strct)
    println("Hello Independent Normal Inverse Wishart")
    YY = values(data_strct.data_tab);
    store_β, store_Σ = Chan2020iniw(YY,set_strct,hyper_str);
    out_strct = VAROutput_Chan2020iniw(store_β,store_Σ,YY)
    return out_strct
end


function beavar(::CPZ2023_type, set_strct, hyp_strct, data_strct)
    println("Hello CPZ2023")
    @unpack dataHF_tab,dataLF_tab, aggMix, var_list = data_strct
    store_YY,store_β, store_Σt_inv, M_zsp, z_vec, Sm_bit,store_Σt, freq_mix_tp = CPZ2023(dataHF_tab,dataLF_tab,var_list,set_strct,hyp_strct,aggMix);
    out_strct = VAROutput_CPZ2023(store_β,store_Σt_inv,store_YY,M_zsp, z_vec, Sm_bit,store_Σt,var_list,freq_mix_tp);
    return out_strct
end


@doc raw"""
    Main function for Blagov2025
"""
function beavar(::Blagov2025_type, set_strct, hyp_strct, data_strct)
    println("Hello Blagov2025")
    @unpack dataHF_tab,dataLF_tab, aggMix, var_list = data_strct
    store_YY,store_β, store_Σt_inv, M_zsp, z_vec, Sm_bit,store_Σt, freq_mix_tp = Blagov2025(dataHF_tab,dataLF_tab,var_list,set_strct,hyp_strct,aggMix)    
    out_strct = VAROutput_CPZ2023(store_β,store_Σt_inv,store_YY,M_zsp, z_vec, Sm_bit,store_Σt,var_list,freq_mix_tp)
    return out_strct
end


function beavar(::BGR2010_type, set_strct, hyp_strct, data_strct)
    println("Hello BGR2010")
    YY = values(data_strct.data_tab);
    store_β, store_Σ = BGR2010(YY,set_strct,hyp_strct);
    out_strct = VAROutput_BGR2010(store_β,store_Σ,YY);
    return out_strct
end

function beavars(vint_in_dict::ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup})
    vint_out_dict = ThreadSafeDict{String,BEAVARs.BVARmodelOutput}()
    fcast_out_dict = ThreadSafeDict{String,Array{Float64, 3}}()
    for (index, value) in pairs(vint_in_dict)
        # println("$index $value")
        println("Estimating data vintage $index")
        model_type, set_strct, hyp_strct, data_strct = BEAVARs.unpackLoopSetup(value);
        out_strct = beavar(model_type, set_strct, hyp_strct, data_strct);
        YYfcast3D_mat = BEAVARs.forecast(out_strct,set_strct);
        vint_out_dict[index] = out_strct;
        fcast_out_dict[index] = YYfcast3D_mat;
    end
    
    return vint_out_dict, fcast_out_dict
end

function beavars_multi(vint_in_dict::ThreadSafeDict{String,BEAVARs.BVARmodelLoopSetup})
    vint_out_dict = ThreadSafeDict{String,BEAVARs.BVARmodelOutput}()
    fcast_out_dict = ThreadSafeDict{String,Array{Float64, 3}}()
    ks = collect(keys(vint_in_dict))
    # Threads.@threads for (index, value) in pairs(vint_in_dict)
    Threads.@threads for index in ks
        # println("$index $value")
        println("Estimating data vintage $index")
        value=vint_in_dict[index];
        model_type, set_strct, hyp_strct, data_strct = BEAVARs.unpackLoopSetup(value);
        out_strct = beavar(model_type, set_strct, hyp_strct, data_strct);
        YYfcast3D_mat = BEAVARs.forecast(out_strct,set_strct);
        vint_out_dict[index] = out_strct;
        fcast_out_dict[index] = YYfcast3D_mat;
    end
    
    return vint_out_dict, fcast_out_dict
end


#-------------------------------------
end # END OF MODULE
#-------------------------------------

