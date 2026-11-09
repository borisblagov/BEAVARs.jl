using BEAVARs
using Test
using Dates
using TimeSeries
using ThreadSafeDicts
using Parameters
using LinearAlgebra, SparseArrays, Statistics
using BlockBandedMatrices, BlockArrays

struct TestBVARmodelOutput <: BEAVARs.BVARmodelOutput end

struct TestForecastOutput <: BEAVARs.BVARmodelOutput
    fdatesLF::Vector{Date}
    store_YY_LF::Array{Float64,3}
    freq_mix_tp::NTuple{3,Int}
end

struct TestForecastData{TA<:TimeArray} <: BEAVARs.BVARmodelDataSetup
    dataLF_tab::TA
    var_list::Vector{Symbol}
end

@testset "BEAVARs.jl" begin
    # Write your tests here.
    @test 1 == 1

    @testset "beavars_prep_eval accepts concrete dictionary value types" begin
        output_dict = ThreadSafeDict{String, TestBVARmodelOutput}()
        setup_dict = ThreadSafeDict{String, BEAVARs.LoopSetup}()

        @test hasmethod(
            BEAVARs.beavars_prep_eval,
            Tuple{typeof(output_dict), typeof(setup_dict), TimeArray},
        )
        @test hasmethod(
            BEAVARs.beavars_prep_eval2,
            Tuple{typeof(output_dict), typeof(setup_dict), TimeArray},
        )
    end

    @testset "forecast evaluation inference" begin
        dates = collect(Date(2000, 1, 1):Month(1):Date(2000, 5, 1))
        model_data = TimeArray(dates[1:3], reshape([1.0, 2.0, 3.0], 3, 1), [:x])
        true_data = TimeArray(dates[4:5], reshape([4.0, 5.0], 2, 1), [:x])
        forecasts = reshape(
            [3.8, 4.8, 4.2, 5.2, 3.9, 4.9, 4.1, 5.1, 4.0, 5.0],
            5,
            1,
            2,
        )
        output = TestForecastOutput(dates, forecasts, (0, 0, 0))
        data = TestForecastData(model_data, [:x])
        setup = BEAVARs.VARSetup(1, 2, 1, 0, 2, 1, 0, false, false)

        result = @inferred BEAVARs.eval_forecast(output, data, setup, true_data)
        @test result.fcastDatesOverlap == dates[4:5]

        loop_setup = BEAVARs.LoopSetup(
            BEAVARs.Blagov2027_type(),
            setup,
            BEAVARs.hypDefault_struct(),
            data,
        )
        output_dict = ThreadSafeDict{String,typeof(output)}()
        output_dict["v1"] = output
        setup_dict = ThreadSafeDict{String,typeof(loop_setup)}()
        setup_dict["v1"] = loop_setup
        prepared, _ = @inferred BEAVARs.beavars_prep_eval2(
            output_dict,
            setup_dict,
            true_data,
        )
        @test prepared.list_keys == ["v1"]

        input = TimeArray(dates, reshape([1.0, 2.0, 3.0, 4.0, 5.0], 5, 1), [:x])
        changes = @inferred BEAVARs._eval_percentchange(input)
        @test values(changes) == values(percentchange(input))

        vector_input = TimeArray(dates, [1.0, 2.0, 3.0, 4.0, 5.0])
        vector_changes = @inferred BEAVARs._eval_percentchange(vector_input)
        @test vector_changes isa TimeArray{Float64,1}
        @test values(vector_changes) ≈ values(percentchange(vector_input))
        @test timestamp(vector_changes) == timestamp(percentchange(vector_input))
    end


    # Test for whether the XSurFormMatrix structure gives the correct result
    @testset "testset_sur.jl" begin
        include("testset_sur.jl")    
    end



    @testset "testset_CPZ2023.jl" begin
        include("testset_CPZ2023.jl")    
    end

    @testset "test_update_hyp_and_llm.jl" begin
        include("test_update_hyp_and_llm.jl")
    end

    @testset "optimized impulse percentiles match reference" begin
        draws = reshape(Float64.(1:60), 4, 3, 5)

        reference = BEAVARs.get_imp_percentiles(draws)
        optimized = BEAVARs.get_imp_percentiles_llm(draws)

        @test length(reference) == length(optimized) == 7
        for (expected, actual) in zip(reference, optimized)
            @test actual == expected
        end

        draws_2d = reshape(Float64.(1:20), 4, 5)
        probs = (0.025, 0.05, 0.16, 0.5, 0.84, 0.95, 0.975)
        actual_2d = BEAVARs.get_imp_percentiles_llm(draws_2d)
        expected_2d = ntuple(i -> BEAVARs.percentile_mat(draws_2d, probs[i]; dims=2), 7)

        @test length(actual_2d) == 7
        for (expected, actual) in zip(expected_2d, actual_2d)
            @test actual == expected
            @test size(actual) == (size(draws_2d, 1),)
        end
    end

    @testset "test_mf_constraints.jl" begin
        include("test_mf_constraints.jl")
    end
end
