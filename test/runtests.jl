using BEAVARs
using Test
using TimeSeries
using Parameters
using LinearAlgebra, SparseArrays, Statistics
using BlockBandedMatrices, BlockArrays
@testset "BEAVARs.jl" begin
    # Write your tests here.
    @test 1 == 1

    

    # Test for whether the XSurFormMatrix structure gives the correct result
    @testset "testset_sur.jl" begin
        include("testset_sur.jl")    
    end



    @testset "testset_CPZ2023.jl" begin
        include("testset_CPZ2023.jl")    
    end

    @testset "testset_mf_functions.jl" begin
        include("testset_mf_functions.jl")
    end

    @testset "test_update_hyp_and_llm.jl" begin
        include("test_update_hyp_and_llm.jl")
    end
end
