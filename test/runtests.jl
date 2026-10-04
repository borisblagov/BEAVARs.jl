using BEAVARs
using Test
using TimeSeries
using Parameters
using LinearAlgebra, Statistics
@testset "BEAVARs.jl" begin
    # Write your tests here.
    @test 1 == 1

    # Test for whether the XSurFormMatrix structure gives the correct result
    @testset "testset_sur.jl" begin
        include("testset_sur.jl")    
    end

    # @testset "testset_CPZ2023.jl" begin
    #     include("testset_CPZ2023.jl")    
    # end
end



