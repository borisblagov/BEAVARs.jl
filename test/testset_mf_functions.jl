@testset "cross-sectional mixed-frequency mapping" begin
    dates = Date(2020,1,1):Month(1):Date(2020,3,1)
    weights = TimeArray(
        dates,
        [0.75 99.0 0.25; 0.5 99.0 0.5; 0.2 99.0 0.8],
        [:b,:unused,:a],
    )
    missing_tab = TimeArray(dates,fill(NaN,3,2),[:a,:b])
    z = TimeArray(dates,[10.0,20.0,30.0],[:total])
    indices = [1 2; 3 4; 5 6]

    M, z_values, T_z = BEAVARs.makeM_crosssect(weights,indices,[:a,:b],missing_tab,z,6)

    @test T_z == 3
    @test z_values == [10.0,20.0,30.0]
    @test M == [
        0.25 0.75 0.0 0.0 0.0 0.0
        0.0 0.0 0.5 0.5 0.0 0.0
        0.0 0.0 0.0 0.0 0.8 0.2
    ]

    constant_weights = TimeArray([first(dates)],[0.4 0.6],[:a,:b])
    M_constant, _, _ = BEAVARs.makeM_crosssect(
        constant_weights,indices,[:a,:b],missing_tab,z,6,
    )
    @test M_constant[3,5:6] == [0.4,0.6]

    one_variable_weights = TimeArray(dates,[0.1,0.2,0.3],[:a])
    one_variable_missing = TimeArray(dates,fill(NaN,3,1),[:a])
    M_one_variable, _, _ = BEAVARs.makeM_crosssect(
        one_variable_weights,reshape([1,2,3],3,1),[:a],one_variable_missing,z,3,
    )
    @test M_one_variable == [0.1 0.0 0.0; 0.0 0.2 0.0; 0.0 0.0 0.3]

    incomplete_weights = weights[1:2]
    @test_throws ArgumentError BEAVARs.makeM_crosssect(
        incomplete_weights,indices,[:a,:b],missing_tab,z,6,
    )
end
