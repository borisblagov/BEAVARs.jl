@testset "mixed-frequency constraint builders return identical entities" begin
    hf_dates = Date(2020, 1, 1):Month(1):Date(2020, 12, 1)
    lf_dates = Date(2020, 1, 1):Month(3):Date(2020, 10, 1)
    dataHF_tab = TimeArray(hf_dates, reshape(Float64.(1:length(hf_dates)), :, 1), [:hf])
    dataLF_tab = TimeArray(lf_dates, reshape(Float64.(1:length(lf_dates)), :, 1), [:lf])

    model_type, set_struct, _ = makeSetup(
        "Blagov2027";
        p=2,
        n_burn=0,
        n_save=0,
        n_fcst=0,
        prior_RW=0,
        show_prog=false,
    )
    data_struct = makeDataSetup(
        model_type,
        dataHF_tab,
        dataLF_tab;
        var_list=[:hf, :lf],
    )

    old_entities = BEAVARs.make_mf_constraints_old(set_struct, data_struct)
    new_entities = BEAVARs.make_mf_constraints(set_struct, data_struct)

    @test length(old_entities) == length(new_entities) == 11
    for (old_entity, new_entity) in zip(old_entities, new_entities)
        @test isequal(old_entity, new_entity)
    end
end