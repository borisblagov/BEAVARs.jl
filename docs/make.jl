using Pkg
Pkg.activate(@__DIR__)
using Documenter, BEAVARs, TimeSeries, LiveServer, DocumenterTools

makedocs(
    sitename="BEAVARs.jl",
    pages = [
        "Introduction" => "introduction.md",
        "Estimation" => [
            "Chan2020minn" => "Chan2020minn.md",
            "Chan2020iniw" => "Chan2020iniw.md",
        ],
        "Forecasting" => "forecasting.md",
        "Structural analysis" => "irfs.md",
        "File library" => [
            "Constructors" => "Constructors.md",
            "Initialization" => "init_functions.md",
            "Data transformation" => "dataPrep.md",
        ]
    ]
)
deploydocs(
    target = "build",
    repo = "github.com/borisblagov/BEAVARs.jl.git",
)