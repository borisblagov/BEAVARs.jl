# General introduction

## Installation
Installing the package follows the typical Julia scheme
```julia
julia> ]
pkg> add BEAVARs
pkg>
```
and, after pressing backspace  to get back to the julia terminal, typing
```julia
julia> using BEAVARs
```


## Usage overview
The main function of the package is
```julia
beavar(model_type, set_strct, hyp_strct, data_strct)
```
which calls the relevant models and performs the estimation. Using the package boils down to the correct specification of these arguments, for which special functions exist to help you create them. Before jumping in the details let's give a brief overview.

- `model_type`: An object that allows Julia to know which model you want to use and call the relevant functions. 
- `set_strct`:  A structure containing the general VAR setup such as number of lags, number of draws, etc.
- `hyp_strct`:  A structure for setting the hyperparameters for the Bayesian estimation.
- `data_strct`: A structure containing your data.

The first three objects are generated using a helper function `make_setup`. The fourth object is generated using the helper function `makeDataSetup()`. Let us showcase each of these by estimating a simple example based on the Minnesota prior.



## Quickstart

We start by loading the packages that we need for this example: `TimeSeries` and `Dates`. `TimeSeries` is needed to generate a specific `TimeArray` structure so it is required.  The package `Dates` is used only to generate random data for this example and is typically not required - you will probably load your own data. 

!!! note "Installing additional packages"
    The first time you run this code, you probably don't have these packages installed. Install them using the usual Julia approach.
    ```julia
    julia> ]
    pkg> add TimeSeries
    pkg> add Dates
    ```
    pressing backspace now lands you back at the `julia>` terminal

Add the packages with `using`

```julia
julia> using BEAVARs, TimeSeries
julia> using Dates     # these are required only for the example, your data may already have a time-series format
```

### Using the `makeSetup` function

Next we will create the setup for the VAR model using the `makeSetup()` function. The function needs a specific string as input to know which structures to initialize and we need to figure the synthax out. 

!!! note "Getting help"
    In Julia you can get help for functions by typing a question mark `?` in the Julia terminal, which switches to the `help>` mode and then typing the function name (without brackets).

Let us fetch the documentation using the help mode.
```julia
julia> ?
help?> makeSetup
```

```@docs
makeSetup
```

We can see in the help that for the first argument `model_str` one of the permissive values is "Chan2020minn" which is the model we want to estimate here.For our example, we will also change the number of lags to `p=2`, as well as have a very small `burn in = 20` and `n_save = 50`. We will not change any hyperparameters, therefore not pass any structure `hyp` to the function `makeSetup`. 

!!! note "Julia function call synthax"
    To know which function inputs are mandatory look for a semi-colon `;` in the definition. Everything after the semicolon is optional and has default values. E.g. `functionName(a::String; a::Int=2,b=3)` means that typing `functionName("test")` is equivalent to typing `functionName("test"; a=2, b=3)` and `functionName("test"; a=5)` is equivalent to `functionName("test"; a=25, b=3)`. Also note that when you *call* the function, you do not have to respect the semi-colon, e.g. `functionName("test"; a=5)` and `functionName("test", a=5)` are both supported.

Let's call the function `makeSetup` with our specifications.
```julia
julia> model_type, set_strct, hyp_strct = makeSetup("Chan2020minn";n_burn=20,n_save=50,p=2)
```

After pressing enter we are greeted with the following output, which is a bit hard to read but if you follow the commas you will see that we have generated three outputs: (1) `BEAVARs.Chan2020minn_type()`; (2) a structure `BEAVARs.VARSetup` with some general VAR settings; (3) a structure `hypChan2020` with a lot of hyperparameters; . We will not go into details why some of these are prefaced with `BEAVARs.` and others are not.

```julia
(BEAVARs.Chan2020minn_type(), BEAVARs.VARSetup
  p: Int64 2
  nsave: Int64 20
  nburn: Int64 50
  n_irf: Int64 16
  n_fcst: Int64 8
  const_loc: Int64 1
, hypChan2020
  c1: Float64 0.04
  c2: Float64 0.01
  c3: Float64 100.0
  ρ: Float64 0.8
  σ_h2: Float64 0.1
  v_h0: Float64 5.0
  S_h0: Float64 0.04
  ρ_0: Float64 0.9
  V_ρ: Float64 0.04
  q: Float64 0.5
  nu0: Int64 3
)
```

From now on, we will not use the string `"Chan2020minn"` but always use the binding `model_type` if we ever need to call a function that is model specific.

```julia
julia> model_type
BEAVARs.Chan2020minn_type()
```

We can inspect the other elements as well by typing their binding, e.g.
```julia
julia> set_strct
BEAVARs.VARSetup
  p: Int64 2
  nsave: Int64 20
  nburn: Int64 50
  n_irf: Int64 16
  n_fcst: Int64 8
  const_loc: Int64 1
```

!!! note "Changes to these structures"
    Suppose we wanted to change some of the settings. It might be logical for you to then try something like `set_strct(p=2)` but this is not the way to go. Use the function `makeSetup` again. Supose we actually wanted 3 lags, we can do:
    ```julia
    model_type, set_strct, hyp_strct = makeSetup("Chan2020minn";n_burn=20,n_save=50,p=3)
    ```
    Do not forget to add the settings we are **not** changing, in the above example `n_burn` and `n_save`.



### Loading the data

The next and final step is to load our data. I am constantly amazed how hard it is to load data in programming languages made for **scientific analysis**. You know, where analysing data is the main thing that you do! This is not a tutorial on that, and you might need additional packages to import your own data (e.g. `CSV`, `XLSX`, and countless others). 

For this tutorial we will generate random data. 

Moreover, Vector Autoregressions are typically used for time-series, which are data where each value corresponds to a specific time point. This package uses a speicific type `TimeArray` (provided by the above installed `TimeSeries`) to deal with this. 

First generate 30 random observations from 3 variables and 30 random date values using the following command. Don't forget to check the docs for `TimeArray` like we did above with `julia>?` and `help>TimeArray`.

```julia
julia> data = TimeArray(DateTime(2020,1,1):Quarter(1):DateTime(2027,4,1),rand(30,3))
30×3 TimeArray{Float64, 2, DateTime, Matrix{Float64}} 2020-01-01T00:00:00 to 2027-04-01T00:00:00
┌─────────────────────┬────────────┬───────────┬──────────┐
│                     │ A          │ B         │ C        │
├─────────────────────┼────────────┼───────────┼──────────┤
│ 2020-01-01T00:00:00 │ 0.00817292 │  0.939333 │ 0.372302 │
│ 2020-04-01T00:00:00 │   0.420362 │ 0.0207827 │ 0.134192 │
│          ⋮          │     ⋮      │     ⋮     │    ⋮     │
│ 2027-04-01T00:00:00 │   0.942019 │  0.414029 │ 0.208983 │
└─────────────────────┴────────────┴───────────┴──────────┘
                                            27 rows omitted

```
Since the data will be random, the above numbers will not correspond to yours. That is fine.

!!! note "Importing your own data from csv"
    If your data is in a csv file, the `TimeSeries` package provides the function `readtimearray()` to help.

Now that we have our data in the format that we need, we can actually call the main function to generate the necessary structures, `makeDataSetup`

```@docs
makeDataSetup(::BEAVARs.Chan2020minn_type,::TimeSeries.TimeArray)
```

It takes two mandatory inputs and one optional one. We will only supply the first two: `model_type` variable and `data`.

```julia
julia> data_strct = makeDataSetup(model_type,data)
BEAVARs.dataBVAR_TA
  data_tab: TimeArray{Float64, 2, DateTime, Matrix{Float64}}
  var_list: Array{Symbol}((3,))
``` 
We did not specify any variable names, thus `var_list` will simply take the names from the `TimeArray`. Note that this list is important only in very few specific circumstances such as calculating IRFs using the Cholesky decomposition, where the ordering of the variables matters. You can still use it to reorder the variables before estimation for plots.

### Estimating the model

Now we are ready to estimate the model. The generic function is 
```julia
julia> out_strct = beavar(model_type, set_strct, hyp_strct, data_strct);
```

**That's it!** `out_strct` contains the relevant output from the model and is used as input for further analyses such as forecasts or structural analysis (impulse response functions).

```julia
julia> out_strct
BEAVARs.VAROutput_Chan2020minn
  store_β: Array{Float64}((21, 20)) [0.32320381087683125 0.5761295647868285 … 0.4607520139245541 0.42639561226264844; -0.06868758161342711 -0.01088669918413799 … -0.07791244344526811 -0.2605076421733443; … ; -0.06689834851525589 -0.005589195016996569 … 0.002709639906057543 -0.052079260901279636; -0.05091553975975063 -0.17067934162978135 … 0.0682082052074954 0.12190155764719483]
  store_Σ: Array{Float64}((9, 20)) [0.08287691219041017 0.08287691219041017 … 0.08287691219041017 0.08287691219041017; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.08246390271378752 0.08246390271378752 … 0.08246390271378752 0.08246390271378752]
  YY: Array{Float64}((30, 3)) [0.008172916632652849 0.9393327809093057 0.3723016243970907; 0.4203621377519353 0.02078266995620126 0.13419170214720488; … ; 0.831104672828403 0.7002479005194212 0.742052576571863; 0.9420190363481958 0.41402905467680584 0.20898335466884976]
```

!!! note "Julia is a compiled language"
    Julia compiles the function the first time it is run and every run afterwards only executes the compiled code. Thus the first run is very slow. For this package it makes sense to always run the function the first time with very few draws and burn-in. Then you can run it with the desired number.
    ```julia
    julia> model_type, set_strct, hyp_strct = makeSetup("Chan2020minn";n_burn=20,n_save=50,p=2)
    julia> out_strct = beavar(model_type, set_strct, hyp_strct, data_strct);
    julia> model_type, set_strct, hyp_strct = makeSetup("Chan2020minn";n_burn=2000,n_save=5000,p=2)
    ```

The structure `out_strct` contains the output of the model which can be used for further analysis such as forecasting or structural analysis using impulse response functions.