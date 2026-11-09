# References

## Scientific papers

Banbura, Marta, Giannone, Domenico and Reichlin, Lucrezia, (2010), Large Bayesian vector auto regressions, _Journal of Applied Econometrics_, 25, issue 1, p. 71-92. [https://doi.org/10.1002/jae.1137](https://doi.org/10.1002/jae.1137). 

Boris Blagov, Clara Krause (2026). Does it add up? Reconciling forecasts and Impulse responses for hierarchical macroeconomic data. _Ruhr Economic Papers_, 1224(1). [https://doi.org/10.4419/96973409](https://www.rwi-essen.de/fileadmin/user_upload/RWI/Publikationen/Ruhr_Economic_Papers/REP_26_1224.pdf)

Chan, J.C.C. (2020), Large Bayesian Vecotrautoregressions, P. Fuleky (Eds), _Macroeconomic Forecasting in the Era of Big Data_, 95-125, Springer, Cham, [https://doi.org/10.1007/978-3-030-31150-6](https://doi.org/10.1007/978-3-030-31150-6)

Chan, J.C.C., Poon, A, Zhu, D. (2023) High-dimensional conditionally Gaussian state space models with missing data, _Journal of Econometrics_, Volume 236, Issue 1, September 2023, 105468, [https://doi.org/10.1016/j.jeconom.2023.05.005](https://doi.org/10.1016/j.jeconom.2023.05.005)

Carriero, Andrea, Todd E. Clark, and Massimiliano Marcellino. 2013. Bayesian VARs: Specification Choices and Forecast Accuracy. _Journal of Applied Econometrics_ 30 (1): 46-73. https://doi.org/10.1002/jae.2315.

Carriero, Andrea, Todd E. Clark, and Massimiliano Marcellino. 2016. Common Drifting Volatility in Large Bayesian VARs. _Journal of Business & Economic Statistics_ 34 (3): 375–390. https://doi.org/10.1080/07350015.2015.1040116.


## Function index
```@index
```


## Constructors

```@docs
hypChan2020
```

```@docs
hypBGR2010
```

## Simple time-series methods

```@docs
BEAVARs.ols(Y,X)
```

```@docs
BEAVARs.mlag(Yfull::Matrix{Float64},p::Integer)
```

```@docs
BEAVARs.trainPriors(Z0::Matrix{Float64},p::Int64)
```

```@docs
BEAVARs.percentile_mat(A, p; dims)
```


## Data transformation functions

```@docs
BEAVARs.TAtrans(dataHF_TA,varList_HF,trans_dictA)
```

## Functions for dealing with mixed-frequency data


```@docs
BEAVARs.init_MF_TimeArrays(dataLF_tab::TimeArray{T,N,D,A},dataHF_tab::TimeArray{T,N,D,A},var_list::Array{Symbol,1},prior_RW::Int,n_fcst::Int)  where {T <: AbstractFloat, N, D, A <: AbstractArray{T, N}}
```

```@docs
BEAVARs.make_mf_constraints(set_struct,data_struct)
```

## Functions for out-of-sample evaluation

```@docs
BEAVARs.pseudo_oos(fdataHF_tab,fdataLF_tab,pseudoHF_beg_date,pseudoHF_end_date,ragged_beg_date,ragged_end_date,pubDelay,aggMix,model_type, set_strct, hyp_strct)
```

