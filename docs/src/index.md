# BEAVARs.jl Documentation

![Credit Mikelde Ferro: https://www.youtube.com/watch?v=WIYQWK4pkqg&ab_channel=MikedelFerro-Music](BeaverMunchingCabbage.png)


**BEAVARs.jl: Bayesian Econometric Analysis using Vector Autoregressive models**

This is a personal package implementing various Bayesian VARs for economic analysis and forecasting. 



# Available models

VAR models with a single frequency:

 - [Chan2020minn](@ref): BVAR with classical  Minnesota prior (homoscedastic fixed variance-covariance matrix) as in Chan, J.C.C. (2020), Large Bayesian Vecotrautoregressions, P. Fuleky (Eds), _Macroeconomic Forecasting in the Era of Big Data_, 95-125, Springer, Cham, [https://doi.org/10.1007/978-3-030-31150-6](https://doi.org/10.1007/978-3-030-31150-6), see also [joshuachan.org](https://joshuachan.org) and his [pdf](https://joshuachan.org/papers/large_BVAR.pdf).

 - [Chan2020iniw](@ref): BVAR with Minnesota prior and an independent normal inverse Wishart (iniw) prior on the variance-covariance matrix  as in Chan, J.C.C. (2020), Large Bayesian Vecotrautoregressions, P. Fuleky (Eds), _Macroeconomic Forecasting in the Era of Big Data_, 95-125, Springer, Cham, [https://doi.org/10.1007/978-3-030-31150-6](https://doi.org/10.1007/978-3-030-31150-6), see also [joshuachan.org](https://joshuachan.org) and his [pdf](https://joshuachan.org/papers/large_BVAR.pdf).

 - [Chan2020csv](@ref): BVAR with Minnesota prior and common stochastic volatility (csv) as in Chan, J.C.C. (2020), Large Bayesian Vecotrautoregressions, P. Fuleky (Eds), _Macroeconomic Forecasting in the Era of Big Data_, 95-125, Springer, Cham, [https://doi.org/10.1007/978-3-030-31150-6](https://doi.org/10.1007/978-3-030-31150-6), see also [joshuachan.org](https://joshuachan.org) and his [pdf](https://joshuachan.org/papers/large_BVAR.pdf).



 - [BGR2010](@ref): BVAR with dummy observations as in Banbura, M., Giannone, D., and Reichlin, L. (2010), Large Bayesian vecotr auto regressions, _Journal of Applied Econometrics_, Vol 25(1), [doi.org/10.1002/jae.1137](https://doi.org/10.1002/jae.1137).

Mixed frequency VARs:

- [CPZ2023](@ref): Mixed-frequency Bayesian VAR as in Chan, J.C.C., Poon, A, Zhu, D. (2023) High-dimensional conditionally Gaussian state space models with missing data, _Journal of Econometrics_, Volume 236, Issue 1, September 2023, 105468, [https://doi.org/10.1016/j.jeconom.2023.05.005](https://doi.org/10.1016/j.jeconom.2023.05.005). Important: the mixed-frequency representation in the original paper does not rely on prior assumptions. This version uses the prior from [Chan2020iniw](@ref) above.  


Each model is implemented in a separate function, callable using the interface `beavar()`. See the documetnation for details. Note that notation follows the original reference. Consequently variable and parameter names are different across functions (e.g. $\lambda_1$ in one paper can be $c_1$ in another, even if they mean the same thing). 

Some codes have been translated from Matlab, so there is a lot of room for optimization. 

# First steps
Consider going to the [General introduction](@ref) next
```@contents
Pages = ["BGR2010.md", "Chan2020minn.md", "Chan2020csv.md"]
Depth = 2
```

#  Acknowledgmenets
I would like to thank [Guillaume Dalle](https://github.com/gdalle). He **is not associated** with this package but went out of his way to help me get my first steps in Github and Julia optimization. Also, many users in the [Julia discourse](https://discourse.julialang.org/) helped me often when I was struggling. This community is great.

# Notes on the name
- The name BEAVARs is an obvious play of words with a misspelled version of my favourite animal.

- It is also a nod to the [BEAR Toolbox](https://www.ecb.europa.eu/press/research-publications/working-papers/html/bear-toolbox.en.html) - Bayesian  Estimation, Analysis and Regression, which is a powerful Matlab toolbox for estimating various VAR, BVAR, and Panel VAR models. While this is not an attempt to reach the size and scope of BEAR in the Julia ecosystem, there are some clear similarities in the idea of easy estimation of various models.

- The name does not conform to the widely accepted convention of naming Julia packages (capital letter followed by all lowercase) but it [doesn't break any rules either](https://pkgdocs.julialang.org/v1/creating-packages/#Package-naming-rules). It isn't the only package with more than one capita letter, e.g. FFT, CUDA, CSV etc. Yes, it's an acronym, which can always be misleading. CSV may mean for you comma-separated value, but in my world it stands for common stochastic volatility :). And I still have no idea what FFT stands for and never googled it to make a point. Nevertheless, there should be minial confusion, because it's misspelled on purpose - the name Beaver.jl remains open, and if someone wants to use that we can still distinguish the packages BEAVARs.jl and Beavers.jl easily. 

