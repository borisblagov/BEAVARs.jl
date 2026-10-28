# BEAVARs.jl Documentation

![Credit Mikelde Ferro: https://www.youtube.com/watch?v=WIYQWK4pkqg&ab_channel=MikedelFerro-Music](BeaverMunchingCabbage.png)


**BEAVARs.jl: Bayesian Econometric Analysis using Vector Autoregressive models**

This is a personal package implementing various Bayesian VARs for economic analysis and forecasting. It is in a very early stage and not ready for use.



# Available models in single frequency

 - [Chan2020minn](@ref): BVAR with classical  Minnesota prior (homoscedastic fixed variance-covariance matrix) as in Chan, J.C.C. (2020), Large Bayesian Vecotrautoregressions, P. Fuleky (Eds), _Macroeconomic Forecasting in the Era of Big Data_, 95-125, Springer, Cham, [https://doi.org/10.1007/978-3-030-31150-6](https://doi.org/10.1007/978-3-030-31150-6), see also [joshuachan.org](https://joshuachan.org) and his [pdf](https://joshuachan.org/papers/large_BVAR.pdf).

 - [Chan2020iniw](@ref): BVAR with classical Minnesota prior and an independent normal inverse Wishart (iniw) prior on the variance-covariance matrix  as in Chan, J.C.C. (2020), Large Bayesian Vecotrautoregressions, P. Fuleky (Eds), _Macroeconomic Forecasting in the Era of Big Data_, 95-125, Springer, Cham, [https://doi.org/10.1007/978-3-030-31150-6](https://doi.org/10.1007/978-3-030-31150-6), see also [joshuachan.org](https://joshuachan.org) and his [pdf](https://joshuachan.org/papers/large_BVAR.pdf).

 - [CCK2016csv](@ref) BVAR with Minnesota prior and common-stochastic volatility (csv), with the csv as in Carriero, Andrea, Todd E. Clark, and Massimiliano Marcellino. 2016. Common Drifting Volatility in Large Bayesian VARs. _Journal of Business & Economic Statistics_ 34 (3): 375–90. https://doi.org/10.1080/07350015.2015.1040116.

 - [Chan2020csv](@ref): BVAR with natural conjugate and common stochastic volatility (csv) as in Chan, J.C.C. (2020), Large Bayesian Vecotrautoregressions, P. Fuleky (Eds), _Macroeconomic Forecasting in the Era of Big Data_, 95-125, Springer, Cham, [https://doi.org/10.1007/978-3-030-31150-6](https://doi.org/10.1007/978-3-030-31150-6), see also [joshuachan.org](https://joshuachan.org) and his [pdf](https://joshuachan.org/papers/large_BVAR.pdf).


 - [BGR2010](@ref): BVAR with dummy observations as in Banbura, M., Giannone, D., and Reichlin, L. (2010), Large Bayesian vecotr auto regressions, _Journal of Applied Econometrics_, Vol 25(1), [doi.org/10.1002/jae.1137](https://doi.org/10.1002/jae.1137).

# Available models with mixed-frequencies

- [CPZ2023](@ref): Mixed-frequency Bayesian VAR based on the precision based sampler from Chan, J.C.C., Poon, A, Zhu, D. (2023) High-dimensional conditionally Gaussian state space models with missing data, _Journal of Econometrics_, Volume 236, Issue 1, September 2023, 105468, [https://doi.org/10.1016/j.jeconom.2023.05.005](https://doi.org/10.1016/j.jeconom.2023.05.005). This version uses an and independet Normal inverse-Wishart prior from [Chan2020iniw](@ref) above and not the csv prior in the CPZ(2023) paper.  

- [Blagov2027](@ref): A mixed-frequency Bayesian VAR based on an upcoming paper of mine. It is based on the precision-based sampler above, incorporates common stochastic volatilty, estimation of hyperparameters, and cross-sectional constraints.



# Usage

Each model is implemented in a separate function, callable using the interface `beavar()`. See the documetnation for details. Note that notation follows the original reference. Consequently variable and parameter names are different across functions (e.g. $\lambda_1$ in one paper can be $c_1$ in another, even if they mean the same thing). 


# First steps
Consider going to the [BEAVARs.jl Documentation](@ref) next
```@contents
Pages = ["priors.md", "methods.md", "irfs.md"]
Depth = 2
```

#  Acknowledgmenets
I would like to thank the following people, who are all **not associated** with this package, but have helped me a lot throughout my Julia journey, mostly through the [Julia discourse](https://discourse.julialang.org/). [Guillaume Dalle](https://github.com/gdalle) went out of his way to help me get my first steps in Github and Julia optimization. [mikmoore](https://discourse.julialang.org/u/mikmoore/summary) and [stevengj](https://discourse.julialang.org/u/stevengj/summary) who have helped me on the julia discourse I want to thank the community, which is really great.

# Notes on the name
- The name BEAVARs is an obvious play of words with a misspelled version of my favourite animal.

- It is also a nod to the [BEAR Toolbox](https://www.ecb.europa.eu/press/research-publications/working-papers/html/bear-toolbox.en.html) - Bayesian  Estimation, Analysis and Regression, which is a powerful Matlab toolbox for estimating various VAR, BVAR, and Panel VAR models. While this is not an attempt to reach the size and scope of BEAR in the Julia ecosystem, there are some clear similarities in the idea of easy estimation of various models.

- The name does not conform to the widely accepted convention of naming Julia packages (capital letter followed by all lowercase) but it [doesn't break any rules either](https://pkgdocs.julialang.org/v1/creating-packages/#Package-naming-rules). It isn't the only package with more than one capita letter, e.g. FFT, CUDA, CSV etc. Yes, it's an acronym, which can always be misleading. CSV may mean for you comma-separated value, but in my world it stands for common stochastic volatility :). And I still have no idea what FFT stands for and never googled it to make a point. Nevertheless, there should be minial confusion, because it's misspelled on purpose - the name Beaver.jl remains open, and if someone wants to use that we can still distinguish the packages BEAVARs.jl and Beavers.jl easily. 

