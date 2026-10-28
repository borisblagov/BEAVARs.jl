# Priors on $B$ and $\Sigma$

Estimating Bayesian VARs boils down to the choice of priors regarding the parameters for the chosen VAR model. The package contains various specifications of a VAR with their respective prior choices. The following section outlines them.



## Minnesota prior

This is the classical Minnesota prior with a known variance-covariance matrix for the following VAR:
```math
\begin{equation}
\mathbf{y}_t = \mathbf{b} + \mathbf{B}_1 \mathbf{y}_{t-1} + \mathbf{B}_2 \mathbf{y}_{t-2} + \dots + \mathbf{B}_p \mathbf{y}_{t-p} +  \varepsilon_t \qquad \varepsilon \sim N(\mathbf{0}, \hat{\Sigma}).
\end{equation}
```

We have to make inference only on the $\mathbf{B}$ matrices of the model, since the variance covariance matrix is considered fixed at $\hat{\Sigma}$.

Consider a normal prior for the vector of coefficients $\boldsymbol{\beta}$.

```math
\begin{equation}
    \boldsymbol{\beta} \sim N(\boldsymbol{\beta_{Minn}},\mathbf{V_{\boldsymbol{\beta}}})
\end{equation}
```

The Minnesota variance covariance matrix is defined as
```math
\mathbf{V_{\boldsymbol{\beta}}}_{i,jj} = 
\begin{cases}
  \frac{{c}_1}{l^2}, & \text{for coefficients on own lag } l \text{ for } l = 1, \dots, p \\[2ex]
  \frac{{c}_2 \sigma_{ii}}{l^2 \sigma_{jj}}, & \text{for coefficients on lag } l \text{ of variable } j \neq i \\
  & \text{for } l = 1, \dots, p \\[2ex]
  {c}_3 \sigma_{ii}, & \text{for coefficients on exogenous variables}
\end{cases}
```

The code initializes this prior using the function `init_priorMinn` and does so in two steps. 

```@docs
    BEAVARs.init_priorMinn(n,p,sigmaP_vec,data_trans,hyp_struct)
```

It first initializes a matrix $\mathbf{C_{\boldsymbol{\beta}}}_{i,jj}$  that does not incorporate the hyperpameters $c_1, c_2,$ and $c_3$. 

```math
\mathbf{C_{\boldsymbol{\beta}}}_{i,jj} = 
\begin{cases}
  \frac{1}{l^2}, & \text{for coefficients on own lag } l \text{ for } l = 1, \dots, p \\[2ex]
  \frac{1 \sigma_{ii}}{l^2 \sigma_{jj}}, & \text{for coefficients on lag } l \text{ of variable } j \neq i \\
  & \text{for } l = 1, \dots, p \\[2ex]
  \sigma_{ii}, & \text{for coefficients on exogenous variables}
\end{cases}
```

Then, using the vector form multiplies the relevant entries. The matrix can be updated later in place using `BEAVERs.update_priorMinn!` function.

```@docs
    BEAVARs.update_priorMinn!(prior_struct::BEAVARs.BVARpriorMinn,hyp_struct::BEAVARs.BVARmodelHypSetup)
```




The conditional distribution is 

```math
(\boldsymbol{\beta} | \mathbf{y}) \sim \mathcal{N}(\widehat{\boldsymbol{\beta}}, \mathbf{K}_{\beta}^{-1}),
```

where

```math
\mathbf{K}_{\beta} = \mathbf{V}_{\text{Minn}}^{-1} + \mathbf{X}' (\mathbf{I}_T \otimes \widehat{\boldsymbol{\Sigma}}^{-1}) \mathbf{X}, \quad \widehat{\boldsymbol{\beta}} = \mathbf{K}_{\beta}^{-1} \left( \mathbf{V}_{\text{Minn}}^{-1} \boldsymbol{\beta}_{\text{Minn}} + \mathbf{X}' (\mathbf{I}_T \otimes \widehat{\boldsymbol{\Sigma}}^{-1}) \mathbf{y} \right),
```


## Independent-Normal inverse-Wishart 

A VAR model with a Minnesota prior with and an unkown variance-covariance matrix, with both blocks considered to be independent of one another.
```math
\begin{equation}
\mathbf{y}_t = \mathbf{b} + \mathbf{B}_1 \mathbf{y}_{t-1} + \mathbf{B}_2 \mathbf{y}_{t-2} + \dots + \mathbf{B}_p \mathbf{y}_{t-p} +  \varepsilon_t \qquad \varepsilon \sim N(\mathbf{0}, \Sigma).
\end{equation}
```

We draw the $\mathbf{B}$ matrices using the Minnesota prior above and use an inverse-Wishart to draw the variance-covariance matrix $\Sigma$


## Natural conjugate prior 

The natural conjugate prior is drawn again using the normal distribution for the $\mathbf{B}$ and an inverse-Wishart for the variance-covariance matrix $\Sigma$. The main benefit is that it is very fast, because there is no MCMC. The drawback is that you cannot implement the features of the Minnesota prior of shrinking different variables differently and scaling the variables differently. This means that your data is best to be standardized.


## Common stochastic volatility