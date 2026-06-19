#-----------------------------------------------
# Banbura et al. 2010 LBA functions


#-------------------------------------
# The Den: this is where the beavars live
#-------------------------------------


function beavar(::BGR2010_type, set_struct, hyp_struct, data_struct)
    println("Hello BGR2010")
    YY = values(data_struct.data_tab);
    store_β, store_Σ = BGR2010(YY,set_struct,hyp_struct);
    out_struct = VAROutput_BGR2010(store_β,store_Σ,YY);
    return out_struct
end

function makeHypSetup(::BGR2010_type)
    return hypBGR2010()
end

@doc raw"""
    hypBGR2010()

    Generate a structure with hyperparameters for Banbura, Giannone, and Reichlin (2010) Large Bayesian VARs

# Arguments
    lambda: shrinkage parameter between AR(1) model and maximum likelihood. Default 0.1
    epsi:

# Examples 

- Using default values. Note that the main function will auto-generate this for you. If you don't plan to change any there is rarely need to ever call it.
```julia-repl
julia> hyp = hypBGR2010()
hypBGR2010
  lambda: Float64 0.1
  epsi: Float64 0.001
```

- If a tighter prior (shrinkage towards AR(1)) is desired due to a larger VAR: 
```julia-repl
julia> hyp = hypBGR2010(lambda=0.05)
hypBGR2010
  lambda: Float64 0.05
  epsi: Float64 0.001
```

"""
@with_kw struct hypBGR2010 <: BVARmodelHypSetup
    lambda::Float64     = 0.1; # hyperparameter shrinkage between AR(1) and OLS
    epsi::Float64     = 0.001; # hyperparameter on the constant
end


@doc raw"""
    makeDataSetup(::BGR2010_type,data_tab::TimeArray; var_list =  colnames(data_tab))
    
Generate a dataset strcture for use with the single-frequency models
"""
function makeDataSetup(::BGR2010_type,data_tab::TimeArray; var_list =  colnames(data_tab))
    return dataBVAR_TA(data_tab, var_list)
end




@doc raw"""
# makeDummiesMinn!(sigma::Vector{Float64},delta,lambda,n::Integer,p::Integer,Y_d1))

Fills a matrix $Y_d$ and $X_d$ following eq. (5) in [Banbura, Giannone, Reichling (2010), JAE,
Large Bayesian Autoregressions](https://doi.org/10.1002/jae.1137). 

```math
Y_d =   \begin{bmatrix}
         diag(σ_1*δ_1 \dots σ_N*δ_n)/λ\\
         \mathbf{0}_{n*(p-1) \times n}\\
         diag(σ_1, \dots, σ_n)\\
         \mathbf{0}_{1,n \times n}
         \end{bmatrix}\\
```

```math
X_d = \begin{bmatrix}
        diag(1,\dots,p) \otimes diag(σ_1, \dots σ_n)./ λ \quad \mathbf{0}_{n*p,1}\\
        \mathbf{0}_{n,n*p+1}\\
        \mathbf{0}_{1,n*p}  ε\\
        \end{bmatrix}
```

Instad of creating the matrix every time, the function uses mutation on matrices called `Y_d1` and `X_d1`.

For `Y_d1` it is the the diagonal of the first `(1:n,1:n)` block and in the diagonal of the third block `n+n*(p-1)+1:n+n*(p-1)+n`

For `X_d1` it is the diagonal of the first (n * p x n * p), esesntially `X_d1` is populated along its diagonal and only the constant is added at the end

```lang=julia
function makeDummiesMinn!(sigma::Vector{Float64},delta::Vector{Float64},lambda,n::Integer,p::Integer,epsi,Y_d1,X_d1)
    CI1_Yd1 = CartesianIndex.(1:n,1:n)  # These are the diagonal indices for the top block 
    CI2_Yd1 = CartesianIndex.(n+n*(p-1)+1:n+n*(p-1)+n,1:n)  # These are the diagonal indices for the block in the middle
    CI_Xd1 = CartesianIndex.(1:n*p,1:n*p)
    Y_d1[CI1_Yd1] = sigma.*delta./lambda;
    Y_d1[CI2_Yd1] = sigma;
    X_d1[end,end] = epsi;
    X_d1[CI_Xd1]  = repeat(sigma./lambda,p).*repeat(1:p,inner=n)
    return Y_d1, X_d1;
end
```

"""
function makeDummiesMinn!(sigma::Vector{Float64},delta::Vector{Float64},lambda,n::Integer,p::Integer,epsi,Y_d1,X_d1)
    CI1_Yd1 = CartesianIndex.(1:n,1:n)  # These are the diagonal indices for the top block 
    CI2_Yd1 = CartesianIndex.(n+n*(p-1)+1:n+n*(p-1)+n,1:n)  # These are the diagonal indices for the block in the middle
    CI_Xd1 = CartesianIndex.(1:n*p,1:n*p)
    Y_d1[CI1_Yd1] = sigma.*delta./lambda;
    Y_d1[CI2_Yd1] = sigma;
    X_d1[end,end] = epsi;
    X_d1[CI_Xd1]  = repeat(sigma./lambda,p).*repeat(1:p,inner=n)
    return Y_d1, X_d1;
end

"""
    makeDummiesSumOfCoeff!()

TBW
"""
function makeDummiesSumOfCoeff!(delta,mu,tau,n,p,Yd_2,Xd_2)
    CI1_Yd2 = CartesianIndex.(1:n,1:n)
    Yd_2[CI1_Yd2] = delta.*mu./tau;
    #xd2=[kron(ones(1,p),yd2) zeros(n,1)];
    
    CI1_Xd2 = CartesianIndex.(repeat(1:n,p),1:n*p);
    Xd_2[CI1_Xd2] = repeat(delta.*mu./tau,p)
    return Yd_2, Xd_2
end


"""
    getBeta!(beta::Vector{Float64},Sigma_b::Hermitian)

    Draws from the multivariate normal with mean β and var-covar Σ_b and mutates β

"""
function getBeta!(beta_mean::Vector{Float64},beta::Vector{Float64},Sigma_b::Hermitian)
    d=MvNormal(beta_mean,Sigma_b);
    beta[:]=vec(rand(d,1));
end

function getSigma!(Tstar,P,Sigma_tilde)
    d=InverseWishart(Tstar,P);
    Sigma_tilde[:,:]=rand(d);
    return Sigma_tilde
end



"""
    gibbs_beta_sigma(Tstar,n,p,Ystar,Xstar,n_burn,n_save)

    Implements the gibbs sampler for the Bayesian VAR using dummy variables 
     as in Banbura, Giannone, and Reichlin (2010), Large Bayesian Vectorautoregressions, _Journal of Applied Econometrics_
"""
function gibbs_beta_sigma(Tstar,n,p,Ystar,Xstar,n_burn,n_save)
    k = n*(n*p+1)
    ndraws = n_save+n_burn;
    store_beta=zeros(k,n_save);
    store_sigma=zeros(n*n,n_save);
    beta_mean = vec(Xstar\Ystar)
    beta = similar(beta_mean)
    iXX = inv(Xstar'*Xstar)

    Sigma_tilde = zeros(n,n); Sigma_tilde[CartesianIndex.(1:n,1:n)] = ones(n);

    for ii = 1:ndraws
        Sigma_b = Hermitian(kron(Sigma_tilde,iXX))
            
        getBeta!(beta_mean,beta,Sigma_b)

        res = Ystar-Xstar*reshape(beta,n*p+1,n);
        P   = res'*res

        getSigma!(Tstar,P,Sigma_tilde)

        if ii>n_burn
            store_beta[:,ii-n_burn] = beta;
            store_sigma[:,ii-n_burn] = vec(Sigma_tilde);
        end
    end
    return store_beta, store_sigma
end




@doc raw"""
    BGR2010(YY,set_struct,hyp_struct)

Implements the BVAR with Minnesota prior with a SUR form and common stochastic volatilty (csv) following Chan (2020)

"""
function BGR2010(Z::Matrix{Float64},VARSetup::BVARmodelSetup,hypSetup::BVARmodelHypSetup)
    @unpack lambda, epsi = hypSetup
    @unpack p, n_save, n_burn = VARSetup
    # p = lags;
    deltaP_mat, sigmaP_vec, mu_prior = trainPriors(Z,1)
    delta = deltaP_mat[1,:];

    Y,X,T,n = mlag(Z,p);

    
    tau = 10*lambda;
    Tstar = T+n+n*(p-1)+n+1+n;

    Xstar = zeros(Tstar,n*p+1)
    Ystar = zeros(Tstar,n);                   # holds the data with the prior. This is Y_star in eq. (6) of Banbura, Giannone, and Reichlin (2010), JAE,


    Ystar[1:T,:] = Y;                                   # The top part of Ystar is the data
    Xstar[1:T,:] = X;
    Yd1          = @view Ystar[T+1:T+n+n*(p-1)+n+1,:];  # views into Ystar and defines part of it as Yd1 which will be mutated later in makeDummiesMinn!
    Yd2          = @view Ystar[T+n+n*(p-1)+n+2:Tstar,:]; 

    Xd1          = @view Xstar[T+1:T+n+n*(p-1)+n+1,:];
    Xd2          = @view Xstar[T+n+n*(p-1)+n+1+1:Tstar,:];

    Yd1, Xd1 = makeDummiesMinn!(sigmaP_vec,delta,lambda,n,p,epsi,Yd1,Xd1);
    Yd2, Xd2 = makeDummiesSumOfCoeff!(delta,mu_prior,tau,n,p,Yd2,Xd2);


    store_beta, store_sigma =  gibbs_beta_sigma(Tstar,n,p,Ystar,Xstar,n_burn,n_save);

    # display("done")

    return store_beta, store_sigma;
end


# types for output export
@with_kw struct VAROutput_BGR2010 <: BVARmodelOutput
    store_β::Array{}      # 
    store_Σ::Array{}      # 
    YY::Array{}             #
end



function forecast(VAROutput::VAROutput_BGR2010,VARSetup)
    @unpack store_β, store_Σ, YY = VAROutput
    @unpack n_fcst,p,n_save = VARSetup
    n = size(YY,2);

    Yfor3D    = fill(NaN,(p+n_fcst,n,n_save))
    Yfor3D[1:p,:,:] .= @views YY[end-p+1:end,:];
    
    for i_draw = 1:n_save
        Yfor = @views Yfor3D[:,:,i_draw];
        A_draw = @views reshape(store_β[:,i_draw],n*p+1,n);
        Σ_draw = @views reshape(store_Σ[:,i_draw],n,n);
                
        for i_for = 1:n_fcst
            tclass = @views vec(reverse(Yfor[1+i_for-1:p+i_for-1,:],dims=1)')
            tclass = [1;tclass];
            Yfor[p+i_for,:]=tclass'*A_draw  .+ (cholesky(Σ_draw).U*randn(n,1))';    
        end
    end
    return Yfor3D

end # end function fcastChan2020minn()

