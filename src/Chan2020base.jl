#----------------------------------------
# Chan 2020 LBA functions 


# Structure for the hyperparameters for the models
@doc raw"""
    hypChan2020()

Populate a hyperparamater structure for models based on Chan (2020) priors 
        
# Arguments
        c1: hyperparameter on own lags 
        c2: hyperparameter on other lags
        c3: hyperparameter on the constant
        ρ:  ar(1) parameter for the stochastic volatility (Chan2020csv only)
        σ_h2: variance of the log-volatility (Chan2020csv only)
"""
@with_kw struct hypChan2020 <: BVARmodelHypSetup
    c1::Float64     = 0.04; # hyperparameter on own lags
    c2::Float64     = 0.01; # hyperparameter on other lags
    c3::Float64     = 100;  # hyperparameter on the constant
    ρ::Float64      = 0.8;
    σ_h2::Float64   = 0.1;
    v_h0::Float64    = 5.0; 
    S_h0::Float64    = 0.01*(v_h0-1.0); 
    ρ_0::Float64     = 0.9; # prior for ρ_0, the t=0 value of the autoregressive coefficient
    V_ρ::Float64     = 0.04;
    q::Float64       = 0.5;
    nu0::Int         = 3;   # degrees of freedom to add for the inverse wishart distribution of the variance-covariance matrices
end


@doc raw"""
    Xsur = SUR_form(X,n)

    Creates a sparse matrix in a form for Seemingly Unrelated Regression (SUR). 
        
    Returns
        Xsur:    matrix with a structure for SUR

"""
function SUR_form(X,n)
    repX = kron(X,ones(n,1));
    T,k = size(X);
    idi = repeat(1:T*n,inner=k);
    idj=repeat(1:k*n,T);
    Xout = sparse(idi,idj,vec(repX'));

    return Xout
end


@doc raw"""
    Xsur, Xsur_CI, X_CI = SUR_form_dense(X,n)

    Creates matrix in a form for Seemingly Unrelated Regression (SUR). 
        
    Returns
        Xsur:    matrix with a structure for SUR
        Xsur_CI: Cartesian indices of the elements in Xsur that are matched to x
        X_CI:    Cartesian indices of the elements of X that are matched to Xsur

"""
function SUR_form_dense(X,n)
    repX = kron(X,ones(n,1));
    T,k = size(X);
    idi = repeat(1:T*n,inner=k);
    idj=repeat(1:k*n,T);
    Xsur = Matrix(sparse(idi,idj,vec(repX')));
    Xsur_CI = CartesianIndex.(idi,idj);

    idi_X = repeat(1:T,inner=k*n);
    idj_X=repeat(1:k,T*n)
    X_CI = CartesianIndex.(idi_X,idj_X);  # a vector with the indices in X that match the indices in Xsur
    return Xsur, Xsur_CI, X_CI
end


#----------------------------------------
#
#----------------------------------------

@doc raw"""
    blkDiagMat_sp,blockMatInd_vec = makeBlkDiag(Tfn::Int,n::Int,p::Int,blockMat)

Initializes a block-diagonal matrix with p optional blocks below the main diagonal matrix and populates it.

After you have done so, you can later update the entries of the matrix by simply using 

```lang=julia
    blkDiagMat_sp.nzval[:] = @views blockMat[blkDiagMat_sp]
```

It can be used to generate the shape of H_B or Σ from Chan, Poon, and Zhu (2024) which hast the linear indices of the coefficient matrix.

"""
function makeBlkDiag(Tfn::Int,n::Int,p::Int,blockMat)

    type = eltype(blockMat)

    # initialize first the large matrix
    blkDiagMat = zeros(type,Tfn,Tfn)

    # The loop first populates the main block-diagonal (ij = 0)
    # then it populates the first off-diagonal (ij = 1), 
    # but stops before the last bottom right block, which is on the main diagonal and under which there is no entry
    for ij = 0:p
        for ii = 1:div(Tfn,n)-ij
            blkDiagMat[ ij*n + (ii-1)*n + 1 : n + (ii-1)*n +  ij*n, (ii-1)*n + 1 : n + (ii-1)*n] = ones(type,n,n)
        end
    end
    blkDiagMat_sp = sparse(blkDiagMat)

    # Now we will use linear indices to map the values in blockMat to their positions in blkDiagMat
    # - first initialize a matrix with Int as indices cannot be other than Int numbers
    blkDiagMatInt_sp = convert.(Int,blkDiagMat_sp)
    # - take the linear indices of blockMat
    blockMatInd = LinearIndices(blockMat)

    # - populate the diagonal of the Int block-diag matrix with the linear indices of blockMat
    for ij = 0:p
        for ii = 1:div(Tfn,n)-ij
            blkDiagMatInt_sp[ ij*n + (ii-1)*n + 1 : n + (ii-1)*n +  ij*n, (ii-1)*n + 1 : n + (ii-1)*n] = blockMatInd[ :, 1 + (ij-0)*n : n + (ij-0)*n]
        end
    end

    # - copy those linear indices to be used
    blockMatInd_vec = deepcopy(blkDiagMatInt_sp.nzval)

       # we can now update
    blkDiagMat_sp.nzval[:] = blockMat[blockMatInd_vec]

    return blkDiagMat_sp,blockMatInd_vec


    # if you would like to use the BlockBandedMatrices package you can do so. The code for the loops above would be
    # blk = blockMat  
    # lin = LinearIndices(blk)
    # bbm = BlockBandedMatrix(ones(Float64,Tfn,Tfn),n*ones(Int,Tf,),n*ones(Int,Tf,),(p,0))
    # bbm_lin = BlockBandedMatrix(ones(Int,Tfn,Tfn),n*ones(Int,Tf,),n*ones(Int,Tf,),(p,0))
    # for ij = 0:p
    #     for ii in 1:div(Tfn,n)-ij
    #         bbm_lin[Block(ii+ij,ii)]=lin[ 1 + (ij-0)*n : n + (ij-0)*n ,:]
    #     end
    # end
    # blklin_ind = deepcopy(bbm_lin.data)
    # blkDiagMat_sp=sparse(bbm)
    # blkDiagMat_sp.nzval[:] = @view blk[blklin_ind];

end



@doc raw"""
# blkDiagMat_sp,blockMatInd_vec = makeBlkDiag_ns(Tfn::Int,n::Int,p::Int,blockMat)

Initializes a non-sparse block-diagonal matrix with p optional blocks below the main diagonal matrix and populates it.

After you have done so, you can later update the entries of the matrix by simply using 

```lang=julia
    blkDiagMat[blkDiagMat_CI] = blockMat[blockMatInd_vec]
```

It can be used to generate the shape of H_B or Σ from Chan, Poon, and Zhu (2024) which hast the linear indices of the coefficient matrix.

"""
function makeBlkDiag_ns(Tfn::Int,n::Int,p::Int,blockMat)

    type = eltype(blockMat)

    # initialize first the large matrix
    blkDiagMat = zeros(type,Tfn,Tfn)

    # The loop first populates the main block-diagonal (ij = 0)
    # then it populates the first off-diagonal (ij = 1), 
    # but stops before the last bottom right block, which is on the main diagonal and under which there is no entry
    for ij = 0:p
        for ii = 1:div(Tfn,n)-ij
            blkDiagMat[ ij*n + (ii-1)*n + 1 : n + (ii-1)*n +  ij*n, (ii-1)*n + 1 : n + (ii-1)*n] = ones(type,n,n)
        end
    end

    # Now we will use linear indices to map the values in blockMat to their positions in blkDiagMat
    # - first initialize a matrix with Int as indices cannot be other than Int numbers
    blkDiagMatInt = convert.(Int,blkDiagMat)
    # - take the linear indices of blockMat
    blockMatInd = LinearIndices(blockMat)

    # - populate the diagonal of the Int block-diag matrix with the linear indices of blockMat
    for ij = 0:p
        for ii = 1:div(Tfn,n)-ij
            blkDiagMatInt[ ij*n + (ii-1)*n + 1 : n + (ii-1)*n +  ij*n, (ii-1)*n + 1 : n + (ii-1)*n] = blockMatInd[ :, 1 + (ij-0)*n : n + (ij-0)*n]
        end
    end

    blkDiagMat_CI = findall(blkDiagMat.==1.0);  # Carteisan indices with data
    # - copy those linear indices to be used
    blockMatInd_vec = blkDiagMatInt[blkDiagMat_CI];

       # we can now update
    blkDiagMat[blkDiagMat_CI] = blockMat[blockMatInd_vec]

    return blkDiagMat, blkDiagMat_CI, blockMatInd_vec

end



"""
    
"""
function Chan2020_drawβ(Σ_invsp,Xsur_den,XtΣ_inv_den,XtΣ_inv_X,Vβminn_inv,βminn,K_β,Y,n,k)
        mul!(XtΣ_inv_den,Xsur_den',Σ_invsp);            #  X'*( I(T) ⊗ Σ^{-1} )
        mul!(XtΣ_inv_X,XtΣ_inv_den,Xsur_den);           #  X'*( I(T) ⊗ Σ^{-1} )*X
        K_β[:,:] .= Vβminn_inv .+ XtΣ_inv_X;            #  K_β = V^{-1} + X'*( I(T) ⊗ Σ^{-1} )*X
        prior_mean = Vβminn_inv*βminn;                  #  V^-1 * βminn 
        mul!(prior_mean,XtΣ_inv_den, vec(Y'),1.0,1.0);  # (V^-1_Minn * beta_Minn) + X' ( I(T) ⊗ Σ-1 ) y
        cholK_β = cholesky(Hermitian(K_β));             # C is lower triangular, C' is upper triangular
        beta_hat = ldiv!(cholK_β.U,ldiv!(cholK_β.L,prior_mean));    # C'\(C*(V^-1_Minn * beta_Minn + X' ( I(T) ⊗ Σ-1 ) y)
        beta = beta_hat + ldiv!(cholK_β.U,randn(k*n,)); # draw for β
        return beta
end

"""
"""
function Chan2020_drawΣt(Y,Xsur_den,beta,n,T,S_0,nu0)
    U = reshape(vec(Y') - Xsur_den*beta,n,T);       
    Σt = rand(InverseWishart(nu0+n+T,S_0+U*U'));    # draw for Σ
    Σt_inv = Σt\I;
    return Σt, Σt_inv
end







@doc raw"""
    (idx_kappa1,idx_kappa2, C) = prior_Minn(n,p,sigmaP,hypSetup)

Impements a Minnesota prior with scaling for the off-diagonal elements as in Chan, J.C.C. (2020). Large Bayesian Vector Autoregressions. In: P. Fuleky (Eds),
Macroeconomic Forecasting in the Era of Big Data, 95-125, Springer, Cham. If you remove the hyperparameters ``c_1``, ``c_2``, ``c_3`` (or set them equal to 1),
you would end up with the getC function from Chan, J.C.C. (2021). Minnesota-Type Adaptive Hierarchical Priors for Large Bayesian VARs, International Journal of Forecasting, 

Outputs: 
- C is the variance of the Minnesota prior defined as
```math
C_{n^2p+n \times 1} = 
\begin{cases}
        \dfrac{c_1}{l^2}, & \text{for the coefficient on the $l$-th lag of variable $i$}\\
        \dfrac{c_2 s^2_i}{l^2 s^2_j}, & \text{for the coefficient on the $l$-th lag of variable $j$, $j\neq i$}\\
        c_3, & \text{for the intercept}\\
\end{cases}
```
It is commonly known as `V_Minn` or `V_theta` in other codes.

The hyperparameters have default values of ``c_1 = 0.04``; ``c_2 = 0.01``; ``c_3 = 100``. In the Chan (2021) IJF paper they
are referred to as ``\kappa_1``, ``\kappa_2``, and ``\kappa_4`` (``\kappa_3`` is used there for prior on contemporaneous relationships)

"""
function prior_Minn(n::Integer,p::Integer,sigmaP_vec::Vector{Float64},hypSetup)    
    @unpack c1,c2,c3 = hypSetup
    C = zeros(n^2*p+n,);
    # beta_Minn = zeros(n^2*p+n);
    np1 = n*p+1 # number of parameters per equation
    idx_kappa1 =  Vector{Int}()
    idx_kappa2 =  Vector{Int}()
    idx_count = 1    
    Ci = zeros(np1,1)     # vector for equation i

    for ii = 1:n
      for j = 1:n*p+1       # for j=1:n*p+1 
        l = ceil((j-1)/n)       # Here we need a float, as afterwards we will divide by l 
        idx = mod(j-1,n);       # this will count if its own lag, non-own lag, or constant
        if idx==0
            idx = n;
        end
        if j == 1
            Ci[j] = c3;
        elseif idx == ii
            Ci[j] = c1/l^2;
            push!(idx_kappa1,idx_count)
        else
            Ci[j] = c2*sigmaP_vec[ii]/(l^2*sigmaP_vec[idx]);
            push!(idx_kappa2,idx_count)
        end
        idx_count += 1
    end

    C[(ii-1)*np1+1:ii*np1] = Ci

    end
    return idx_kappa1,idx_kappa2, C

end


@doc raw"""
    (idx_kappa1,idx_kappa2, C, beta_Minn) = prior_NonConj(n,p,sigmaP,hyp)

Imlpements a Minnesota prior for a non-conjugate case as in Chan, J.C.C. (2020). Large Bayesian Vector Autoregressions. In: P. Fuleky (Eds),
Macroeconomic Forecasting in the Era of Big Data, 95-125, Springer, Cham.

Outputs: 
- C is the variance of the Minnesota prior defined as
```math
C_{n*p+1 \times 1} = 
\begin{cases}
        \dfrac{c_1}{l^2 s^2_i}, & \text{for the coefficient on the $l$-th lag of variable $i$}\\
        c_3, & \text{for the intercept}\\
\end{cases}
```

Note that C is now (n*p+1 x 1) and not n^2*p+n as [`prior_Minn(x)`](@ref)

"""
function prior_NonConj(n::Integer,p::Integer,sigmaP_vec::Vector{Float64},hypSetup)    
    @unpack c1,c3 = hypSetup;
    C = zeros(n^2*p+n,);
    beta_Minn = zeros(n^2*p+n);
    np1 = n*p+1 # number of parameters per equation
    idx_kappa1 =  Vector{Int}()
    idx_kappa2 =  Vector{Int}()
    idx_count = 1    
    C = zeros(np1,)     # vector for equation i


    for j = 1:n*p+1       # for j=1:n*p+1 
    l = ceil((j-1)/n)       # Here we need a float, as afterwards we will divide by l 
    idx = mod(j-1,n);       # this will count if its own lag, non-own lag, or constant
    if idx==0
        idx = n;
    end
    if j == 1
        C[j] = c3;
    else
        C[j] = c1/(l^2*sigmaP_vec[idx]);
        push!(idx_kappa1,idx_count)
    end

    end
    return idx_kappa1,idx_kappa2, C, beta_Minn

end



@doc raw"""
    BEAVARs.initMinn(YY, p)

    Initializes necessary matrices for a Minnesota prior following Chan (2020).

# Arguments
- `YY::Matrix{Float64}`: A matrix of time-series data where rows represent time periods and columns represent variables.
- `p::Int`: The number of lags to include in the model.

# Returns
- `Y::Matrix{Float64}`:         Dependent variables matrix after lag transformation.
- `X::Matrix{Float64}`:         Independent variables matrix after lag transformation.
- `T::Int`:                     Number of time periods after accounting for lags.
- `n::Int`:                     The number of variables in the system.
- `sigmaP::Vector{Float64}`:    A vector of variances for each variable, initialized using univariate AR(4) regressions.
- `S_0::Diagonal{Float64}`:     The diagonal matrix of `sigmaP` prior variances for the Minnesota prior.
- `Σt_inv::Matrix{Float64}`:    The inverse of the covariance matrix `Σ` initialized as `S_0 \ I`.
- `Vβ_inv::Matrix{Float64}`:    The prior precision matrix for the regression coefficients, initialized as an identity matrix scaled by 1.0.
- `Vβ_inv_vecView`:             A view of the diagonal elements of `Vβ_inv`, used for efficient updates.
- `Σ_invsp::SparseMatrixCSC{Float64}`: A sparse matrix representing `I(T) ⊗ Σ^-1`, where `⊗` is the Kronecker product.
- `Σt_LI::Vector{Int}`: Linear indices for updating the sparse matrix `Σ_invsp`.
- `XtΣ_inv_den::Matrix{Float64}`:   A dense matrix for storing `X' * (I(T) ⊗ Σ^-1)`.
- `XtΣ_inv_X::Matrix{Float64}`:     A dense matrix for storing `X' * (I(T) ⊗ Σ^-1) * X`.
- `Xsur_den::Matrix{Float64}`:      A dense matrix in Seemingly Unrelated Regression (SUR) form.
- `Xsur_CI::Vector{CartesianIndex}`: Cartesian indices mapping elements of `Xsur_den` to the original `X` matrix.
- `X_CI::Vector{CartesianIndex}`:   Cartesian indices mapping elements of `X` to `Xsur_den`.
- `k::Int`:                         The total number of regression coefficients, calculated as `n * p + intercept`.
- `K_β::Matrix{Float64}`:           The variance-covariance matrix of the regression coefficients.
- `beta::Vector{Float64}`:          A vector of regression coefficients, initialized to zeros.
- `intercept::Int`:                 A flag indicating where the intercept is.
"""
function initMinn(YY,p)
    Y, X, T, n, intercept       = mlagL(YY,p);
    k                           = n*p+intercept
    sigmaP                      = ar4!(YY,zeros(n,));                       # do OLS to initialize priors
    S_0                         = Diagonal(sigmaP);              
    Σt_inv                      = S_0\I;                                    # initialize Σ^-1              
    Vβ_inv                      = 1.0*Matrix(I,n*k,n*k);                    # prior matrix
    Vβ_inv_vecView              = @view(Vβ_inv[diagind(Vβ_inv)]);           # vector, element view, used to update the diagonal    
    Σ_invsp, Σt_LI              = BEAVARs.makeBlkDiag(T*n,n,0,Σt_inv);      # I(T) ⊗ Σ-1 and its indices for update
    XtΣ_inv_den                 = zeros(k*n,T*n);                           # will be X' ( I(T) ⊗ Σ-1 )   from page 6 in Chan 2020 LBA
    XtΣ_inv_X                   = zeros(n*k,n*k);                           # will be X' ( I(T) ⊗ Σ-1 ) X from page 6 in Chan 2020 LBA    
    Xsur_den, Xsur_CI, X_CI     = BEAVARs.SUR_form_dense(X,n);              # prepares the SUR form and the indices of the parameters for updating
    K_β                         = zeros(n*k,n*k);                           # Variance covariance matrix of the parameters
    beta                        = zeros(n*k,);                              # the parameters in a vector
    
    return Y, X, T, n, sigmaP, S_0, Σt_inv, Vβ_inv, Vβ_inv_vecView, Σ_invsp, Σt_LI, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, K_β, beta, intercept
end