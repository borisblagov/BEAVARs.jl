"""
    BVARpriorMinn(β_prior, Cβ_vec, Vβ_vec, Vβ_inv, Vβ_inv_vecView, Vinvβ_prior, idx_kappa1, idx_kappa2, idx_kappa3)

    Returns a structure containing the main elements of a Minnesota prior for a Bayesian VAR model. The structure includes the prior mean, variance, and indices for the location of the different types of coefficients (own lags, other lags, constants).
"""
struct BVARpriorMinn{T <: AbstractFloat}
    β_prior::Array{T,1}        # vector of either zeros or ones depending on the data transformation
    Cβ_vec::Array{T,1}         # collects the diagonal elements of V but without the hyperparameters
    Vβ_vec::Array{T,1}         # under classical minnesota prior variance is diagonal
    Vβ_inv::Array{T,2}         # inverse of the matrix with a diagonal Vβ_vec
    Vβ_inv_vecView::SubArray{}  # view of the diagonal of Vβ_inv for fast updating
    Vinvβ_prior::Array{T,1}        # vector of V^-1 * β_prior
    idx_kappa1::Array{Int,1} # indices of parameters associated with own lag
    idx_kappa2::Array{Int,1} # indices of parameters associated with other variables and their lags
    idx_kappa3::Array{Int,1} # indices of parameters associated with the constant term
end


# Julia structure for the SUR form of an X matrix
@doc raw"""
    Xsur = XSurFormMatrix(X,n)

    Creates a matrix in SUR form from a lagged matrix of the VAR with constants. The resulting matrix has dimensions (T*n, k*n) where T is the number of time periods, n is the number of variables, and k is the number of predictors (including constants). The SUR form is used in the context of Seemingly Unrelated Regressions (SUR) to facilitate efficient computations.

    Let $X$ be $(T \times k)$ and there be $n$ variables, then for $n=3$ 
    $$
        Xsur_{n=3} = \begin{bmatrix}
                    X[1,:] & \mathbf{0}_{1,k} &  \mathbf{0}_{1,k}\\
                    \mathbf{0}_{1,k} & X[1,:] &\mathbf{0}_{1,k}\\
                    \mathbf{0}_{1,k} & \mathbf{0}_{1,k}  & X[1,:]  \\
                    X[2,:] & \mathbf{0}_{1,k} &  \mathbf{0}_{1,k}\\
                    \mathbf{0}_{1,k} & X[2,:] &\mathbf{0}_{1,k}\\
                    \mathbf{0}_{1,k} & \mathbf{0}_{1,k}  & X[2,:]  \\
                    \vdots & \vdots & \vdots \\
                    \mathbf{0}_{1,k} & \mathbf{0}_{1,k}  & X[T,:]  \\
                \end{bmatrix}
    $$

"""
struct XSurFormMatrix{T} <: AbstractArray{T,2}
    # 
    X::Array{T,2}   # lagged matrix of the VAR with constants
    n::Int          # number of variables
end

# implement AbstractArray interface for XSurFormMatrix
function Base.getindex(Xsur::XSurFormMatrix, i::Int, j::Int)
    @boundscheck checkbounds(Xsur, i, j)
    k = size(Xsur.X, 2)

    t = div(i - 1, Xsur.n) + 1
    row_id = mod(i - 1, Xsur.n) + 1

    col_block_id = div(j - 1, k) + 1
    predictor = mod(j - 1, k) + 1
    if  row_id == col_block_id
        return Xsur.X[t, predictor]
    else
        return zero(eltype(Xsur))
    end
end

function Base.setindex!(Xsur::XSurFormMatrix, value, i::Int, j::Int)
    @boundscheck checkbounds(Xsur, i, j)

    n = Xsur.n
    k = size(Xsur.X, 2)

    t = div(i - 1, n) + 1
    equation = mod(i - 1, n) + 1

    column_equation = div(j - 1, k) + 1
    predictor = mod(j - 1, k) + 1

    if equation == column_equation
        Xsur.X[t, predictor] = value
    elseif !iszero(value)
        throw(ArgumentError("cannot store a nonzero value outside the SUR structure"))
    end

    return value
end

Base.size(Xsur::XSurFormMatrix) = (size(Xsur.X,1)*Xsur.n, size(Xsur.X,2).*Xsur.n)
     


function _scale_output!(C, β)
    if iszero(β)
        fill!(C, zero(eltype(C)))
    else
        C .*= β
    end
    return C
end

# C = A * B, where A is in SUR form
@doc raw"""
    C = mul!(C, A::XSurFormMatrix, B::AbstractMatrix; α=1.0, β=0.0)

    Multiplies a matrix in SUR form with a general dense matrix. The resulting matrix C has dimensions (T*n, size(B, 2)), where T is the number of time periods, n is the number of variables, and size(B, 2) is the number of columns in B.

    # Arguments
    - `C`: Output matrix to store the result.
    - `A`: An `XSurFormMatrix` representing the lagged matrix of the VAR with constants.
    - `B`: A general dense matrix to be multiplied with A.
    - `α`: Scalar multiplier for the product (default is 1.0).
    - `β`: Scalar multiplier for the existing values in C (default is 0.0).

    # Returns
    - `C`: The resulting matrix after multiplication.

    # This function was written by an LLM. There is a test in the test suite that checks the correctness against my own implementation.
"""
function LinearAlgebra.mul!(
    C::AbstractMatrix, A::XSurFormMatrix, B::AbstractMatrix,
    α::Number=one(eltype(C)), β::Number=zero(eltype(C)),
)
    T, k = size(A.X)
    n = A.n
    size(B, 1) == k*n && size(C) == (T*n, size(B, 2)) ||
        throw(DimensionMismatch("incompatible dimensions for SURDesign * B"))

    # Each (time, equation) contributes to its own output row.
    for t in 1:T, e in 1:n
        row = (t - 1)*n + e
        cols = (e - 1)*k + 1:e*k
        mul!(
            view(C, row:row, :),
            transpose(view(A.X, t, :)),
            view(B, cols, :),
            α, β,
        )
    end
    return C
end

# C = L * A, where A is in SUR form
@doc raw"""
  # This function was written by an LLM. There is a test in the test suite that checks the correctness against my own implementation.
"""
function LinearAlgebra.mul!(
    C::AbstractMatrix, L::AbstractMatrix, A::XSurFormMatrix,
    α::Number=one(eltype(C)), β::Number=zero(eltype(C)),
)
    T, k = size(A.X)
    n = A.n
    size(L, 2) == T*n && size(C) == (size(L, 1), k*n) ||
        throw(DimensionMismatch("incompatible dimensions for L * XSurFormMatrix"))

    # Each output column has contributions only from one equation.
    for e in 1:n, j in 1:k
        col = (e - 1)*k + j
        rows = e:n:T*n
        mul!(
            view(C, :, col),
            view(L, :, rows),
            view(A.X, :, j),
            α, β,
        )
    end
    return C
end

@doc raw"""
  # This function was written by an LLM. There is a test in the test suite that checks the correctness against my own implementation.
"""
# C = A' * B, for a general dense B
function LinearAlgebra.mul!(
    C::AbstractMatrix, At::Adjoint{<:Any,<:XSurFormMatrix}, B::AbstractMatrix,
    α::Number=one(eltype(C)), β::Number=zero(eltype(C)),
)
    A = parent(At)
    T, k = size(A.X)
    n = A.n
    size(B, 1) == T*n && size(C) == (k*n, size(B, 2)) ||
        throw(DimensionMismatch("incompatible dimensions for XSurFormMatrix' * B"))

    for e in 1:n
        cols = (e - 1)*k + 1:e*k
        rows = e:n:T*n
        mul!(
            view(C, cols, :),
            adjoint(A.X),
            view(B, rows, :),
            α, β,
        )
    end
    return C
end

# Sparse specialization: skips zero entries in B as well as the zeros in A.
@doc raw"""
  # This function was written by an LLM. There is a test in the test suite that checks the correctness against my own implementation.
"""
function LinearAlgebra.mul!(
    C::AbstractMatrix, At::Adjoint{<:Any,<:XSurFormMatrix},
    B::SparseMatrixCSC,
    α::Number=one(eltype(C)), β::Number=zero(eltype(C)),
)
    A = parent(At)
    T, k = size(A.X)
    n = A.n
    size(B, 1) == T*n && size(C) == (k*n, size(B, 2)) ||
        throw(DimensionMismatch("incompatible dimensions for XSurFormMatrix' * B"))

    _scale_output!(C, β)
    rows = rowvals(B)
    vals = nonzeros(B)

    for col in axes(B, 2), p in nzrange(B, col)
        r = rows[p]
        t = div(r - 1, n) + 1
        e = mod(r - 1, n) + 1
        outrows = (e - 1)*k + 1:e*k
        @inbounds for j in 1:k
            C[outrows[j], col] += α * conj(A.X[t, j]) * vals[p]
        end
    end
    return C
end


@doc raw"""
    update_Xsur!(Xsur::XSurFormMatrix, Xnew::AbstractMatrix)

    Updates the X matrix of an XSurFormMatrix with a new matrix in place to avoid allocations.
    Does not check for the dimensions (although this is implemented but commented out)

    # Arguments
    - `Xsur`: The XSurFormMatrix to update.
    - `Xnew`: The new matrix to replace the existing X matrix.

    # Returns
    - `Xsur`: The updated XSurFormMatrix.
"""
function update_Xsur!(Xsur::XSurFormMatrix, Xnew::AbstractMatrix)
    # size(Xsur.X) == size(Xnew) ||
    #     throw(DimensionMismatch("new X must have the same size as the existing X"))
    #
    copyto!(Xsur.X, Xnew)
    return Xsur
end

## --------------------------------------
# FUNCTIONS

@doc raw"""
    prior_struct = init_priorMinn(n,p,sigmaP_vec,data_trans,hyp_struct)

    Initializes the Minnesota prior structure for a Bayesian VAR model. The function computes the prior mean and variance for the coefficients based on the number of variables, lags, and hyperparameters provided.

    # Arguments
    - `n`: Number of endogenous variables in the VAR.
    - `p`: Number of lags in the VAR.
    - `sigmaP_vec`: Vector of standard deviations for each variable, used to scale the prior variances.
    - `data_trans`: Indicator for data transformation (e.g., levels or growth rates).
    - `hyp_struct`: Structure containing hyperparameters for the Minnesota prior.

    # Returns
    - `prior_struct`: A `BVARpriorMinn` structure containing the prior mean, variance, and indices for different types of coefficients (own lags, other lags, constants).
"""
function init_priorMinn(n,p,sigmaP_vec,data_trans,hyp_struct)
    Cβ_vec = zeros(n^2*p+n,);       
    Vβ_vec = zeros(n^2*p+n,);                       # C diag but without the hyperparameters
    βMinn = zeros(n^2*p+n);                         # initialize the prior mean for growth rates, e.g. center at 0
    np1 = n*p+1                                     # number of parameters per equation
    idx_kappa1 =  Vector{Int}()
    idx_kappa2 =  Vector{Int}()
    idx_kappa3 =  Vector{Int}()
    idx_count = 1    
    Vi = zeros(np1,1)     # vector for equation i

    for ii = 1:n
      for j = 1:n*p+1       # for j=1:n*p+1 
        l = ceil((j-1)/n)           # Here we need a float, as afterwards we will divide by l 
        idx = mod(j-1,n);           # this will count if its own lag, non-own lag, or constant
        if idx==0
            idx = n;
        end
        if j == 1                   # Constant is the first element in each equation
            Vi[j] = 1;
            push!(idx_kappa3,idx_count)
        elseif idx == ii            # These are the own lags of the variable
            if l == 1 && data_trans == 1              # If we have levels instead of growth rates we add a "1" for the first lag
                βMinn[idx_count,] = 1;
            end
            Vi[j] = 1/l^2;
            push!(idx_kappa1,idx_count)
        else
            Vi[j] = sigmaP_vec[ii]/(l^2*sigmaP_vec[idx]);
            push!(idx_kappa2,idx_count)
        end
        idx_count += 1
    end

    Cβ_vec[(ii-1)*np1+1:ii*np1] = Vi

    end
    
    Vβ_vec[idx_kappa1] .= Cβ_vec[idx_kappa1] .* hyp_struct.c1;  # own lags
    Vβ_vec[idx_kappa2] .= Cβ_vec[idx_kappa2] .* hyp_struct.c2;  # other lags
    Vβ_vec[idx_kappa3] .= Cβ_vec[idx_kappa3] .* hyp_struct.c3;  # constant
    Vβ_inv = diagm(1.0./Vβ_vec);
    Vβ_inv_vecView = @view(Vβ_inv[diagind(Vβ_inv)]);
    Vinvβ_prior = Vβ_inv*βMinn;
    priorMinn_struct = BVARpriorMinn(βMinn, Cβ_vec, Vβ_vec, Vβ_inv, Vβ_inv_vecView, Vinvβ_prior, idx_kappa1, idx_kappa2, idx_kappa3)

    return priorMinn_struct

end

"""
    update_priorMinn_broadcast!(prior_struct::BVARpriorMinn,hyp_struct::BEAVARs.BVARmodelHypSetup)

    Updates the variance-covariance matrix of the Minnesota prior coefficients with new hyperparameters.
    The function is easy on the eyes but slower than a hot-loop, therefore not used

    See
"""
function update_priorMinn_broadcast!(prior_struct::BVARpriorMinn,hyp_struct::BEAVARs.BVARmodelHypSetup)
    # hot loops are faster than broadcasting
    @views prior_struct.Vβ_vec[prior_struct.idx_kappa1] .= prior_struct.Cβ_vec[prior_struct.idx_kappa1] .* hyp_struct.c1;  # own lags
    @views prior_struct.Vβ_vec[prior_struct.idx_kappa2] .= prior_struct.Cβ_vec[prior_struct.idx_kappa2] .* hyp_struct.c2;  # other lags
    @views prior_struct.Vβ_vec[prior_struct.idx_kappa3] .= prior_struct.Cβ_vec[prior_struct.idx_kappa3] .* hyp_struct.c3;  # constants
end


"""
    update_priorMinn!(prior_struct,hyp_struct)
"""
function update_priorMinn!(prior_struct::BEAVARs.BVARpriorMinn,hyp_struct::BEAVARs.BVARmodelHypSetup)
    @unpack Cβ_vec, Vβ_vec, idx_kappa1, idx_kappa2, idx_kappa3 = prior_struct    
    for ii in eachindex(idx_kappa1)
        Vβ_vec[idx_kappa1[ii]] = Cβ_vec[idx_kappa1[ii]] * hyp_struct.c1
    end
    for ii in eachindex(idx_kappa2)
        Vβ_vec[idx_kappa2[ii]] = Cβ_vec[idx_kappa2[ii]] * hyp_struct.c2
    end
    for ii in eachindex(idx_kappa3)
        Vβ_vec[idx_kappa3[ii]] = Cβ_vec[idx_kappa3[ii]] * hyp_struct.c3
    end
end


function calc_beta_hat!(prior_struct,β_hat_struct,X,Y)
    @unpack Xsur, XtΣ_inv_den, XtΣ_inv_X, Σ_inv_sp, K_β, β_hat = β_hat_struct;
    @unpack Vβ_vec, Vβ_inv, Vβ_inv_vecView, β_prior,Vinvβ_prior = prior_struct;
    Vβ_inv_vecView[:] .= 1.0./Vβ_vec;                   # update the diagonal of Vβ_inv
    update_Xsur!(Xsur, X);                              # update Xs                   
    mul!(Vinvβ_prior, Vβ_inv, β_prior)                  #  update V^-1 * β_Minn 

    mul!(XtΣ_inv_den,Xsur',Σ_inv_sp);                   #  X'*( I(T) ⊗ Σ^{-1} )
    mul!(XtΣ_inv_X,XtΣ_inv_den,Xsur);
    K_β .= Vβ_inv .+ XtΣ_inv_X;                         #  K_β = V^{-1} + X'*( I(T) ⊗ Σ^{-1} )*X
    mul!(Vinvβ_prior,XtΣ_inv_den, vec(Y'),1.0,1.0);     # (V^-1_Minn * beta_Minn) + X' ( I(T) ⊗ Σ-1 ) y

    cholK_β = cholesky(Hermitian(K_β));                # Cholesky factor
    β_hat[:] = ldiv!(cholK_β.U,ldiv!(cholK_β.U',Vinvβ_prior));    # C'\(C*(V^-1_Minn * beta_Minn + X' ( I(T) ⊗ Σ-1 ) y)
    return cholK_β
end



struct make_βdraw_struct{T <: AbstractFloat}
    β_draw::Array{T,1}        # 
    β_hat::Array{T,1}        # 
    Xsur::XSurFormMatrix{T}         # 
    XtΣ_inv_den::Array{T,2}         # 
    XtΣ_inv_X::Array{T,2}         # 
    Σ_inv_sp::SparseMatrixCSC{T,Int}  #
    K_β::Array{T,2} # 
end


# function Chan2020new(set_struct, hyp_struct, data_struct)
#     @unpack data_tab, data_mat, var_list = data_struct;
#     @unpack n_fcst, n_save = set_struct;
#     freqL_date = BEAVARs.get_data_freq(data_tab);
#     datesLF = timestamp(data_tab);
#     datesLF_fcast = collect(datesLF[end]+freqL_date:freqL_date:datesLF[end]+freqL_date*(n_fcst));
#     fdatesLF = [datesLF;datesLF_fcast];
#     YY = data_mat;
#     T,n = size(YY);
#     store_YY = fill(NaN,(T+n_fcst,n,n_save))
#     @unpack p,n_burn,n_save,prior_RW = set_struct

#     Y, X, T, n, sigmaP, S_0, Σt_inv, Vβ_inv, Vβ_inv_vecView, Σ_invsp, Σt_LI, XtΣ_inv_den, XtΣ_inv_X, Xsur_den, Xsur_CI, X_CI, k, K_β, beta, intercept, betOLS = BEAVARs.init_Minn(YY,p);

#     priorMinn_struct = BEAVARs.init_priorMinn(n,p,sigmaP,prior_RW,hyp_struct)
 
#     BEAVARs.update_priorMinn!(priorMinn_struct,hyp_struct);
#     Xsur = BEAVARs.XSurFormMatrix(X,n)

#     β_hat = similar(priorMinn_struct.Vinvβ_prior)
#     β_draw = similar(priorMinn_struct.Vinvβ_prior)
#     β_hat_struct = BEAVARs.make_βdraw_struct(β_draw,β_hat, Xsur, XtΣ_inv_den, XtΣ_inv_X, Σ_invsp, K_β)
#     cholK_β = BEAVARs.calc_beta_hat!(priorMinn_struct,β_hat_struct,X,Y);

#     ndraws = n_save+n_burn;
#     store_β=zeros(n^2*p+n,n_save);

#     for ii = 1:ndraws
#         randn!(β_draw);
#         β_hat_struct.β_draw .= β_hat_struct.β_hat .+ ldiv!(cholK_β.U, β_draw); # draw for β
#         if ii>n_burn
#             store_β[:,ii-n_burn] = β_hat_struct.β_draw;
#         end
#     end

#     store_Σ = repeat(vec(S_0),1,n_save);
#     return store_β, store_Σ
# end