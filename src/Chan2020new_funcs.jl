"""
    BVARpriorMinn(β_prior, Cβ_vec, Vβ_vec, Vβ_inv, Vβ_inv_vecView, Vinvβ_prior, idx_kappa1, idx_kappa2, idx_kappa4, idx_kappa5)

    Returns a structure containing the main elements of a Minnesota prior for a Bayesian VAR model. The structure includes the prior mean, variance, and indices for the location of the different types of coefficients (own lags, other lags, constants).
"""
@with_kw struct BVARpriorMinn{T <: AbstractFloat}
    β_prior::Array{T,1}         # vector of either zeros or ones depending on the data transformation
    Cβ_vec::Array{T,1}          # collects the diagonal elements of V but without the hyperparameters
    Vβ_vec::Array{T,1}          # under classical minnesota prior variance is diagonal
    Vβ_inv::Array{T,2}          # inverse of the matrix with a diagonal Vβ_vec
    Vβ_inv_vecView::SubArray{}  # view of the diagonal of Vβ_inv for fast updating
    Vinvβ_prior::Array{T,1}        # vector of V^-1 * β_prior
    idx_kappa1::Array{Int,1}    # indices of parameters associated with own lag
    idx_kappa2::Array{Int,1}    # indices of parameters associated with other variables and their lags
    idx_kappa4::Array{Int,1}    # indices of parameters associated with the constant term
    idx_kappa5::Array{Int,1}    # indices of parameters associated with exogenous variables
    Ψ_vec::Array{T,1}
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

@doc raw"""
    C = mul!(C, A::XSurFormMatrix, x::AbstractVector; α=1.0, β=0.0)

    Matrix-vector multiplication for XSurFromMatrix

    # This function was written by an LLM. There is a test in the test suite that checks the correctness against my own implementation.

"""
function LinearAlgebra.mul!(
    y::AbstractVector, A::XSurFormMatrix, x::AbstractVector,
    α::Number=one(eltype(y)), β::Number=zero(eltype(y)),
)
    T, k = size(A.X)
    n = A.n
    length(x) == k*n && length(y) == T*n ||
        throw(DimensionMismatch("incompatible dimensions for SURDesign * vector"))

    @inbounds for t in 1:T, e in 1:n
        row = (t - 1)*n + e
        xstart = (e - 1)*k
            acc = zero(promote_type(eltype(A), eltype(x)))
        for j in 1:k
                acc += A.X[t, j] * x[xstart + j]
            end
            if iszero(β)
                y[row] = α * acc
            else
                y[row] = β * y[row] + α * acc
        end
    end
    return y
end

# C = L * A, where A is in SUR form
@doc raw"""
    C = mul!(C, L::AbstractMatrix, A::XSurFormMatrix; α=1.0, β=0.0)

    Multiplies a general dense matrix with a matrix in SUR form.

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
    C = mul!(C, A::Adjoint{XSurFormMatrix}, B::AbstractMatrix; α=1.0, β=0.0)

    Multiplies an adjoint matrix in SUR form with a general dense matrix with.

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
    C = mul!(C, A::Adjoint{XSurFormMatrix}, B::SparseMatrixCSC; α=1.0, β=0.0)

    Multiplies an adjoint matrix in SUR form with a sparse matrix with.

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
    C = mul!(C, A::Adjoint{XSurFormMatrix}, B::AbstractBlockBandedMatrix; α=1.0, β=0.0)

    Multiplies the adjoint SUR design matrix by a block-banded matrix whose row
    and column blocks match the time-by-variable layout of the SUR design.

    This function has been written by an LLM and has a testset that checks versus my own implementation
"""
function LinearAlgebra.mul!(
    C::AbstractMatrix, At::Adjoint{<:Any,<:XSurFormMatrix},
    B::BlockBandedMatrices.AbstractBlockBandedMatrix,
    α::Number=one(eltype(C)), β::Number=zero(eltype(C)),
)
    A = parent(At)
    T, k = size(A.X)
    n = A.n
    size(B) == (T*n, T*n) && size(C) == (k*n, T*n) ||
        throw(DimensionMismatch("incompatible dimensions for XSurFormMatrix' * block-banded matrix"))
    all(==(n), BlockArrays.blocklengths(axes(B, 1))) &&
        all(==(n), BlockArrays.blocklengths(axes(B, 2))) ||
        throw(DimensionMismatch("block sizes must match the number of variables in XSurFormMatrix"))

    _scale_output!(C, β)
    lower_bw, upper_bw = BlockBandedMatrices.blockbandwidths(B)

    if iszero(lower_bw) && iszero(upper_bw) &&
       B isa BlockBandedMatrices.BlockBandedMatrix &&
       length(B.data) == T*n*n
        # Dense diagonal blocks are stored contiguously in column-major order.
        data = B.data
        @inbounds for t in 1:T, q in 1:n, e in 1:n, j in 1:k
            row = (e - 1)*k + j
            col = (t - 1)*n + q
            block_data_index = (t - 1)*n*n + (q - 1)*n + e
            C[row, col] += α * conj(A.X[t, j]) * data[block_data_index]
        end
        return C
    end

    @inbounds for s in 1:T
        firstcol = (s - 1)*n
        for t in max(1, s - upper_bw):min(T, s + lower_bw)
            for e in 1:n
                firstrow = (e - 1)*k
                for q in 1:n
                    col = firstcol + q
                    b_row = (t - 1)*n + e
                    b_col = (s - 1)*n + q
                    for j in 1:k
                        C[firstrow + j, col] += α * conj(A.X[t, j]) * B[b_row, b_col]
                    end
                end
            end
        end
    end
    return C
end

function LinearAlgebra.mul!(
    C::AbstractMatrix, At::Adjoint{<:Any,<:SparseMatrixCSC},
    B::BlockBandedMatrices.AbstractBlockBandedMatrix,
    α::Number=one(eltype(C)), β::Number=zero(eltype(C)),
)
    A = parent(At)
    n = first(BlockArrays.blocklengths(axes(B, 1)))
    size(A, 1) % n == 0 ||
        throw(DimensionMismatch("sparse matrix row count must be divisible by the block size"))
    T = div(size(A, 1), n)
    size(B) == (size(A, 1), size(A, 1)) &&
        size(C) == (size(A, 2), size(A, 1)) ||
        throw(DimensionMismatch("incompatible dimensions for sparse adjoint * block-banded matrix"))
    all(==(n), BlockArrays.blocklengths(axes(B, 1))) &&
        all(==(n), BlockArrays.blocklengths(axes(B, 2))) ||
        throw(DimensionMismatch("block sizes must be uniform"))

    _scale_output!(C, β)
    iszero(α) && return C

    rows = rowvals(A)
    vals = nonzeros(A)
    lower_bw, upper_bw = BlockBandedMatrices.blockbandwidths(B)

    if iszero(lower_bw) && iszero(upper_bw) &&
       B isa BlockBandedMatrices.BlockBandedMatrix &&
       length(B.data) == T*n*n
        data = B.data
        @inbounds for i in axes(A, 2), p in nzrange(A, i)
            r = rows[p]
            t, e = fldmod1(r, n)
            outputrow = i
            blockstart = (t - 1)*n
            blockdata = (t - 1)*n*n
            scale = α * conj(vals[p])
            for q in 1:n
                C[outputrow, blockstart + q] +=
                    scale * data[blockdata + (q - 1)*n + e]
            end
        end
        return C
    end

    @inbounds for i in axes(A, 2), p in nzrange(A, i)
        r = rows[p]
        t, e = fldmod1(r, n)
        scale = α * conj(vals[p])
        for s in max(1, t - lower_bw):min(T, t + upper_bw)
            blockstart = (s - 1)*n
            for q in 1:n
                C[i, blockstart + q] += scale * B[r, blockstart + q]
            end
        end
    end
    return C
end

function LinearAlgebra.mul!(
    C::AbstractMatrix,
    A::BlockArrays.BlockedArray{TA,2,<:AbstractMatrix{TA}},
    B::SparseMatrixCSC,
    α::Number=one(eltype(C)), β::Number=zero(eltype(C)),
) where {TA}
    size(A, 2) == size(B, 1) && size(C) == (size(A, 1), size(B, 2)) ||
        throw(DimensionMismatch("incompatible dimensions for block array * sparse matrix"))

    _scale_output!(C, β)
    iszero(α) && return C

    Adata = parent(A)
    rows = rowvals(B)
    vals = nonzeros(B)
    @inbounds for j in axes(B, 2), p in nzrange(B, j)
        k = rows[p]
        scale = α * vals[p]
        for i in axes(A, 1)
            C[i, j] += Adata[i, k] * scale
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
    k_ii   = n*p+1                                 # number of parameters per equation
    k_beta = k_ii*n;
    Cβ_vec = zeros(k_beta,);       
    Vβ_vec = zeros(k_beta,);                       # C diag but without the hyperparameters
    Ψ_vec  = ones(k_beta);                         # For global local priors (not yet implemented)
    β_prior = zeros(k_beta);                       # initialize the prior mean for growth rates, e.g. center at 0
    idx_kappa1 =  Vector{Int}()
    idx_kappa2 =  Vector{Int}()
    idx_kappa4 =  Vector{Int}()
    idx_kappa5 =  Vector{Int}()
    idx_count = 1    
    Vi      = zeros(k_ii,1)                              # vector for equation i

    for ii = 1:n
      for j = 1:n*p+1       # for j=1:n*p+1 
        l = ceil((j-1)/n)           # Here we need a float, as afterwards we will divide by l 
        idx = mod(j-1,n);           # this will count if its own lag, non-own lag, or constant
        if idx==0
            idx = n;
        end
        if j == 1                   # Constant is the first element in each equation
            Vi[j] = 1;
            push!(idx_kappa4,idx_count)
        elseif idx == ii            # These are the own lags of the variable
            if l == 1 && data_trans == 1              # If we have levels instead of growth rates we add a "1" for the first lag
                β_prior[idx_count,] = 1.0;
            end
            Vi[j] = 1.0/l^2;
            push!(idx_kappa1,idx_count)
        else
            Vi[j] = sigmaP_vec[ii]/(sigmaP_vec[idx]*(l^2));
            push!(idx_kappa2,idx_count)
        end
        idx_count += 1
    end

    Cβ_vec[(ii-1)*k_ii+1:ii*k_ii] = Vi
    end
    
    
    Vβ_vec[idx_kappa1] .= Cβ_vec[idx_kappa1] .* hyp_struct.c_hyp_vec[1];  # own lags
    Vβ_vec[idx_kappa2] .= Cβ_vec[idx_kappa2] .* hyp_struct.c_hyp_vec[2];  # other lags
    Vβ_vec[idx_kappa4] .= Cβ_vec[idx_kappa4] .* hyp_struct.c_hyp_vec[4];  # constant
    Vβ_inv = diagm(1.0./Vβ_vec);
    Vβ_inv_vecView = @view(Vβ_inv[diagind(Vβ_inv)]);
    Vinvβ_prior = Vβ_inv*β_prior;
    priorMinn_struct = BVARpriorMinn(β_prior, Cβ_vec, Vβ_vec, Vβ_inv, Vβ_inv_vecView, Vinvβ_prior, idx_kappa1, idx_kappa2, idx_kappa4, idx_kappa5, Ψ_vec)

    return priorMinn_struct

end


@doc raw"""
    prior_struct = init_priorMinn(n,nex,p,vq,sigmaP_vec,sigmaPex_vec,data_trans,hyp_struct)

    Initializes the Minnesota prior structure for a Bayesian VARX model with exogenous variables. The function computes the prior mean and variance for the coefficients based on the number of variables, lags, and hyperparameters provided.

    # Arguments
    - `n`: Number of endogenous variables in the VAR.
    - `p`: Number of lags in the VAR.
    - `sigmaP_vec`: Vector of standard deviations for each variable, used to scale the prior variances.
    - `data_trans`: Indicator for data transformation (e.g., levels or growth rates).
    - `hyp_struct`: Structure containing hyperparameters for the Minnesota prior.

    # Returns
    - `prior_struct`: A `BVARpriorMinn` structure containing the prior mean, variance, and indices for different types of coefficients (own lags, other lags, constants).
"""
function init_priorMinn(n,nex,p,vq,sigmaP_vec,sigmaPex_vec,data_trans,hyp_struct)
    k_ii    = n*p+1+nex*vq                          # number of parameters per equation
    k_beta  = k_ii*n;                               # number of parameters for β
    Cβ_vec  = zeros(k_beta,);       
    Vβ_vec  = zeros(k_beta,);                       # C diag but without the hyperparameters
    Ψ_vec   = ones(k_beta);                         # For global local priors (not yet implemented)
    β_prior = zeros(k_beta);                        # initialize the prior mean for growth rates, e.g. center at 0
    idx_kappa1 =  Vector{Int}()
    idx_kappa2 =  Vector{Int}()
    idx_kappa4 =  Vector{Int}()
    idx_kappa5 =  Vector{Int}()
    idx_count = 1    
    Vi = zeros(k_ii,1)     # vector for equation i

    for ii = 1:n
        for j = 1:n*p+1       # for j=1:n*p+1 
            l = ceil((j-1)/n)           # Here we need a float, as afterwards we will divide by l 
            idx = mod(j-1,n);           # this will count if its own lag, non-own lag, or constant
            if idx==0
                idx = n;
            end
            if j == 1                   # Constant is the first element in each equation
                Vi[j] = 1;
                push!(idx_kappa4,idx_count)
            elseif idx == ii            # These are the own lags of the variable
                if l == 1 && data_trans == 1              # If we have levels instead of growth rates we add a "1" for the first lag
                    β_prior[idx_count,] = 1.0;
                end
                Vi[j] = 1.0/l^2;
                push!(idx_kappa1,idx_count)
            else
                Vi[j] = sigmaP_vec[ii]/(sigmaP_vec[idx]*(l^2));
                push!(idx_kappa2,idx_count)
            end
            idx_count += 1        
        end
        for j = n*p+1+1:n*p+1+nex*vq   
            idx = mod(mod(j-1,n),nex);      # this cycles through 1,2,3 and gives 0 when it reaches nex
            lx = ceil((j-(n*p+1))/n)        # Discounting older lags
            if idx == 0
                idx = nex;
            end
            Vi[j] = sigmaP_vec[ii]/( sigmaPex_vec[idx]*(lx^2) );
            push!(idx_kappa5,idx_count)
            idx_count += 1
        end

        Cβ_vec[(ii-1)*k_ii+1:ii*k_ii] = Vi
    end
    
    
    Vβ_vec[idx_kappa1] .= Cβ_vec[idx_kappa1] .* hyp_struct.c_hyp_vec[1];  # own lags
    Vβ_vec[idx_kappa2] .= Cβ_vec[idx_kappa2] .* hyp_struct.c_hyp_vec[2];  # other lags
    Vβ_vec[idx_kappa4] .= Cβ_vec[idx_kappa4] .* hyp_struct.c_hyp_vec[4];  # constant
    Vβ_vec[idx_kappa5] .= Cβ_vec[idx_kappa5] .* hyp_struct.c_hyp_vec[5];  # exogenous variables
    Vβ_inv = diagm(1.0./Vβ_vec);
    Vβ_inv_vecView = @view(Vβ_inv[diagind(Vβ_inv)]);
    Vinvβ_prior = Vβ_inv*β_prior;
    priorMinn_struct = BVARpriorMinn(β_prior, Cβ_vec, Vβ_vec, Vβ_inv, Vβ_inv_vecView, Vinvβ_prior, idx_kappa1, idx_kappa2, idx_kappa4, idx_kappa5, Ψ_vec)

    return priorMinn_struct

end

"""
    update_priorMinn_broadcast!(prior_struct::BVARpriorMinn,hyp_struct::BEAVARs.BVARmodelHypSetup)

    Updates the variance-covariance matrix of the Minnesota prior coefficients with new hyperparameters.
    The function is easy on the eyes but slower than a hot-loop, therefore not used

    
"""
function update_priorMinn_broadcast!(prior_struct::BVARpriorMinn,hyp_struct::BEAVARs.BVARmodelHypSetup)
    # hot loops are faster than broadcasting
    @views prior_struct.Vβ_vec[prior_struct.idx_kappa1] .= prior_struct.Cβ_vec[prior_struct.idx_kappa1] .* hyp_struct.c_hyp_vec[1];  # own lags
    @views prior_struct.Vβ_vec[prior_struct.idx_kappa2] .= prior_struct.Cβ_vec[prior_struct.idx_kappa2] .* hyp_struct.c_hyp_vec[2];  # other lags
    @views prior_struct.Vβ_vec[prior_struct.idx_kappa4] .= prior_struct.Cβ_vec[prior_struct.idx_kappa4] .* hyp_struct.c_hyp_vec[4];  # constants
    
    prior_struct.Vβ_inv_vecView .= 1.0./prior_struct.Vβ_vec;
    mul!(prior_struct.Vinvβ_prior,prior_struct.Vβ_inv,prior_struct.β_prior)
end


"""
    update_priorMinn!(prior_struct,hyp_struct)

    This is a hot loop version of update_priorMinn_broadcast!()
"""
function update_priorMinn!(prior_struct::BEAVARs.BVARpriorMinn,hyp_struct::BEAVARs.BVARmodelHypSetup)
    @unpack Cβ_vec, Vβ_vec, idx_kappa1, idx_kappa2, idx_kappa4, idx_kappa5, Vβ_inv_vecView, Vβ_inv,Vinvβ_prior,β_prior= prior_struct    
    for ii in eachindex(idx_kappa1)
        Vβ_vec[idx_kappa1[ii]] = 2.0*Cβ_vec[idx_kappa1[ii]] * hyp_struct.c_hyp_vec[1]
    end
    for ii in eachindex(idx_kappa2)
        Vβ_vec[idx_kappa2[ii]] = 2.0*Cβ_vec[idx_kappa2[ii]] * hyp_struct.c_hyp_vec[2]
    end
    for ii in eachindex(idx_kappa4)
        Vβ_vec[idx_kappa4[ii]] = 2.0*Cβ_vec[idx_kappa4[ii]] * hyp_struct.c_hyp_vec[4]
    end
    for ii in eachindex(idx_kappa5)
        Vβ_vec[idx_kappa5[ii]] = 2.0*Cβ_vec[idx_kappa5[ii]] * hyp_struct.c_hyp_vec[5]
    end
    Vβ_inv_vecView .= 1.0./Vβ_vec;
    mul!(Vinvβ_prior,Vβ_inv,β_prior);
end


function calc_β_hat!(prior_struct,β_hat_struct,X,Y)
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


function Chan2020_draw_βsur!(prior_struct,β_hat_struct,X,vecYt)
    @unpack Xsur, XtΣ_inv_den, XtΣ_inv_X, Σ_inv_sp, K_β, β_hat, β_draw, β_rand  = β_hat_struct;
    @unpack Vβ_vec, Vβ_inv, Vβ_inv_vecView, β_prior,Vinvβ_prior = prior_struct;
    Vβ_inv_vecView[:] .= 1.0./Vβ_vec;                   # update the diagonal of Vβ_inv
    update_Xsur!(Xsur, X);                              # update Xs                   
    mul!(Vinvβ_prior, Vβ_inv, β_prior)                  #  update V^-1 * β_Minn 

    mul!(XtΣ_inv_den,Xsur',Σ_inv_sp);                   #  X'*( I(T) ⊗ Σ^{-1} )
    mul!(XtΣ_inv_X,XtΣ_inv_den,Xsur);
    K_β .= Vβ_inv .+ XtΣ_inv_X;                         #  K_β = V^{-1} + X'*( I(T) ⊗ Σ^{-1} )*X
    mul!(Vinvβ_prior,XtΣ_inv_den, vecYt,1.0,1.0);     # (V^-1_Minn * beta_Minn) + X' ( I(T) ⊗ Σ-1 ) y

    cholK_β = cholesky(Hermitian(K_β));                # Cholesky factor
    β_hat[:] = ldiv!(cholK_β.U,ldiv!(cholK_β.U',Vinvβ_prior));    # C'\(C*(V^-1_Minn * beta_Minn + X' ( I(T) ⊗ Σ-1 ) y)
    randn!(β_rand)
    β_draw[:] .= β_hat .+ ldiv!(cholK_β.U,β_rand); # draw for β
end


"""
    function Chan2020_draw_βsur_bb!(prior_struct,β_hat_struct,X,vecYt)

    Draws β using the SUR method. The drawback is that this function mutates Vinvβ_prior in an unexpected way, so should be depreciated
"""
function Chan2020_draw_βsur_bb!(prior_struct,β_hat_struct,X,vecYt)
    @unpack Xsur, XtΣ_inv_den, XtΣ_inv_X, Σ_inv_bb, K_β, β_hat, β_draw, β_rand  = β_hat_struct;
    @unpack Vβ_vec, Vβ_inv, Vβ_inv_vecView, β_prior,Vinvβ_prior = prior_struct;
    Vβ_inv_vecView[:] .= 1.0./Vβ_vec;                   # update the diagonal of Vβ_inv 
    mul!(Vinvβ_prior, Vβ_inv, β_prior)                  #  update V^-1 * β_Minn 
    update_Xsur!(Xsur, X);                              # update Xs                  

    mul!(XtΣ_inv_den,Xsur',Σ_inv_bb);                   #  X'*( I(T) ⊗ Σ^{-1} )
    mul!(XtΣ_inv_X,XtΣ_inv_den,Xsur);
    K_β .= Vβ_inv .+ XtΣ_inv_X;                         #  K_β = V^{-1} + X'*( I(T) ⊗ Σ-1 )*X
    mul!(Vinvβ_prior,XtΣ_inv_den, vecYt,1.0,1.0);       # (V^-1_Minn * beta_Minn) + XtΣ_inv_den* y

    cholK_β = cholesky!(Hermitian(K_β));                # Cholesky factor
    β_hat[:] = ldiv!(cholK_β.U,ldiv!(cholK_β.U',Vinvβ_prior));    # C'\(C*(V^-1_Minn * beta_Minn + X' ( I(T) ⊗ Σ-1 ) y)
    randn!(β_rand)
    β_draw[:] .= β_hat .+ ldiv!(cholK_β.U,β_rand); # draw for β
end

"""
    function Chan2020_draw_βsur_bb_NoPrUpd!(prior_struct,β_hat_struct,X,vecYt)

    Draws β using the SUR method. Does not update the minnesota prior in the function, should be updated outside if needed.
"""
function Chan2020_draw_βsur_bb_NoPrUpd!(prior_struct,β_hat_struct,X,vecYt)
    @unpack Xsur, XtΣ_inv_den, XtΣ_inv_X, Σ_inv_bb, K_β, β_hat, β_draw, β_rand  = β_hat_struct;
    @unpack Vβ_inv, Vinvβ_prior = prior_struct;
    update_Xsur!(Xsur, X);                              # update Xs                  

    mul!(XtΣ_inv_den,Xsur',Σ_inv_bb);                   #  X'*( I(T) ⊗ Σ^{-1} )
    mul!(XtΣ_inv_X,XtΣ_inv_den,Xsur);
    K_β .= Vβ_inv .+ XtΣ_inv_X;                         #  K_β = V^{-1} + X'*( I(T) ⊗ Σ-1 )*X
    mul!(β_hat,XtΣ_inv_den,vecYt)                       # using β_hat as temporary storage for XtΣ_inv_den*y
    β_hat .= Vinvβ_prior .+  β_hat;                     # (V^-1_Minn * beta_Minn) + XtΣ_inv_den* y

    cholK_β = cholesky!(Hermitian(K_β));                # Cholesky factor
    β_hat[:] = ldiv!(cholK_β.U,ldiv!(cholK_β.U',β_hat));    # C'\(C*(V^-1_Minn * beta_Minn + X' ( I(T) ⊗ Σ-1 ) y)
    randn!(β_rand)
    β_draw[:] .= β_hat .+ ldiv!(cholK_β.U,β_rand); # draw for β
end



struct make_βdraw_struct{T <: AbstractFloat}
    β_draw::Array{T,1}        # final draw of β
    β_hat::Array{T,1}         #  posterior mean of β
    β_rand::Array{T,1}        # random draw
    Xsur::XSurFormMatrix{T}         # 
    XtΣ_inv_den::Array{T,2}         # 
    XtΣ_inv_X::Array{T,2}         # 
    Σ_inv_sp::SparseMatrixCSC{T,Int}  #
    # Σ_inv_bb::BlockBandedMatrix{Float64}  #
    K_β::Array{T,2}             # 
end


struct make_iniw_Σt_struct{T <: AbstractFloat}
    Uvec::Array{T,1}
    Σt::Array{T,2}
    Σt_inv::Array{T,2}
    S_0::Diagonal{T}
    nu0::Int
end



# custom H_B matrix
struct HBMatrix{T, M<:AbstractVecOrMat{T}} <: AbstractMatrix{T}
    mBI::M
    Tf::Int
end

getTf(X::HBMatrix) = X.Tf
getn(X::HBMatrix) = size(X.mBI, 1)
getp(X::HBMatrix) = cld(size(X.mBI, 2), getn(X)) - 1
Base.parent(X::HBMatrix) = X.mBI
Base.size(X::HBMatrix) = (getn(X)*getTf(X), getn(X)*getTf(X))
function Base.getindex(X::HBMatrix, i::Int, j::Int)
    @boundscheck checkbounds(X, i, j)
    n = getn(X)
    rowblk, iA = fldmod1(i, n)
    colblk, jA = fldmod1(j, n)
    sourceblk = rowblk - colblk + 1
    sourcecol = (sourceblk - 1)*n + jA
    return get(parent(X), CartesianIndex(iA, sourcecol), zero(eltype(X)))
end

function LinearAlgebra.mul!(C::AbstractMatrix, A::HBMatrix, S::SparseMatrixCSC)
    return LinearAlgebra.mul!(C, A, S, one(eltype(C)), zero(eltype(C)))
end
"""
    mul!(C::AbstractMatrix, A::HBMatrix, S::SparseMatrixCSC,α::Number, β::Number)

    A five-argument mul! for HBMatrix*SparseMatrixCSC multiplication. Written by an LLM, there is a testset to check for correctness.
"""
function LinearAlgebra.mul!(
    C::AbstractMatrix, A::HBMatrix, S::SparseMatrixCSC,
    α::Number, β::Number,
)
    size(A, 2) == size(S, 1) && size(C) == (size(A, 1), size(S, 2)) ||
        throw(DimensionMismatch("incompatible dimensions for HBMatrix * sparse matrix"))

    _scale_output!(C, β)
    iszero(α) && return C

    n = getn(A)
    rows = rowvals(S)
    vals = nonzeros(S)

    for outcol in axes(S, 2), k in nzrange(S, outcol)
        sourcecol = rows[k]
        sourceblock, sourceoffset = fldmod1(sourcecol, n)
        lastblock = min(getTf(A), sourceblock + getp(A))
        scale = α * vals[k]

        for rowblock in sourceblock:lastblock
            lagblock = rowblock - sourceblock + 1
            blockcol = (lagblock - 1)*n + sourceoffset
            blockcol > size(parent(A), 2) && continue
            rowstart = (rowblock - 1)*n
            @inbounds for i in 1:n
                C[rowstart + i, outcol] += scale * parent(A)[i, blockcol]
            end
        end
    end
    return C
end

@doc raw"""
    update_hyp!(priorMinn_struct,β_hat_struct,hyp_struct,n,p)

    Draw the Minnesota hyperparameters κ1 and κ2 using a Generalized Inverse Gaussian (GiG) distribution.

    Uses a prior κj ∼ Gamma(c_{1,j}, c_{2,j}).
"""
function update_hyp!(priorMinn_struct,β_hat_struct,hyp_struct,n,p)
    @unpack β_prior, Cβ_vec, Vβ_vec, Vβ_inv, Vβ_inv_vecView, Vinvβ_prior, idx_kappa1, idx_kappa2, idx_kappa4, Ψ_vec = priorMinn_struct
    @unpack c1_1, c2_1, c1_2, c2_2 = hyp_struct;
    @unpack β_draw = β_hat_struct;

    sum_S_κ1 = sum( (β_draw[idx_kappa1] .- β_prior[idx_kappa1]).^2 ./(2.0.*Ψ_vec[idx_kappa1].*Cβ_vec[idx_kappa1]))
    sum_S_κ2 = sum( (β_draw[idx_kappa2] .- β_prior[idx_kappa2]).^2 ./(2.0.*Ψ_vec[idx_kappa2].*Cβ_vec[idx_kappa2]))

    κ1 = BEAVARs.gigrnd(c1_1 - 0.5*n*p, 2.0*c1_2,sum_S_κ1,1)[];
    κ2 = BEAVARs.gigrnd(c2_1 - 0.5*(n-1)*n*p, 2.0*c2_2,sum_S_κ2,1)[];

    return κ1, κ2
    
end

@doc raw"""
    update_hyp_llmopt!(priorMinn_struct,β_hat_struct,hyp_struct,n,p)

    Draw the Minnesota hyperparameters κ1 and κ2 using a Generalized Inverse Gaussian (GiG) distribution.
    
    Uses a prior κj ∼ Gamma(c_{1,j}, c_{2,j}) and draws from
    $$
    p\left( \kappa_2 \mid \forall \right) 
    \sim \mathcal{GIG} \left( c_{2,1} - \frac{(n - 1)np}{2}, \, 2c_{2,2}, \, \sum_{(i,j) \in S_{\kappa_2}} \frac{(\beta_{i,j} - \underline{\beta}_{i,j})^2}{2\psi_{i,j} C_{i,j}} \right)
    $$

    An LLM-optimized version of update_hyp!(). There is a test that checks whether sum_S_k1 and sum_S_k2 are equivalent
"""
function update_hyp_llmopt!(priorMinn_struct,β_hat_struct,hyp_struct,n,p)
    @unpack β_prior, Cβ_vec, idx_kappa1, idx_kappa2, Ψ_vec = priorMinn_struct
    @unpack c1_1, c2_1, c1_2, c2_2 = hyp_struct;
    @unpack β_draw = β_hat_struct;

    sum_S_κ1 = 0.0
    for idx in idx_kappa1
        diff = β_draw[idx] - β_prior[idx]
        sum_S_κ1 += diff^2 / (2.0 * Ψ_vec[idx] * Cβ_vec[idx])
    end

    sum_S_κ2 = 0.0
    for idx in idx_kappa2
        diff = β_draw[idx] - β_prior[idx]
        sum_S_κ2 += diff^2 / (2.0 * Ψ_vec[idx] * Cβ_vec[idx])
    end

    κ1 = BEAVARs.gigrnd(c1_1 - 0.5*n*p, 2.0*c1_2, sum_S_κ1, 1)[1]
    κ2 = BEAVARs.gigrnd(c2_1 - 0.5*(n-1)*n*p, 2.0*c2_2, sum_S_κ2, 1)[1]

    return κ1, κ2
end



@doc raw"""
    update_hyp_llmopt!(priorMinn_struct,β_hat_struct,hyp_struct,n,p,nx,vq)

    Draw the Minnesota hyperparameters κ1 and κ2 using a Generalized Inverse Gaussian (GiG) distribution.
    
    Uses a prior κj ∼ Gamma(c_{1,j}, c_{2,j}).

    $$
    p\left( \kappa_2 \mid \forall \right) 
    \sim \mathcal{GIG} \left( c_{2,1} - \frac{(n - 1)np}{2}, \, 2c_{2,2}, \, \sum_{(i,j) \in S_{\kappa_2}} \frac{(\beta_{i,j} - \underline{\beta}_{i,j})^2}{2\psi_{i,j} C_{i,j}} \right)
    $$

"""
function update_hyp_llmopt!(priorMinn_struct,β_hat_struct,hyp_struct,n,p,nx,vq)
    @unpack β_prior, Cβ_vec, idx_kappa1, idx_kappa2, idx_kappa5, Ψ_vec = priorMinn_struct
    @unpack c1_1, c2_1, c1_2, c2_2, c5_1, c5_2 = hyp_struct;
    @unpack β_draw = β_hat_struct;

    sum_S_κ1 = 0.0
    for idx in idx_kappa1
        diff = β_draw[idx] - β_prior[idx]
        sum_S_κ1 += diff^2 / (2.0 * Ψ_vec[idx] * Cβ_vec[idx])
    end

    sum_S_κ2 = 0.0
    for idx in idx_kappa2
        diff = β_draw[idx] - β_prior[idx]
        sum_S_κ2 += diff^2 / (2.0 * Ψ_vec[idx] * Cβ_vec[idx])
    end

    sum_S_κ5 = 0.0
    for idx in idx_kappa5
        diff = β_draw[idx] - β_prior[idx]
        sum_S_κ5 += diff^2 / (2.0 * Ψ_vec[idx] * Cβ_vec[idx])
    end


    κ1 = BEAVARs.gigrnd(c1_1 - 0.5*n*p, 2.0*c1_2, sum_S_κ1, 1)[1]           # n*p is the number of c1 elements in β
    κ2 = BEAVARs.gigrnd(c2_1 - 0.5*(n-1)*n*p, 2.0*c2_2, sum_S_κ2, 1)[1]     # (n-1)*n*p is the number of c2 elements
    κ5 = BEAVARs.gigrnd(c5_1 - 0.5*nx*vq*n, 2.0*c5_2, sum_S_κ5, 1)[1]       # nx*vq*n is the number of c5 elements

    return κ1, κ2, κ5
end