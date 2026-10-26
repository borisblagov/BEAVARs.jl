# ------------------------------------------------------------------
# Minnesota prior with common stochastic volatility as in Carriero, A., Clark, T. E., & Marcellino, M. (2016). Common Drifting Volatility in Large Bayesian VARs. Journal of Business & Economic Statistics, 34(3), 375–390. https://doi.org/10.1080/07350015.2015.1040116
# Differences to the CCK (2016) implementation lie in the specification of the Minnesota prior, where we introduce cross-variable shrinkage as in Chan2020iniw



#-------------------------------------
# The Den of the beavar
#-------------------------------------

@doc raw"""
    beavar(::CCK2016csv_type, set_struct, hyp_str, data_struct)

    Estimates a BVAR with common stochastic volatility as proposed by Carriero, Clark, and Marcellino (2016). 

    The implementation of csv follows Chan (2020) and draws `h`, the stochastic volatility, using a Newton Rapson method and not the CCK article with Shephard and Chib's 1998 method.

"""
function beavar(::CCK2016csv_type, set_struct, hyp_str, data_struct)
    println("Hello CCK2016csv")
    @unpack data_tab, data_mat, var_list = data_struct;
    freqL_date = BEAVARs.get_data_freq(data_tab);
    datesLF = timestamp(data_tab);
    datesLF_fcast = collect(datesLF[end]+freqL_date:freqL_date:datesLF[end]+freqL_date*(set_struct.n_fcst));
    fdatesLF = [datesLF;datesLF_fcast];
    YY = values(data_struct.data_tab);
    store_β,store_Σ,store_h,store_s2_h, store_ρ, store_σ_h2, store_eh,store_c1, store_c2 = BEAVARs.CCK2016csv(YY,set_struct,hyp_str);
    out_struct = VAROutput_CCK2016csv(store_β,store_Σ,store_h,store_s2_h, store_ρ, store_σ_h2, store_eh,store_c1,store_c2,YY)
    return out_struct
end


@doc raw"""
    makeHypSetup(::CCK2016csv_type)

    Generate a hyperparameter strcture for use with the single-frequency models
    
"""
function makeHypSetup(::CCK2016csv_type)
    return hypChan2020()
end


@doc raw"""
    makeDataSetup(::CCK2016csv_type,data_tab::TimeArray; var_list =  colnames(data_tab))

Generate a dataset strcture for use with the single-frequency models
    
# Arguments
    model_type: The custom model type (not a string)
    data_tab:   TimeArray with the data_de
    var_list (optional):   A symbol list with the variable names. Will be used for oredering the variables. Uses by default the names from data_tab if not supplied. note that Symbol lists have a particular synthax.
# Returns
    data_struct: A dataBVAR_TA structure with the data and metadata
"""
function makeDataSetup(::CCK2016csv_type,data_tab::TimeArray; var_list =  colnames(data_tab))
    return data_BVAR(data_tab,values(data_tab), var_list)
end







struct VAROutput_CCK2016csv{T <: AbstractFloat} <: BVARmodelOutput
    store_β::Array{T,2}        # 
    store_Σ::Array{T,3}        # 
    store_h::Array{T,2}        # 
    store_s2_h::Array{T,2}     # 
    store_ρ::Array{T,1}        # 
    store_σ_h2::Array{T,1}     # 
    store_eh::Array{T,2}       #
    store_c1::Array{T,1}       #
    store_c2::Array{T,1}       #
    YY::Array{T,2}             #
end



@doc raw"""
    BEAVARs.init_Minn_csv(YY, p)

"""
function init_Minn_csv(YY,set_struct,hyp_struct)
    @unpack p = set_struct;
    @unpack ρ = hyp_struct;
    if set_struct.const_loc == 1
        # will err if false, on purpose to catch if we define the constant wrong.
        Y, X, T, n, intercept       = mlagL(YY,p);
    end
    sigmaP_ar4, betaOLS_ar4     = ar4(YY);                                  # do OLS to initialize priors as in CCK2016
    k                           = n*p+intercept
    h                           = zeros(T,);                                # stochastic volatilities
    S_0                         = diagm(sigmaP_ar4);                            # initialize S_0
    Σt_inv                      = S_0\I;                                    # initialize Σ^-1   
    Σ_inv_bb                    = BlockBandedMatrix{Float64}(undef,ones(Int,T,).*n, ones(Int,T,).*n, (0, 0))
    XtΣ_inv_den                 = zeros(k*n,T*n);                           # will be X' ( I(T) ⊗ Σ-1 )   from page 6 in Chan 2020 LBA
    XtΣ_inv_X                   = zeros(n*k,n*k);                           # will be X' ( I(T) ⊗ Σ-1 ) X from page 6 in Chan 2020 LBA    
    Xsur                        = BEAVARs.XSurFormMatrix(X,n);              # prepares the SUR form and the indices of the parameters for updating
    K_β                         = zeros(n*k,n*k);                           # Variance covariance matrix of the parameters
    
    priorMinn_struct = BEAVARs.init_priorMinn(n,p,sigmaP_ar4,set_struct.prior_RW,hyp_struct)

    β_hat_struct = BEAVARs.make_βdraw_bb_struct(zeros(k*n,),zeros(k*n,),zeros(k*n,), Xsur, XtΣ_inv_den, XtΣ_inv_X, Σ_inv_bb, K_β)

    Uvec = similar(vec(Y'))
    Σt   = similar(S_0)
    csv_Σt_struct = BEAVARs.make_csv_Σt_struct(Uvec,Σt,Σt_inv,S_0,hyp_struct.nu0,hyp_struct.nu0+T+n,cholesky(S_0).U)

    csv_h_struct = init_csv_struct(h,hyp_struct,T,T+p)
    return Y, X, T, n, sigmaP_ar4, betaOLS_ar4, priorMinn_struct, β_hat_struct, csv_Σt_struct, csv_h_struct
end


@doc raw"""
    CCK2016_drawΣtsur(vecYt,β_hat_struct,iniw_Σt_struct,n,T)

    Draw Σt matrix and calculates its inverse using an Inverse Wishart distribution.
"""
function CCK2016_drawΣtsur!(vecYt,β_hat_struct,csv_Σt_struct,csv_h_struct,n,T)
    @unpack Uvec, Σt, Σt_inv, S_0, nu0, cholΣt_U = csv_Σt_struct
    @unpack s2_h = csv_h_struct
    Uvec .= vecYt .- mul!(Uvec,β_hat_struct.Xsur,β_hat_struct.β_draw) 
    U = reshape(Uvec,n,T);   
    BEAVARs.fast_triple_product!(Σt, U, csv_h_struct.expmh);                   # store U*Diagonal(exp.(-h)*U' + S_0 in Σt
    Σt .= rand(InverseWishart(csv_Σt_struct.iw_df, Σt .+ csv_Σt_struct.S_0));    # draw for Σ
    Σt_inv .= Σt\I;
    cholΣt_U .= cholesky(Σt).U;
    s2_h .= sum((U'/cholΣt_U).^2,dims=2);
end


@doc raw"""

    BEAVARs.update_Σ_inv_bb_csv!(Σ_inv_bb,Σt_inv,expmh,T,n)

    Updates the block-banded matrix Σ_inv_bb with Σt_inv end exp.(-h) across the block diagonal

"""
function update_Σ_inv_bb_csv!(Σ_inv_bb,Σt_inv,expmh,T,n)
    diagonal_block = vec(Σt_inv)
    for ii in 1:T
        copyto!(Σ_inv_bb.data, (ii - 1)*n*n + 1, diagonal_block.*expmh[ii], 1, n*n)
    end
end


struct make_βdraw_bb_struct{T <: AbstractFloat}
    β_draw::Array{T,1}        # final draw of β
    β_hat::Array{T,1}         #  posterior mean of β
    β_rand::Array{T,1}        # random draw
    Xsur::XSurFormMatrix{T}         # 
    XtΣ_inv_den::Array{T,2}         # 
    XtΣ_inv_X::Array{T,2}         # 
    Σ_inv_bb::BlockBandedMatrix{Float64}  #
    K_β::Array{T,2}             # 
end



struct make_csv_h_struct{T <: AbstractFloat}
    h::Array{T,1}                   # h for T
    expmh::Array{T,1}               # exp.(-h) for T
    ht::Array{T,1}
    eht::Array{T,1}
    new_ht::Array{T,1}
    sieht::Array{T,1}
    G_h::Array{T,1}
    f_h::Array{T,1}
    s2_h::Array{T,1}
    eh::Array{T,1}
    H_ρ::SparseMatrixCSC{T,Int}
    ρ_vec::Array{T,1}
    σ_h2_vec::Array{T,1}
    K_h::SparseMatrixCSC{T,Int}
    expmhf::Array{T,1}                  # exp.(-h) for T+p = Tf, used for the presicion based sampler, where we need the initial conditions
end


struct make_csv_Σt_struct{T <: AbstractFloat}
    Uvec::Array{T,1}
    Σt::Array{T,2}
    Σt_inv::Array{T,2}
    S_0::Array{T,2}
    nu0::Int
    iw_df::Int             # degrees of freedom for the inverse Wishart
    cholΣt_U::UpperTriangular{T, Matrix{T}};
end

function init_csv_struct(h,hyp_struct,T,Tf)
    @unpack ρ, σ_h2, v_h0, S_h0, ρ_0, V_ρ = hyp_struct
    expmhf      = [repeat([exp.(-h[1])],Tf-T); exp.(-h)];  # currently we just copy the first one. drawing the initial conditions would be better for correctness
    ρ_vec       = zeros(1,); σ_h2_vec = zeros(1,);
    ρ_vec[1]    = ρ;
    σ_h2_vec[1] = σ_h2;
    H_ρ         = sparse(Matrix(1.0I, T, T)) - sparse(ρ*diagm(-1=>repeat(1:1,T-1)));
    H_inv       = H_ρ'*(Diagonal([(1.0-ρ^2)/σ_h2; 1.0/σ_h2*ones(T-1,)]))*H_ρ;
    K_h         = similar(H_inv);
    csv_h_struct = BEAVARs.make_csv_h_struct(h,exp.(-h),similar(h),similar(h),similar(h),similar(h),similar(h),similar(h),similar(h),similar(h),H_ρ,ρ_vec,σ_h2_vec,K_h,expmhf)
    return csv_h_struct
end


"""
    fast_triple_product!(C, U, dv)

    Calculates U*Diagonal(dv)*U' without allocations and stores it in C
"""
function fast_triple_product!(C, U, dv)
    m, n = size(U) # m = 10, n = 76
    
    for j in 1:m
        for i in 1:j
            val = zero(eltype(C))
            @inbounds @simd for k in 1:n                
                val += U[i, k] * dv[k] * U[j, k] # Essentially C[i,j] = ∑ (U[i,k] * dv[k] * U[j,k])
            end
            C[i, j] = val
            C[j, i] = val # Exploit symmetry to save half the work
        end
    end
    return C
end





function csv_draw_h_ρ!(csv_h_struct, n,T,hyp_struct)
    @unpack v_h0, S_h0, ρ_0, V_ρ = hyp_struct
    @unpack h, expmh, ht, eht, new_ht, sieht, G_h, f_h, s2_h, eh, H_ρ, ρ_vec,σ_h2_vec, K_h = csv_h_struct
    accept = false;
    ρ = ρ_vec[1]; 
    σ_h2 = σ_h2_vec[1];
    # H_ρ[diagind(H_ρ,-1)]=fill(-ρ,T-1);
    H_ρ.nzval[2:2:end].=-ρ;
    H_inv = H_ρ'*(Diagonal([(1.0-ρ^2)/σ_h2; 1.0/σ_h2*ones(T-1,)]))*H_ρ;
    ϵ_h = 1.0;
    ht .= h; 
    
    while ϵ_h > 10^(-3)
        eht .= exp.(ht)
        sieht .= s2_h[:,]./eht
        G_h .= 0.5*sieht
        f_h .= -n*0.5 .+ G_h;
        K_h .= H_inv .+ Diagonal(G_h);
        new_ht .= K_h\(f_h + G_h.*ht)
        ϵ_h = maximum(abs.(new_ht-ht))
        ht[:,] = new_ht[:,]
    end

    C_K_h = cholesky(K_h)
    
    hstar = similar(ht)
    hstar .= ht
    logc = -0.5*hstar'*H_inv*hstar - 0.5*n*sum(hstar) .- 0.5*exp.(-hstar)'*s2_h[:,] .+ log(3)
    
    flag = false
    hc = similar(h)
    alpARc = 0.0;
    while flag == 0
        hc = ht + C_K_h.U\randn(T,);        
        alpARc = -0.5*hc'*H_inv*hc .- (0.5*n)*sum(hc,dims=1)[1] .- 0.5*exp.(-hc)'*s2_h[:,] .+ 0.5*(hc-ht)'*K_h*(hc-ht) .- logc;
        
        if alpARc > log(rand())
            flag = true;
        end
    end        
    alpAR = -0.5*h'*H_inv*h .- (0.5*n)*sum(h) .- 0.5*exp.(-h)'*s2_h[:,] .+ 0.5*(h-ht)'*K_h*(h-ht) .- logc;
    if alpAR < 0
        alpMH = 1.0;
    elseif alpARc < 0
        alpMH = - alpAR;
    else
        alpMH = alpARc - alpAR;
    end    
    if alpMH > log(rand())
        h[:,] = hc[:,]; 
        accept = true;
    end

    # update exp.(-h)
    expmh .= exp.(-h);

    # draw ρ and σ_h2
    eh[1,] = h[1]*sqrt(1.0-ρ^2);
    eh[2:end,] = h[2:end,].-ρ.*h[1:end-1,]
    σ_h2 = 1.0./rand(Gamma(v_h0+T/2.0, 1.0/( S_h0 + sum(eh.^2)./2.0 ) ) )

    K_rho = 1.0/V_ρ + sum(h[1:T-1,].^2)./σ_h2;
    ρ_hat = K_rho\(ρ_0/V_ρ + h[1:T-1,]'*h[2:T,]./σ_h2);
    ρ_c = ρ_hat + sqrt(K_rho)\randn();

    g_ρ(x) = -0.5*log(σ_h2./(1.0-x.^2))-0.5*(1.0-x.^2)/σ_h2*h[1]^2;
    if abs(ρ_c)<.999
        alpMH = exp(g_ρ(ρ_c)-g_ρ(ρ));
        if alpMH>rand()
            ρ = ρ_c;            
        end
    end    

    # update ρ
    ρ_vec[1] = ρ;
    σ_h2_vec[1] = σ_h2;

end 



function CCK2016csv(YY::Array{Float64},set_struct::BVARmodelSetup,hyp_struct::BVARmodelHypSetup)
    # @unpack ρ, σ_h2, v_h0, S_h0, ρ_0, V_ρ = hyp_struct
    @unpack p, n_save, n_burn, prior_RW, est_hyp = set_struct
    n_draws  = n_save+n_burn;


    Y, X, T, n, sigmaP_ar4, betaOLS_ar4, priorMinn_struct, β_hat_bb_struct, csv_Σt_struct, csv_h_struct = BEAVARs.init_Minn_csv(YY,set_struct,hyp_struct)
    vecYt = vec(Y')
    
    store_β = zeros(n^2*p+n,n_save);
    store_Σ = zeros(n,n,n_save);
    store_h = zeros(T,n_save);
    store_s2_h = zeros(T,n_save);
    store_ρ = zeros(n_save,);
    store_σ_h2 = zeros(n_save,); 
    store_eh = zeros(T,n_save);
    store_c1 = zeros(n_save,);
    store_c2 = zeros(n_save,);
    
    for ii = 1:n_draws 
        BEAVARs.update_Σ_inv_bb_csv!(β_hat_bb_struct.Σ_inv_bb,csv_Σt_struct.Σt_inv,csv_h_struct.expmh,T,n);
        BEAVARs.Chan2020_draw_βsur_bb_NoPrUpd!(priorMinn_struct,β_hat_bb_struct,X,vecYt);
        BEAVARs.CCK2016_drawΣtsur!(vecYt,β_hat_bb_struct,csv_Σt_struct,csv_h_struct,n,T);
        BEAVARs.csv_draw_h_ρ!(csv_h_struct, n,T,hyp_struct); 

        if est_hyp == true
            κ1, κ2 = BEAVARs.update_hyp_llmopt!(priorMinn_struct,β_hat_bb_struct,hyp_struct,n,p)
            hyp_struct = hypChan2020(c1=κ1,c2=κ2);
            BEAVARs.update_priorMinn!(priorMinn_struct,hyp_struct);
        end

        if ii>n_burn
            store_β[:,ii-n_burn] = β_hat_bb_struct.β_draw;
            store_Σ[:,:,ii-n_burn] = csv_Σt_struct.Σt;
            store_h[:,ii-n_burn] = csv_h_struct.h;
            store_s2_h[:,ii-n_burn] = csv_h_struct.s2_h;
            store_ρ[ii-n_burn,] = csv_h_struct.ρ_vec[1];
            store_σ_h2[ii-n_burn,] = csv_h_struct.σ_h2_vec[1];
            store_eh[:,ii-n_burn] = csv_h_struct.eh;
            store_c1[ii-n_burn,] = hyp_struct.c1;
            store_c2[ii-n_burn,] = hyp_struct.c2;
        end
    end
    return store_β, store_Σ, store_h, store_s2_h, store_ρ, store_σ_h2, store_eh,store_c1, store_c2
end 



"""
    
"""
function forecast(VAROutput::VAROutput_CCK2016csv,set_struct)
    @unpack store_β, store_Σ, store_h, store_s2_h, store_ρ, store_σ_h2,store_eh, YY = VAROutput
    @unpack n_fcst,p,n_save = set_struct
    n = size(YY,2);

    Yfor3D    = fill(NaN,(p+n_fcst,n,n_save))
    hfor3D    = fill(NaN,(p+n_fcst,n_save)); 
    
    
    Yfor3D[1:p,:,:] .= @views YY[end-p+1:end,:];
    hfor3D[1:p,:] = @views store_h[end-p+1:end,:];
    
    for i_draw = 1:n_save
    
        hfor = @views hfor3D[:,i_draw];
        Yfor = @views Yfor3D[:,:,i_draw];
        A_draw = @views reshape(store_β[:,i_draw],n*p+1,n);
        ρ_draw = @view store_ρ[i_draw];
        σ_h2_draw = @views store_σ_h2[i_draw];
        Σ_draw = @views store_Σ[:,:,i_draw];
                
        for i_for = 1:n_fcst
            hfor[p+i_for,] = ρ_draw.*hfor[p+i_for-1,] + sqrt(σ_h2_draw).*randn()
            tclass = @views vec(reverse(Yfor[1+i_for-1:p+i_for-1,:],dims=1)')
            tclass = [1;tclass];
            Yfor[p+i_for,:]=tclass'*A_draw  .+ (exp.(hfor[p+i_for,]./2.0)*cholesky(Σ_draw).U*randn(n,1))';    
        end
    end
    return Yfor3D


end # end function forecast Chan2020csv()
