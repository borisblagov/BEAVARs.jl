


function makeHypSetup(::Chan2020csv_type)
    return hypChan2020()
end


@doc raw"""
    makeDataSetup(::Chan2020csv_type,data_tab::TimeArray; var_list =  colnames(data_tab))

Generate a dataset strcture for use with the single-frequency models
"""
function makeDataSetup(::Chan2020csv_type,data_tab::TimeArray; var_list =  colnames(data_tab))
    return dataBVAR_TA(data_tab, var_list)
end




@doc raw"""
# Chan2020csv(YY,VARSetup,hypSetup)

Implements the BVAR with Minnesota prior with a SUR form and common stochastic volatilty (csv) following Chan (2020)

"""
function Chan2020csv(YY::Array{Float64},VARSetup::BVARmodelSetup,hypSetup::BVARmodelHypSetup)
    @unpack ρ, σ_h2, v_h0, S_h0, ρ_0, V_ρ = hypSetup
    @unpack p, nsave, nburn, prior_RW = VARSetup

    Y, X, T, n = mlagL(YY,p)
    (deltaP, sigmaP, mu_prior) = trainPriors(YY,p)
    np1 = n*p+1; # number of parameters per equation
    (idx_kappa1,idx_kappa2, V_Minn) = prior_NatConj(n,p,sigmaP,hypSetup);
    

    A_0 = zeros(n*p+1,n)
    if prior_RW == 1
        A_0[2:n+1,1:n] = Matrix(1.0I, n, n)     # account for the constant on the top row
    end
    V_Ainv = sparse(1:np1,1:np1,1.0./V_Minn)
    S_0 = Diagonal(sigmaP);
    Σ = S_0;
    v_0 = n+3;
    h = zeros(T,)
    H_ρ = sparse(Matrix(1.0I, T, T)) - sparse(ρ*diagm(-1=>repeat(1:1,T-1)));
    # dv = ones(T,); ev = -ρ.*ones(T-1,);
    # H_ρ = Bidiagonal(dv,ev,:L)
    
    # This part follows page 19 of Chan, J. (2020)
    ndraws = nsave+nburn;
    store_β = zeros(np1*n,nsave);
    store_h = zeros(T,nsave);
    store_Σ = zeros(n,n,nsave);
    store_s2_h = zeros(T,nsave);
    store_ρ = zeros(nsave,);
    store_σ_h2 = zeros(nsave,); 
    store_eh = zeros(T,nsave);
    
    eh = similar(h);

    Ωinv = sparse(1:T,1:T,exp.(-h));
    dg_ind_Ωinv = diagind(Ωinv);
    for ii = 1:ndraws 
        Ωinv[dg_ind_Ωinv]=exp.(-h);
        ZtΩinv = X'*Ωinv;
        
        K_A = V_Ainv + ZtΩinv*X;
        A_hat = K_A\(V_Ainv\A_0 + ZtΩinv*Y);
        S_hat = S_0 + A_0'*V_Ainv*A_0 + Y'*Ωinv*Y - A_hat'*K_A*A_hat;
        S_hat = (S_hat+S_hat')/2;
    
        Σ = rand(InverseWishart(v_0+T,S_hat));
        cholΣ  = cholesky(Σ).U; # if we get the upper we don't need constantly to transpose
        A = A_hat + (cholesky(Hermitian(K_A)).U\randn(np1,n))*cholΣ;
    
        # Errors
        U = Y - X*A
        s2_h = sum((U/cholΣ).^2,dims=2)
    
        BEAVARs.draw_h_csv!(h,s2_h,ρ,σ_h2,n,H_ρ)
        eh[1,] = h[1]*sqrt(1-ρ^2);
        eh[2:end,] = h[2:end,].-ρ.*h[1:end-1,]
        σ_h2 = 1.0./rand(Gamma(v_h0+T/2.0, 1.0/( S_h0 + sum(eh.^2)./2.0 ) ) )
    
        K_rho = 1.0/V_ρ + sum(h[1:T-1,].^2)./σ_h2;
        ρ_hat = K_rho\(ρ_0/V_ρ + h[1:T-1,]'*h[2:T,]./σ_h2);
        ρ_c = ρ_hat + sqrt(K_rho)\randn();
    
        g_ρ(x) = -0.5*log(σ_h2./(1-x.^2))-0.5*(1.0-x.^2)/σ_h2*h[1]^2;
        if abs(ρ_c)<.999
            alpMH = exp(g_ρ(ρ_c)-g_ρ(ρ));
            if alpMH>rand()
                ρ = ρ_c;            
            end
        end    
        H_ρ[diagind(H_ρ,-1)]=fill(-ρ,T-1);
        # H_ρ.ev.=-ρ.*ones(T-1,)

        if ii>nburn
            store_β[:,ii-nburn] = vec(A);
            store_h[:,ii-nburn] = h;
            store_Σ[:,:,ii-nburn] = Σ;
            store_s2_h[:,ii-nburn] = s2_h;
            store_ρ[ii-nburn,] = ρ;
            store_σ_h2[ii-nburn,] = σ_h2;
            store_eh[:,ii-nburn] = eh;
        end
    end

    return store_β, store_h, store_Σ, store_s2_h, store_ρ, store_σ_h2, store_eh
    
end # end function Chan2020csv


@doc raw"""
    draw_h_csv!(h,s2_h,ρ,σ_h2,n,H_ρ)

Draws the log-volatilities h by mutating the array from the previous draw using an MH algorithm

"""
function draw_h_csv!(h,s2_h,ρ,σ_h2,n,H_ρ)

    accept = false;
    T_h = size(s2_h,1);
    # H_ρ = sparse(Matrix(1.0I, T_h, T_h)) - sparse(ρ*diagm(-1=>repeat(1:1,T_h-1)));
    H_inv = H_ρ'*sparse(1:T_h,1:T_h,[(1-ρ^2)/σ_h2; 1/σ_h2*ones(T_h-1,)])*H_ρ
    ϵ_h = 1.0;
    ht = similar(h); ht[:,] = h[:,];
    
    local(K_h)
    while ϵ_h > 10^(-3)
        eht = exp.(ht);
        sieht = s2_h[:,]./eht
        f_h = -n/2.0 .+ 0.5*sieht;
        G_h = 0.5*sieht
        K_h = H_inv + sparse(1:T_h,1:T_h,G_h);
        new_ht = K_h\(f_h + G_h.*ht)
        ϵ_h = maximum(abs.(new_ht-ht))
        ht[:,] = new_ht[:,]
    end
    C_K_h = cholesky(K_h)
    
    hstar = ht
    logc = -0.5*hstar'*H_inv*hstar - (n/2.0) *sum(hstar) .- 0.5*exp.(-hstar)'*s2_h[:,] .+ log(3)
    
    flag = false
    local hc = similar(h)
    local alpARc = 0.0;
    while flag == 0
        hc = ht + C_K_h.U\randn(T_h,);        
        alpARc = -0.5*hc'*H_inv*hc .- (n/2.0)*sum(hc) .- 0.5*exp.(-hc)'*s2_h[:,] .+ 0.5*(hc-ht)'*K_h*(hc-ht) .- logc;
        
        if alpARc > log(rand())
            flag = true;
        end
    end        
    alpAR = -0.5*h'*H_inv*h .- (n/2.0)*sum(h) .- 0.5*exp.(-h)'*s2_h[:,] .+ 0.5*(h-ht)'*K_h*(h-ht) .- logc;
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

    
    return h, accept
end # end draw_h_csv!





function Chan2020_draw_h!(h,s2_h,ρ,σ_h2,n,H_ρ,T)

    accept = false;
    T_h = size(s2_h,1);
    I_Th = Matrix(I,T_h,T_h);
    H_ρ[diagind(H_ρ,-1)]=fill(-ρ,T-1);
    # H_ρ = sparse(Matrix(1.0I, T_h, T_h)) - sparse(ρ*diagm(-1=>repeat(1:1,T_h-1)));
    # H_inv = H_ρ'*sparse(1:T_h,1:T_h,[(1.0-ρ^2)/σ_h2; 1.0/σ_h2*ones(T_h-1,)])*H_ρ
    H_inv = H_ρ'*(I_Th.*[(1.0-ρ^2)/σ_h2; 1.0/σ_h2*ones(T_h-1,)])*H_ρ;
    ϵ_h = 1.0;
    ht = similar(h); ht[:,] = h[:,];
    
    I_Th = Matrix(I,T_h,T_h);
    local(K_h)
    while ϵ_h > 10^(-3)
        eht = exp.(ht);
        sieht = s2_h[:,]./eht
        f_h = -n/2.0 .+ 0.5*sieht;
        G_h = 0.5*sieht
        # K_h = H_inv + sparse(1:T_h,1:T_h,G_h);
        K_h = H_inv + I_Th.*G_h;
        new_ht = K_h\(f_h + G_h.*ht)
        ϵ_h = maximum(abs.(new_ht-ht))
        ht[:,] = new_ht[:,]
    end
    C_K_h = cholesky(K_h)
    
    hstar = ht
    logc = -0.5*hstar'*H_inv*hstar - (n/2.0) *sum(hstar) .- 0.5*exp.(-hstar)'*s2_h[:,] .+ log(3)
    
    flag = false
    hc = similar(h)
    alpARc = 0.0;
    while flag == 0
        hc = ht + C_K_h.U\randn(T_h,);        
        alpARc = -0.5*hc'*H_inv*hc .- (n/2.0)*sum(hc,dims=1)[1] .- 0.5*exp.(-hc)'*s2_h[:,] .+ 0.5*(hc-ht)'*K_h*(hc-ht) .- logc;
        
        if alpARc > log(rand())
            flag = true;
        end
    end        
    alpAR = -0.5*h'*H_inv*h .- (n/2.0)*sum(h) .- 0.5*exp.(-h)'*s2_h[:,] .+ 0.5*(h-ht)'*K_h*(h-ht) .- logc;
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

    return h
end # end draw_h_csv!

function Chan2020_draw_ρ!(ρ,h,eh,v_h0,S_h0,ρ_0,V_ρ,T)
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

    return ρ, σ_h2, eh
end

@with_kw struct VAROutput_Chan2020csv <: BVARmodelOutput
    store_β::Array{}        # 
    store_Σ::Array{}        # 
    store_h::Array{}        # 
    s2_h_store::Array{}     # 
    store_ρ::Array{}        # 
    store_σ_h2::Array{}     # 
    store_eh::Array{}       #
    YY::Array{}             #
end




"""
    
"""
function forecast(VAROutput::VAROutput_Chan2020csv,VARSetup)
    @unpack store_β, store_Σ, store_h, s2_h_store, store_ρ, store_σ_h2,store_eh, YY = VAROutput
    @unpack n_fcst,p,nsave = VARSetup
    n = size(YY,2);

    Yfor3D    = fill(NaN,(p+n_fcst,n,nsave))
    hfor3D    = fill(NaN,(p+n_fcst,nsave)); 
    
    
    Yfor3D[1:p,:,:] .= @views YY[end-p+1:end,:];
    hfor3D[1:p,:] = @views store_h[end-p+1:end,:];
    
    for i_draw = 1:nsave
    
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






"""
    
"""
function Chan2020_drawA(Y,X,n,k,T,v_0,h,Ωinv,dg_ind_Ωinv,V_Ainv,S_0,VAinvDA0,AVAinvA)
    Ωinv[dg_ind_Ωinv]=exp.(-h);
    ZtΩinv = X'*Ωinv;    
    K_A = V_Ainv + ZtΩinv*X;
    cholK_A = cholesky(Hermitian(K_A));
    # A_hat = K_A\(VAinvDA0 + ZtΩinv*Y);
    A_hat = ldiv!(cholK_A.U,ldiv!(cholK_A.L,(VAinvDA0 + ZtΩinv*Y)));
    S_hat = S_0 + AVAinvA + Y'*Ωinv*Y - A_hat'*K_A*A_hat;
    S_hat = (S_hat+S_hat')/2;

    Σ = rand(InverseWishart(v_0+T,S_hat));
    cholΣU  = cholesky(Σ).U; # if we get the upper we don't need constant transpose
    A = A_hat + ldiv!(cholK_A.U,randn(k,n))*cholΣU;
    
    U = Y - X*A
    s2_h = sum((U/cholΣU).^2,dims=2)
    return A, cholΣU, Σ, s2_h, U, Ωinv
end


# # Chan2020csv(YY,VARSetup,hypSetup)
# """
# Implements the BVAR with Minnesota prior with a SUR form and common stochastic volatilty (csv) following Chan (2020)

# """
# function Chan2020csv2(YY::Array{Float64},VARSetup::BVARmodelSetup,hypSetup::BVARmodelHypSetup)
#     @unpack ρ, σ_h2, v_h0, S_h0, ρ_0, V_ρ = hypSetup
#     @unpack p, nsave, nburn, prior_RW = VARSetup
    
#     Y, X, T, n, k, sigmaP, S_0, Σ, A_0, V_Ainv, v_0, H_ρ,h,eh,Ωinv, dg_ind_Ωinv, VAinvDA0, AVAinvA = BEAVARs.initcsv(YY,p,hypSetup,prior_RW);
#     # This part follows page 19 of Chan, J. (2020)
#     ndraws = nsave+nburn;
#     store_β = zeros(k*n,nsave);
#     store_h = zeros(T,nsave);
#     store_Σ = zeros(n,n,nsave);
#     store_s2_h = zeros(T,nsave);
#     store_ρ = zeros(nsave,);
#     store_σ_h2 = zeros(nsave,); 
#     store_eh = zeros(T,nsave);

#     for ii = 1:ndraws 
#         A, cholΣU, Σ, s2_h = BEAVARs.Chan2020_drawA(Y,X,n,k,T,v_0,h,Ωinv,dg_ind_Ωinv,V_Ainv,S_0,VAinvDA0,AVAinvA);
#         h = BEAVARs.Chan2020_draw_h!(h,s2_h,ρ,σ_h2,n,H_ρ,T);
#         ρ, σ_h2, eh = BEAVARs.Chan2020_draw_ρ!(ρ,h,eh,v_h0,S_h0,ρ_0,V_ρ,T);

#         if ii>nburn
#             store_β[:,ii-nburn] = vec(A);
#             store_h[:,ii-nburn] = h;
#             store_Σ[:,:,ii-nburn] = Σ;
#             store_s2_h[:,ii-nburn] = s2_h;
#             store_ρ[ii-nburn,] = ρ;
#             store_σ_h2[ii-nburn,] = σ_h2;
#             store_eh[:,ii-nburn] = eh;
#         end
#     end

#     return store_β, store_h, store_Σ, store_s2_h, store_ρ, store_σ_h2, store_eh
    
# end # end function Chan2020csv
