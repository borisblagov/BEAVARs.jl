"""
    irf_chol(beta_vec,sigma_vec,n::Integer,p::Integer,const_loc::Integer,n_irf::Integer,IRF_mat;shSize::String = "stdev")

    Calculates impulse responses for a set of parameters in a vector beta_vec and variance-covariance matrix in a vector sigma_vec
    Uses the Cholesky decomposition and the companion form.
        - beta_vec: needs to have the parameters for each equation stacked on top of each other
        - const_loc: the function needs to know whether the intercept is: at the left (top) or the right (bottom) of X (parameter matrix) to drop it correctly
                     use 1 if it is on left (top) or the 0 if it is at the right (bottom) of X (parameter matrix)
        - shSize: 
            - optional parameter, if set to shSize="unity" the shocks are scaled to unity, otherwise they are 1 standard deviation
"""
function irf_chol(beta_vec,sigma_vec,n::Integer,p::Integer,const_loc::Integer,n_irf::Integer,IRF_mat;shSize=0.0)

k_nc = n*p;                     # number of perameters with no constant
# reshape the B matrix and drop the constant
B_draw = reshape(beta_vec,k_nc+1,n)'

# if intercept == 1
    Bnc_draw = B_draw[:,1 + const_loc :k_nc + const_loc]
# else 
#     Bnc_draw = B_draw[:,1:k_nc]
# end

# reshape the Sigma matrix
Σ_draw = reshape(sigma_vec,n,n)

F = [Bnc_draw; 1.0I(n*(p-1)) zeros(n*(p-1),n)]; # Companion form

A_chol = cholesky(Σ_draw);                  # structural identification via Cholesky

# Shock size (if nothing is selected it will default to one s.d.)
if shSize != 0.0        
    shocks_mat    = shSize*(1.0./diag(A_chol.L)).*1.0I(n);    
else 
    shocks_mat    = 1.0I(n);           
end
impulse_full = zeros(k_nc,);
resp_all = similar(impulse_full);
responses = zeros(n,n_irf);


# now cycle over the shocks
for i_var = 1:n
    responses[:,1] = A_chol.L*shocks_mat[:,i_var];
    impulse_full[1:n,1] = responses[:,1];

    for i_irf = 2:n_irf
        mul!(resp_all,F^(i_irf-1), impulse_full);
        responses[:,i_irf] = resp_all[1:n,]
    end

    IRF_mat[:,:,i_var] = transpose(responses)

end     
    
return IRF_mat
end # End of irf_chol

"""
    irf_chol_overDraws(store_beta,store_sigma,VARSetup;shSize = "stdev")

    Impulse response calculation over a distribution of parameters for a homoskedastic VAR 
    Calls [irf_chol](@ref) 
    
    Uses the Cholesky decomposition and the companion form.
        - store_beta is a vector that needs to have the parameters for each equation stacked on top of each other
        - const_loc: the function needs to know whether the intercept is: at the left (top) or the right (bottom) of X (parameter matrix) to drop it correctly
                     use 1 if it is on left (top) or the 0 if it is at the right (bottom) of X (parameter matrix)
        - shSize: 
            - optional parameter, if set to shSize="unity" the shocks are scaled to unity, otherwise they are 1 standard deviation
"""
function irf_chol_overDraws(store_beta,store_sigma,VARSetup;shSize=0.0)
    @unpack n,p,const_loc,n_irf,n_save = VARSetup;
    IRF_4d = zeros(n_irf,n,n,n_save);
    IRF_mat = zeros(n_irf,n,n);
    for i_draw = 1:n_save
        beta_vec = store_beta[:,i_draw];
        sigma_vec = store_sigma[:,i_draw];
    
        IRF_4d[:,:,:,i_draw] = irf_chol(beta_vec,sigma_vec,n,p,const_loc,n_irf,IRF_mat,shSize=shSize);
    end

    
    IRF_median = percentile_mat(IRF_4d, 0.5, dims= 4);
    IRF_68_low = percentile_mat(IRF_4d, 0.16, dims= 4);
    IRF_68_high = percentile_mat(IRF_4d, 0.84, dims= 4);
    return IRF_median, IRF_68_low, IRF_68_high, IRF_4d

end


"""
    irf_chol_overDraws_csv(B_store,Σ_store,h_store,n,p,const_loc,n_irf,n_save;shSize = "stdev")

    Impulse response calculation over a distribution of parameters 
    Calls [irf_chol](@ref) 
    
    Uses the Cholesky decomposition and the companion form.
        - beta_vec needs to have the parameters for each equation stacked on top of each other
        - const_loc: the function needs to know whether the intercept is: at the left (top) or the right (bottom) of X (parameter matrix) to drop it correctly
                     use 1 if it is on left (top) or the 0 if it is at the right (bottom) of X (parameter matrix)
        - shSize: 
            - optional parameter, if set to shSize="unity" the shocks are scaled to unity, otherwise they are 1 standard deviation
"""
function irf_chol_overDraws_csv(store_B,store_Σ,store_h,VARSetup;shSize = 0.0)
    @unpack n,p,const_loc,n_irf,n_save = VARSetup
    # n_save = maximum(size(store_B)); # this doesn't work if its a vector
    IRF_4d = zeros(n_irf,n,n,n_save);
    IRF_mat = zeros(n_irf,n,n);
    # h_mean = mean(median(exp.(store_h),dims=2));
    for i_draw = 1:n_save
        # beta_vec = vec(store_B[:,:,i_draw]);
        h_mean = mean((exp.(store_h[:,i_draw])));
        beta_vec = store_B[:,i_draw];
        sigma_vec = h_mean.*vec(store_Σ[:,:,i_draw]);
    
        IRF_4d[:,:,:,i_draw] = irf_chol(beta_vec,sigma_vec,n,p,const_loc,n_irf,IRF_mat,shSize=shSize);
    end

    
    IRF_median = percentile_mat(IRF_4d, 0.5, dims= 4);
    IRF_68_low = percentile_mat(IRF_4d, 0.16, dims= 4);
    IRF_68_high = percentile_mat(IRF_4d, 0.84, dims= 4);
    return IRF_median, IRF_68_low, IRF_68_high, IRF_4d

end
