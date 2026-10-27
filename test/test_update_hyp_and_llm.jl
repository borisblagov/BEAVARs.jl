using Random


data = TimeArray(DateTime(2020,1,1):Quarter(1):DateTime(2020,1,1)+Quarter(T-1),rand(T,n))

model_type, set_struct, hyp_struct= makeSetup("CCK2016csv",n_burn=50;n_save=50,p=4,prior_RW=0,est_hyp=true)
data_struct = makeDataSetup(model_type,data)

@unpack p, n_save, n_burn, prior_RW, est_hyp = set_struct

n_draws  = n_save+n_burn;

YY = TimeSeries.values(data);
Y, X, T, n, sigmaP_ar4, betaOLS_ar4, priorMinn_struct, β_hat_bb_struct, csv_Σt_struct, csv_h_struct = BEAVARs.init_Minn_csv(YY,set_struct,hyp_struct)
vecYt = vec(Y')

BEAVARs.update_Σ_inv_bb_csv!(β_hat_bb_struct.Σ_inv_bb,csv_Σt_struct.Σt_inv,csv_h_struct.expmh,T,n);
BEAVARs.Chan2020_draw_βsur_bb!(priorMinn_struct,β_hat_bb_struct,X,vecYt);
BEAVARs.CCK2016_drawΣtsur!(vecYt,β_hat_bb_struct,csv_Σt_struct,csv_h_struct,n,T);
BEAVARs.csv_draw_h_ρ!(csv_h_struct, n,T,hyp_struct); 

@unpack β_prior, Cβ_vec, Vβ_vec, Vβ_inv, Vβ_inv_vecView, Vinvβ_prior, idx_kappa1, idx_kappa2, idx_kappa3, Ψ_vec = priorMinn_struct
@unpack c_11, c_12, c_21, c_22 = hyp_struct;
@unpack β_draw = β_hat_bb_struct;

sum_S_κ1 = sum( (β_draw[idx_kappa1] .- β_prior[idx_kappa1]).^2 ./(2.0.*Ψ_vec[idx_kappa1].*Cβ_vec[idx_kappa1]))
sum_S_κ2 = sum( (β_draw[idx_kappa2] .- β_prior[idx_kappa2]).^2 ./(2.0.*Ψ_vec[idx_kappa2].*Cβ_vec[idx_kappa2]))

sum_S_κ1_alt = 0.0
for idx in idx_kappa1
    diff = β_draw[idx] - β_prior[idx]
    global sum_S_κ1_alt += diff^2 / (2.0 * Ψ_vec[idx] * Cβ_vec[idx])
end

sum_S_κ2_alt = 0.0
for idx in idx_kappa2
    diff = β_draw[idx] - β_prior[idx]
    global sum_S_κ2_alt += diff^2 / (2.0 * Ψ_vec[idx] * Cβ_vec[idx])
end
@test sum_S_κ1 ≈ sum_S_κ1_alt 
@test sum_S_κ2 ≈ sum_S_κ2_alt

