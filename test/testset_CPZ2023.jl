#TODO create a custom dataset for testing
# TODO create a VAR Data for this

model_type, set_struct, hyp_struct= makeSetup("CPZ2023",n_burn=50;n_save=10,p=2)
dataHF_tab, dataLF_tab, varOrder = BEAVARs.readSpecMF("bgtest","data/Specifications_mfvar.xlsx");

aggMix = 0;
data_struct = makeDataSetup(model_type,dataHF_tab, dataLF_tab,aggMix)


@unpack p, nburn,nsave, const_loc, n_fcst = set_struct
ndraws = nsave+nburn;
nmdraws = 10;               # given a draw from the parameters to draw multiple time from the distribution of the missing data for better confidence intervals

fdataHF_tab, z_tab, freq_mix_tp, datesHF, varNamesLF, fvarNames = BEAVARs.CPZ_prep_TimeArrays(dataLF_tab,dataHF_tab,varOrder,aggMix,n_fcst)
YYwNA = values(fdataHF_tab);
YY = deepcopy(YYwNA);
Tf,n = size(YY);

B_draw, structB_draw, Σt_inv, b0 = BEAVARs.initParamMatrices(n,p,const_loc) 

YYt, Y0, longyo, nm, H_B, H_B_CI, strctBdraw_LI, Σ_invsp, Σt_LI, Σp_invsp, Σpt_ind, Xb, cB, cB_b0_LI, Smsp, Sosp, Sm_bit, Gm, Go, GΣ, Kym = BEAVARs.CPZ_initMatrices(YY,structB_draw,b0,Σt_inv,p);

M_zsp, z_vec, T_z, MOiM, MOiz = BEAVARs.CPZ_makeM_inter(z_tab,YYt,Sm_bit,datesHF,varNamesLF,fvarNames,freq_mix_tp,nm,Tf);


fdatesHF = timestamp(fdataHF_tab);
fdatesLF = collect(timestamp(z_tab)[1]:Month(freq_mix_tp[2]):fdatesHF[end]);
M_inter_agg = BEAVARs.CPZ_makeM_inter_agg(fdatesLF,fdatesHF,freq_mix_tp);

@test data_struct.aggMix==0
