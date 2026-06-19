using BEAVARs, TimeSeries, ThreadSafeDicts, JLD2


# CPZ2023
model_type, set_struct, hyp_struct= makeSetup("CPZ2023",n_burn=5000;n_save=5000,p=6,n_fcst=2)

n_beg = 2;
n_end = 60;
model_cols_list = ["bg$i" for i in n_beg:n_end]
vintages_names_list = ["v$i" for i in n_beg:n_end]
file_path ="data/Specifications_mfvarBG.xlsx";

vint_dict =  BEAVARs.make_vintages_dict_loop(file_path,model_cols_list,vintages_names_list,model_type,set_struct,hyp_struct);

# vint_out_dict, fcast_out_dict = beavars(vint_dict);
vint_out_dict, fcast_out_dict = BEAVARs.beavars_weave(vint_dict);


# load true data
dataHF_true_ftab, dataLF_true_ftab, varOrder_tt = BEAVARs.readSpec("bg0","data/Specifications_mfvarBG.xlsx");
dataLF_true_ftab = dataLF_true_ftab[:,:gdpBG]

fe_tab, fe_mat, pred_lik_mat, list_keys, eval_vint_dict = BEAVARs.beavars_eval(vint_out_dict, vint_dict,dataLF_true_ftab);

