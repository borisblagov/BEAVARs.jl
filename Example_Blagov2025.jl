using BEAVARs
using Parameters
using Plots
using LinearAlgebra, Statistics
# Blagov2025
model_type, set_struct, hyp_struct= makeSetup("CPZ2023",n_burn=50;n_save=10,p=2)
dataHF_tab, dataLF_tab, varList = BEAVARs.readSpec("bgtest","data/Specifications_mfvar.xlsx");
dataHF_tab_short = dataHF_tab[1:end-7]; 
dataLF_tab_short = dataLF_tab[1:end-1];
data_struct = makeDataSetup(model_type,dataHF_tab_short, dataLF_tab_short,0)
out_struct = beavar(model_type,set_struct,hyp_struct,data_struct);

Yfit, Yact = BEAVARs.modelFit(out_struct,set_struct)
# plot(Yfit[:,2])   # this requires plots
# plot!(Yact[:,2])

fcast_struct = BEAVARs.forecast(out_struct,set_struct,data_struct)
BEAVARs.forecast_plot(fcast_struct,plot_fcastOnly=1)


@unpack M_zsp, store_YY, z_vec, Sm_bit, store_YY = out_struct

yy1 = dropdims(median(store_YY,dims=3),dims=3)
yy_low =  percentile_mat(store_YY, 0.05; dims=3);
yy_high =  percentile_mat(store_YY, 0.95; dims=3);
ik=4;
plot(yy1[:,ik]); plot!(yy_low[:,ik]); plot!(yy_high[:,ik])
plot(M_zsp*yy1[Sm_bit'])
plot!(z_vec)

Yfor3D = BEAVARs.forecast(out_struct,set_struct,data_struct);


