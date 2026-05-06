using BEAVARs
# Blagov2025
model_type, set_strct, hyp_strct= makeSetup("CPZ2023",n_burn=20;n_save=50,p=2)
dataHF_tab, dataLF_tab, varList = BEAVARs.readSpec("bg_L250911","data/Specifications_mfvar.xlsx");
data_strct = makeDataSetup(model_type,dataHF_tab, dataLF_tab,0)
out_struct = beavar(model_type,set_strct,hyp_strct,data_strct)

Yfit, Yact = BEAVARs.modelFit(out_struct,set_strct)
plot(Yfit[:,2])
plot!(Yact[:,2])

fcast_strct = BEAVARs.forecast(out_struct,set_strct,data_strct)
BEAVARs.forecast_plot(fcast_strct,plot_fcastOnly=1)

