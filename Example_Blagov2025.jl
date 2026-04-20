
# Blagov2025
model_type, set_strct, hyp_strct= makeSetup("Blagov2025",n_burn=20;n_save=50,p=2)
dataHF_tab, dataLF_tab, varList = BEAVARs.readSpec("bg_L250911","data/Specifications_mfvar.xlsx");
data_strct = makeDataSetup(model_type,dataHF_tab, dataLF_tab,0)
out_struct = beavar(model_type,set_strct,hyp_strct,data_strct)
# out_struct = beavar(model_type,set_strct,hyp_strct,data_strct);
