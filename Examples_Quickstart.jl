using BEAVARs, TimeSeries
using Dates     # these are required only for the example, your data may already have a time-series format

# Chan2020minn
model_type, set_strct, hyp_strct= makeSetup("Chan2020minn",n_burn=20;n_save=50,p=2)
data = TimeArray(DateTime(2020,1,1):Quarter(1):DateTime(2027,4,1),rand(30,3))
data_strct = makeDataSetup(model_type,data)
out_strct = beavar(model_type, set_strct, hyp_strct, data_strct)

# Chan2020iniw
model_type, set_strct, hyp_strct= makeSetup("Chan2020iniw",n_burn=20;n_save=50,p=2)
data = TimeArray(DateTime(2020,1,1):Quarter(1):DateTime(2027,4,1),rand(30,3));
data_strct = BEAVARs.makeDataSetup(model_type,data);
out_strct = beavar(model_type, set_strct, hyp_strct, data_strct);

# Chan2020csv
model_type, set_strct, hyp_strct= makeSetup("Chan2020csv",n_burn=20;n_save=50,p=2)
data = TimeArray(DateTime(2020,1,1):Quarter(1):DateTime(2027,4,1),rand(30,3));
data_strct = BEAVARs.makeDataSetup(model_type,data);
out_strct = beavar(model_type, set_strct, hyp_strct, data_strct);


# Blagov2025
model_type, set_strct, hyp_strct= makeSetup("Blagov2025",n_burn=20;n_save=50,p=2)
dataHF_tab, dataLF_tab, varList = BEAVARs.readSpec("bg_L250911","data/Specifications_mfvar.xlsx");
data_strct = makeDataSetup(model_type,dataHF_tab, dataLF_tab,0)
out_struct = beavar(model_type,set_strct,hyp_strct,data_strct)
# out_struct = beavar(model_type,set_strct,hyp_strct,data_strct);


# BGR2010
model_type, set_strct, hyp_strct= makeSetup("BGR2010",n_burn=20;n_save=50,p=2)
data = TimeArray(DateTime(2020,1,1):Quarter(1):DateTime(2027,4,1),rand(30,3))
data_strct = makeDataSetup(model_type,data)
out_strct = beavar(model_type, set_strct, hyp_strct, data_strct)