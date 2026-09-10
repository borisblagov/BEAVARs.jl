using BEAVARs, TimeSeries
using Dates     # these are required only for the example, your data may already have a time-series format

# Chan2020minn
model_type, set_struct, hyp_struct= makeSetup("Chan2020minn",n_burn=20;n_save=50,p=2)
data = TimeArray(DateTime(2020,1,1):Quarter(1):DateTime(2027,4,1),rand(30,3))
data_struct = makeDataSetup(model_type,data)
out_struct = beavar(model_type, set_struct, hyp_struct, data_struct)

# Chan2020iniw
model_type, set_struct, hyp_struct= makeSetup("Chan2020iniw",n_burn=20;n_save=50,p=2)
data = TimeArray(DateTime(2020,1,1):Quarter(1):DateTime(2027,4,1),rand(30,3));
data_struct = BEAVARs.makeDataSetup(model_type,data);
out_struct = beavar(model_type, set_struct, hyp_struct, data_struct);

# Chan2020csv
model_type, set_struct, hyp_struct= makeSetup("Chan2020csv",n_burn=20;n_save=50,p=2)
data = TimeArray(DateTime(2020,1,1):Quarter(1):DateTime(2027,4,1),rand(30,3));
data_struct = BEAVARs.makeDataSetup(model_type,data);
out_struct = beavar(model_type, set_struct, hyp_struct, data_struct);


# Blagov2025
model_type, set_struct, hyp_struct= makeSetup("Blagov2025",n_burn=20;n_save=50,p=2)
dataHF_tab, dataLF_tab, varList = BEAVARs.readSpecMF("bg_L250911","data/Specifications_mfvar.xlsx");
data_struct = makeDataSetup(model_type,dataHF_tab, dataLF_tab)
out_struct = beavar(model_type,set_struct,hyp_struct,data_struct)
# out_struct = beavar(model_type,set_struct,hyp_struct,data_struct);


# BGR2010
model_type, set_struct, hyp_struct= makeSetup("BGR2010",n_burn=20;n_save=50,p=2)
data = TimeArray(DateTime(2020,1,1):Quarter(1):DateTime(2027,4,1),rand(30,3))
data_struct = makeDataSetup(model_type,data)
out_struct = beavar(model_type, set_struct, hyp_struct, data_struct)