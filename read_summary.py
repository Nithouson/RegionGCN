import glob
import json

seed_st, seed_ed = 24121, 24130
model_type = 'reggcn-c'
ds_prefix = "uselec"

for seed in range(seed_st, seed_ed+1):
    res_files = glob.glob(f"../log/Election_sys5/{model_type}/res_{ds_prefix}_{model_type}_{seed}_*.txt")
    assert len(res_files) == 1
    res_file = open(res_files[0], "r")
    res_dict = json.load(res_file)
    rmse, mae, rsq = res_dict["test_RMSE"], res_dict["test_MAE"], res_dict["test_Rsq"]
    print(f"{rmse:.4f} {mae:.4f} {rsq:.4f}")
