import os, sys, glob
import torch.nn
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import pandas as pd
from libpysal import weights
from sklearn import preprocessing, metrics
import esda
import wandb
import openpyxl
import pickle
import datetime
from models_6 import *
import json

if len(sys.argv) > 1:
    model_type = sys.argv[1]
    seed = int(sys.argv[2])
else:
    model_type = 'gwgcn'
    seed = 24121

np.random.seed(seed)  # Initial Region
torch.manual_seed(seed)  # Data Split, Initial Params
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

total_epochs = 10000
tol_interval = 1000
train_ratio = 0.6

wandb_log = True
if wandb_log:
    wandb.init(project="Election_sys", entity="naapd")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

time_st = datetime.datetime.now()
timestamp = time_st.strftime("%Y%m%d_%H%M")[2:]

ds_prefix = 'uselec'
db = pd.read_csv('../data/election/county_attr_recollect.csv', index_col='FID')
nodes = len(db.index.values)

varnames = ['Sexratio', 'Pct1829', 'Pct65', 'PctBlack', 'PctHispanic', 'MedIncome',
            'PctBach', 'Gini', 'PctManuf', 'lnPopden', 'Pct3party', 'Turnout',
            'PctFB', 'PctInsured']
n_var = len(varnames)

x_arr = np.asarray(db[varnames].values)
if model_type in ['ann-dw', 'srgcn-dw', 'reggcn-dw']:
    emb_files = glob.glob(f"{ds_prefix}_emb*.pkl")
    assert len(emb_files) == 1
    print(emb_files[0])
    emb_file = open(emb_files[0], "rb")
    emb = pickle.load(emb_file)
    x_arr = np.hstack([x_arr, emb])
x_scaler = preprocessing.StandardScaler()
x_scaled = x_scaler.fit_transform(x_arr)
x = torch.FloatTensor(x_scaled)
y = torch.FloatTensor(db[['pct_dem_dr']].values.tolist())/100.0

rook = weights.Rook.from_shapefile('../data/election/cb_2016_cus_county_500k.shp', idVariable='GEOID')
links = copy.copy(rook.neighbors)
links['25001'] += ['25007', '25019']  # Barnstable County, MA
links['25007'] += ['25001', '25019']  # Dukes County, MA
links['25019'] += ['25001', '25007']  # Nantucket County, MA
links['53055'] += ['53057']  # San Juan Islands, WA
links['53057'] += ['53055']  # Skagit County, WA
links_id = {}
for unit in links.keys():
    uid = rook.id2i[unit]
    links_id[uid] = [rook.id2i[nunit] for nunit in links[unit]]
w = weights.W(links_id)
# print(w.n_components, len(w.islands))  # connected

A = w.full()[0]
norm_adj = standardized_adj(A)
A_std = torch.FloatTensor(norm_adj).to(device)

adj = w.to_adjlist(remove_symmetric=False)
edges_list = adj[['focal', 'neighbor']].values.tolist()
edges = torch.LongTensor(edges_list)
edges_t = torch.t(edges)

data = Data(x=x, edge_index=edges_t, y=y).to(device)
val_test_ratio = (1-train_ratio)/2
randomsplit = RandomNodeSplit("train_rest", num_val=val_test_ratio, num_test=val_test_ratio)
data = randomsplit(data)
train_size, val_size, test_size = list(data.train_mask).count(True), \
                                  list(data.val_mask).count(True), list(data.test_mask).count(True)
print(data, train_size, val_size, test_size)
# Data(x=[3108, 14], y=[3108, 1]) 1864 622 622

regions = reg_opt_period = -1
hid = 4 * n_var

if model_type == 'ann':
    lr = 1e-2
    wd = 1e-4
    mx = None
    model = ANN(n_var, 1, hidden=hid, outfunc=F.sigmoid).to(device)
elif model_type == 'ann-dw':
    lr = 3e-3
    wd = 1e-3
    mx = None
    model = ANN(2*n_var, 1, hidden=hid, outfunc=F.sigmoid).to(device)
elif model_type == 'srgcn':
    lr = 1e-3
    wd = 0.3
    mx = A_std
    model = SRGCN(n_var, 1, hidden=hid, outfunc=F.sigmoid).to(device)
elif model_type == 'srgcn-dw':
    lr = 1e-2
    wd = 0.1
    mx = A_std
    model = SRGCN(2*n_var, 1, hidden=hid, outfunc=F.sigmoid).to(device)
elif model_type == 'gwgcn':
    lr = 1e-4
    wd = 0.1
    mx = A_std
    model = GWGCN(n_var, 1, num_vertex=nodes, hidden=hid, outfunc=F.sigmoid).to(device)
elif model_type in ['reggcn', 'reggcn-f', 'reggcn-c']:
    lr = 1e-4
    wd = 0.3
    regions = 50
    reg_opt_period = 10
    tol_interval = 2 * reg_opt_period
    mx = A_std
    region_index = initial_regions(w, regions)
    pkl_files = glob.glob(f"params_{ds_prefix}_srgcn_{seed}_*.pkl")
    assert len(pkl_files) == 1
    print(pkl_files[0])
    params_file = open(pkl_files[0], "rb")
    params = pickle.load(params_file)
    params_file.close()
    model = RegGCN(n_var, 1, num_vertex=nodes, num_regions=regions, w=w, global_params=params,
                   init_regions=region_index, hidden=hid, outfunc=F.sigmoid).to(device)
elif model_type in ['reggcn-dw']:
    lr = 1e-4
    wd = 1e-2
    regions = 30
    reg_opt_period = 10
    tol_interval = 2 * reg_opt_period
    mx = A_std
    region_index = initial_regions(w, regions)
    pkl_files = glob.glob(f"params_{ds_prefix}_srgcn-dw_{seed}_*.pkl")
    assert len(pkl_files) == 1
    print(pkl_files[0])
    params_file = open(pkl_files[0], "rb")
    params = pickle.load(params_file)
    params_file.close()
    model = RegGCN(2*n_var, 1, num_vertex=nodes, num_regions=regions, w=w, global_params=params,
                   init_regions=region_index, hidden=hid, outfunc=F.sigmoid).to(device)
else:
    raise ModuleNotFoundError

if model_type == 'reggcn-c':
    enc_files = glob.glob(f"{ds_prefix}_enc*.pkl")
    assert len(enc_files) == 1
    print(enc_files[0])
    enc_file = open(enc_files[0], "rb")
    enc_dict = pickle.load(enc_file)

print(model)

if wandb_log:
    wandb.config.seed = seed
    wandb.config.model = model_type
    wandb.config.lr = lr
    wandb.config.epochs = total_epochs
    wandb.config.weight_decay = wd
    wandb.config.hidden = hid
    wandb.config.regions = regions
    wandb.config.tr = train_ratio

criterion = F.mse_loss  # Define loss criterion.
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr, weight_decay=wd)


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, mx)  # Perform a single forward pass.
    if model_type in ['ann', 'ann-dw']:
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()  # Derive gradients.optimizer.step()  # Update parameters based on gradients.
    else:
        loss = criterion(out[data.train_mask], data.y[data.train_mask], reduction='none')
        loss.backward(torch.reshape(torch.ones(train_size).to(device), (train_size, 1)))
    optimizer.step()
    return loss.mean().item()


def evaluate(mask):
    model.eval()
    out = model(data.x, mx)
    loss = criterion(out[mask], data.y[mask])
    return loss


def return_results(mask=None):
    model.eval()
    out = model(data.x, mx)
    if mask is None:
        return out.cpu().detach().numpy(), data.y.cpu().numpy()
    else:
        return out[mask].cpu().detach().numpy(), data.y[mask].cpu().numpy()


lowest_val_loss = 99999.0
min_val_loss_epoch = -1
best_params = dict()
for epoch in range(1, total_epochs+1):
    loss = train()
    val_MSE = evaluate(data.val_mask)
    if wandb_log:
        wandb.log({"train": loss, "val": val_MSE})
    if val_MSE < lowest_val_loss:
        lowest_val_loss = val_MSE
        min_val_loss_epoch = epoch
        cur_params = dict()
        for k in model.state_dict():
            cur_params[k] = model.state_dict()[k]
        best_params = copy.deepcopy(cur_params)
        val_out, val_tar = return_results(data.val_mask)
        test_out, test_tar = return_results(data.test_mask)
        fullout, fulltar = return_results()
        if model_type in ['reggcn', 'reggcn-f', 'reggcn-c', 'reggcn-dw']:
            reg_label = model.region_index
    if epoch > min_val_loss_epoch + tol_interval:  # early stopping
        break
    if epoch % 100 == 0 or epoch == total_epochs - 1:
        print(f"epoch:{epoch:05d} train_loss:{loss:.4f} val_loss:{val_MSE:.4f}")
    if model_type in ['reggcn', 'reggcn-c', 'reggcn-dw'] and epoch % reg_opt_period == 0:  # update regions
        print(f"epoch:{epoch:05d} before train_loss:{loss:.4f} val_loss:{val_MSE:.4f} "
              f"regions:{[list(model.region_index).count(i) for i in range(regions)]}")
        if model_type in ['reggcn', 'reggcn-dw']:
            model.update_regions(data.x, mx, data.y[data.train_mask], data.train_mask, criterion)
        else:
            model.update_connected_regions(data.x, mx, data.y[data.train_mask], data.train_mask,
                                           criterion, enc_dict)
        train_MSE_new, val_MSE_new = evaluate(data.train_mask), evaluate(data.val_mask)
        print(f"epoch:{epoch:05d} after train_loss:{train_MSE_new:.4f} val_loss:{val_MSE_new:.4f} "
              f"regions:{[list(model.region_index).count(i) for i in range(regions)]}")

params_file = open(f"params_{ds_prefix}_{model_type}_{seed}_{timestamp}.pkl", "wb")
pickle.dump(best_params, params_file)
params_file.close()

print(f"Lowest val MSE: {lowest_val_loss} at epoch {min_val_loss_epoch}")
val_rmse = metrics.mean_squared_error(val_tar, val_out, squared=False)
val_mae = metrics.mean_absolute_error(val_tar, val_out)
val_SSE = np.sum((val_tar-val_out)**2)
val_SST = np.sum((val_tar-val_tar.mean())**2)
val_Rsq = 1-val_SSE/val_SST
print(f"val RMSE: {100*val_rmse:.4f} MAE: {100*val_mae:.4f} R^2: {val_Rsq:.4f}")
test_rmse = metrics.mean_squared_error(test_tar, test_out, squared=False)
test_mae = metrics.mean_absolute_error(test_tar, test_out)
test_SSE = np.sum((test_tar-test_out)**2)
test_SST = np.sum((test_tar-test_tar.mean())**2)
test_Rsq = 1-test_SSE/test_SST
print(f"test RMSE: {100*test_rmse:.4f} MAE: {100*test_mae:.4f} R^2: {test_Rsq:.4f}")

log = openpyxl.Workbook()
ws = log.active
ws.append(["No", "GEOID", "pctdem", "splitflag", "region"])
for i in range(len(data.train_mask)):
    id = rook.id_order[i]
    if data.train_mask[i]:
        split_flag = 0
    elif data.val_mask[i]:
        split_flag = 1
    elif data.test_mask[i]:
        split_flag = 2
    else:
        raise ValueError
    if model_type in ['reggcn', 'reggcn-f','reggcn-c','reggcn-dw']:
        ws.append([i, str(id), fullout[i][0], split_flag, reg_label[i]])
    else:
        ws.append([i, str(id), fullout[i][0], split_flag])
log.save(f"log_{ds_prefix}_{model_type}_{seed}_{timestamp}.xlsx")

Ierror = esda.moran.Moran(fullout-fulltar, w)
print(f'Errors Morans I: {Ierror.I}  Z_I: {Ierror.z_norm} p-value: {Ierror.p_sim}')

time_ed = datetime.datetime.now()

result_summary = {"val_RMSE": 100*val_rmse, "val_MAE": 100*val_mae, "val_Rsq": val_Rsq,
                "test_RMSE": 100*test_rmse, "test_MAE": 100*test_mae, "test_Rsq": test_Rsq,
                "Moran_I": Ierror.I, "Moran_z": Ierror.z_norm, "Moran_p": Ierror.p_sim,
                "Time_st": time_st.strftime("%Y%m%d_%H%M")[2:], "Time_ed": time_ed.strftime("%Y%m%d_%H%M")[2:],
                "Duration": str(time_ed-time_st)}
outfile = open(f"res_{ds_prefix}_{model_type}_{seed}_{timestamp}.txt", "w")
json.dump(result_summary, outfile)
outfile.close()
