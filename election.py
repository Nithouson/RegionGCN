import os
import sys
import torch.nn
from torch_geometric.data import Data
import pandas as pd
from libpysal import weights
from sklearn import preprocessing, metrics
import esda
import wandb
import openpyxl
from Models_5 import *

seed = 23901
if len(sys.argv) > 1:
    model_type = sys.argv[1]
else:
    model_type = 'ann'
torch.manual_seed(seed)  # Adam
np.random.seed(seed)  # Initial Region
total_epochs = 10000
tol_interval = 1000
wandb_log = True

if wandb_log:
    wandb.init(project="Election_sys", entity="naapd")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

db = pd.read_csv('./election/election_cs.csv',index_col='GEOID')
nodes = len(db.index.values)

varnames = ['PST045214L', 'PST120214', 'AGE135214', 'AGE295214', 'AGE775214', 'SEX255214',
            'RHI125214', 'RHI225214', 'RHI325214', 'RHI425214', 'RHI725214', 'POP645213',
            'EDU635213', 'EDU685213', 'HSG445213', 'INC910213L', 'PVY020213', 'POP060210L']
n_var = len(varnames)

x_arr = np.asarray(db[varnames].values)
x_scaler = preprocessing.StandardScaler()
x_scaled = x_scaler.fit_transform(x_arr)
x = torch.FloatTensor(x_scaled)
y = torch.FloatTensor(db[['pct_dem_dr']].values.tolist())

rook = weights.Rook.from_shapefile('./election/election_c.shp', idVariable='GEOID')
links = copy(rook.neighbors)
links['25001'] += ['25007','25019']  # Barnstable County, MA
links['25007'] += ['25001','25019']  # Dukes County, MA
links['25019'] += ['25001','25007']  # Nantucket County, MA
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
split = np.asarray(db['split'].values)
data.train_mask = [True if split[i] == 0 else False for i in range(len(x_arr))]
data.val_mask = [True if split[i] == 1 else False for i in range(len(x_arr))]
data.test_mask = [True if split[i] == 2 else False for i in range(len(x_arr))]

train_size, val_size, test_size = list(data.train_mask).count(True), \
                                  list(data.val_mask).count(True), list(data.test_mask).count(True)
print(data, train_size, val_size, test_size)
# Data(x=[3107, 18], y=[3107, 1]) 2175 311 621

regions = -1
hid = 8 * n_var

if model_type == 'ann':
    lr = 3e-4
    wd = 3e-4
    mx = None
    model = ANN(n_var, 1, hidden=hid, outfunc=F.sigmoid).to(device)
elif model_type == 'srgcn':
    lr = 1e-3
    wd = 1e-3
    mx = A_std
    model = SRGCN(n_var, 1, hidden=hid, outfunc=F.sigmoid).to(device)
elif model_type == 'gwgcn':
    lr = 3e-2
    wd = 3e-2
    mx = A_std
    model = GWGCN(n_var, 1, num_vertex=nodes, hidden=hid, outfunc=F.sigmoid).to(device)
elif model_type == 'reggcn':
    lr = 3e-2
    wd = 0.01
    regions = 5
    mx = A_std
    region_index = initial_regions(w, regions)
    model = RegGCN(n_var, 1, num_vertex=nodes, num_regions=regions, w=w,
                   init_regions=region_index, hidden=hid, outfunc=F.sigmoid).to(device)
else:
    raise ModuleNotFoundError
print(model)

if wandb_log:
    wandb.config.seed = seed
    wandb.config.model = model_type
    wandb.config.lr = lr
    wandb.config.epochs = total_epochs
    wandb.config.weight_decay = wd
    wandb.config.hidden = hid
    wandb.config.regions = regions

criterion = F.mse_loss  # Define loss criterion.
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr,weight_decay=wd)


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, mx)  # Perform a single forward pass.
    if model_type in ['ann']:
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

for epoch in range(total_epochs):
    loss = train()
    val_MSE = evaluate(data.val_mask)
    if wandb_log:
        wandb.log({"train": loss, "val": val_MSE})
    if val_MSE < lowest_val_loss:
        lowest_val_loss = val_MSE
        output, target = return_results(data.test_mask)
        fullout, fulltar = return_results()
        if model_type == 'reggcn':
            reg_label = model.region_index
        min_val_loss_epoch = epoch
    if epoch > min_val_loss_epoch + tol_interval:  # early stopping
        break
    if epoch % 100 == 0 or epoch == total_epochs - 1:
        print(f"epoch:{epoch:05d} train_loss:{loss:.4f} val_loss:{val_MSE:.4f}")
    if model_type == 'reggcn' and epoch % 100 == 99:  # update regions
        print(f"before update train_loss:{loss:.4f} val_loss:{val_MSE:.4f} "
              f"regions:{[list(model.region_index).count(i) for i in range(regions)]}")
        model.update_regions(data.x, mx, data.y[data.train_mask], data.train_mask, criterion)
        train_MSE_new, val_MSE_new = evaluate(data.train_mask), evaluate(data.val_mask)
        print(f"after update train_loss:{train_MSE_new:.4f} test_loss:{val_MSE_new:.4f} "
              f"regions:{[list(model.region_index).count(i) for i in range(regions)]}")

rmse = metrics.mean_squared_error(target, output, squared=False)
mae = metrics.mean_absolute_error(target,output)
SSE = np.sum((target-output)**2)
SST = np.sum((target-target.mean())**2)
Rsq = 1-SSE/SST
print(f"Lowest val MSE: {lowest_val_loss} at epoch {min_val_loss_epoch}")
print(f"test RMSE: {rmse} MAE: {mae} R^2: {Rsq}")

log = openpyxl.Workbook()
ws = log.active
for i in range(len(data.train_mask)):
    id = rook.id_order[i]
    if model_type=='reggcn':
        ws.append([i, id, fullout[i][0], reg_label[i]])
    else:
        ws.append([i, id, fullout[i][0]])
log.save(f"log_{model_type}_{seed}.xlsx")

Ierror = esda.moran.Moran(fullout-fulltar, w)
print(f'Errors Morans I: {Ierror.I}  Z_I: {Ierror.z_norm} p-value: {Ierror.p_sim}')
