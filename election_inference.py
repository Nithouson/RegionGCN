import os
import sys
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import pandas as pd
from libpysal import weights
from sklearn import preprocessing, metrics
import wandb
import pickle
from models_6 import *

seed = 23901
if len(sys.argv) > 1:
    model_type = sys.argv[1]
else:
    model_type = 'ann'
torch.manual_seed(seed)  # Data Split
np.random.seed(seed)  # Initial Region
total_epochs = 10000
tol_interval = 1000
wandb_log = False

if wandb_log:
    wandb.init(project="Election_sys", entity="naapd")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

db = pd.read_csv('./election/election_c.csv',index_col='GEOID')
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
links = copy.copy(rook.neighbors)
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
randomsplit = RandomNodeSplit("train_rest", num_val=0.1, num_test=0.2)
data = randomsplit(data)
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
    lr = 3e-3
    wd = 1e-3
    mx = A_std
    model = SRGCN(n_var, 1, hidden=hid, outfunc=F.sigmoid).to(device)
elif model_type == 'gwgcn':
    lr = 3e-2
    wd = 3e-2
    mx = A_std
    model = GWGCN(n_var, 1, num_vertex=nodes, hidden=hid, outfunc=F.sigmoid).to(device)
elif model_type == 'reggcn':
    lr = 3e-3
    wd = 1e-2
    regions = 5
    mx = A_std
    region_index = initial_regions(w, regions)
    model = RegGCN(n_var, 1, num_vertex=nodes, num_regions=regions, w=w,
                   init_regions=region_index, hidden=hid, outfunc=F.sigmoid).to(device)
else:
    raise ModuleNotFoundError
print(model)


def return_results(mask=None):
    model.eval()
    out = model(data.x, mx)
    if mask is None:
        return out.cpu().detach().numpy(), data.y.cpu().numpy()
    else:
        return out[mask].cpu().detach().numpy(), data.y[mask].cpu().numpy()


params_file = open(f"log_{model_type}_{seed}.pkl", "rb")
params = pickle.load(params_file)
params_file.close()
model.load_state_dict(params)
output, target = return_results(data.test_mask)


rmse = metrics.mean_squared_error(target, output, squared=False)
mae = metrics.mean_absolute_error(target, output)
SSE = np.sum((target-output)**2)
SST = np.sum((target-target.mean())**2)
Rsq = 1-SSE/SST
print(f"test RMSE: {100*rmse:.4f} MAE: {100*mae:.4f} R^2: {Rsq:.4f}")
