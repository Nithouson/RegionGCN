import numpy as np
from torch_geometric.data import Data
from sklearn import linear_model, metrics, model_selection, preprocessing
from torch_geometric.transforms import RandomNodeSplit
import torch
import pandas as pd
from copy import copy
from libpysal import weights
import xgboost as xgb
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR


def RMSE(Y_true, Y_pred):
    return metrics.root_mean_squared_error(Y_true, Y_pred)


def MAE(Y_true, Y_pred):
    return metrics.mean_absolute_error(Y_true, Y_pred)


def Rsquare(Y_true, Y_pred):
    return metrics.r2_score(Y_true, Y_pred)


def standardized_adj(mx):
    degree = np.diag(np.sum(mx, axis=1))
    D_inv = np.linalg.inv(degree)
    return np.dot(D_inv, mx)


seed_st, seed_ed = 24121, 24130
model_type = 'GWR'  # ['LR', 'SLX', 'XGB', 'GWR]
validation = False  # validation mode for XGBoost

db = pd.read_csv('county_attr.csv', index_col='FID')
nodes = len(db.index.values)

varnames = ['Sexratio', 'Pct1829', 'Pct65', 'PctBlack', 'PctHispanic', 'MedIncome',
            'PctBach', 'Gini', 'PctManuf', 'lnPopden', 'Pct3party', 'Turnout',
            'PctFB', 'PctInsured']
n_var = len(varnames)

rook = weights.Rook.from_shapefile('cb_2016_cus_county_500k.shp', idVariable='GEOID')
links = copy(rook.neighbors)
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
A = w.full()[0]
norm_adj = standardized_adj(A)
adj = w.to_adjlist(remove_symmetric=False)
edges_list = adj[['focal', 'neighbor']].values.tolist()
edges = torch.LongTensor(edges_list)
edges_t = torch.t(edges)

x = np.asarray(db[varnames].values)
if model_type in ['SLX', 'XGB']:
    # add neighborhood average of the features
    x_lag = np.matmul(norm_adj, x)
    x_arr = np.concatenate((x, x_lag), axis=1)
else:
    x_arr = x
print(x.shape, x_arr.shape)
x_scaler = preprocessing.StandardScaler()
x_scaled = x_scaler.fit_transform(x_arr)
x = torch.FloatTensor(x_scaled)
y = torch.FloatTensor(db[['pct_dem_dr']].values.tolist())

data = Data(x=x, edge_index=edges_t, y=y)

for seed in range(seed_st, seed_ed + 1):
    torch.manual_seed(seed)  # Data split
    randomsplit = RandomNodeSplit("train_rest", num_val=0.2, num_test=0.2)  # default: 6:2:2
    data = randomsplit(data)
    train_size, val_size, test_size = list(data.train_mask).count(True), \
                                      list(data.val_mask).count(True), list(data.test_mask).count(True)

    X_train, Y_train = np.asarray(data.x[data.train_mask]), np.asarray(data.y[data.train_mask])
    X_val, Y_val = np.asarray(data.x[data.val_mask]), np.asarray(data.y[data.val_mask])
    X_test, Y_test = np.asarray(data.x[data.test_mask]), np.asarray(data.y[data.test_mask])
    X_comb, Y_comb = np.concatenate((X_train, X_val), axis=0), np.concatenate((Y_train, Y_val), axis=0)

    if model_type in ['LR', 'SLX']:  # Linear Regression
        reg = linear_model.LinearRegression()
        reg.fit(X_comb, Y_comb) # No parameter to be tuned, use train+valid set for training
        Y_pred = reg.predict(X_test)
        print(f"{RMSE(Y_test, Y_pred):.4f} {MAE(Y_test, Y_pred):.4f} {Rsquare(Y_test, Y_pred):.4f}")
    elif model_type == 'GWR':  # GWR
        coord = np.asarray(db[['x', 'y']].values)
        coord_train = np.asarray([(coord[pid, 0], coord[pid, 1]) for pid in range(nodes)
                      if data.train_mask[pid]])
        coord_val = np.asarray([(coord[pid, 0], coord[pid, 1]) for pid in range(nodes)
                      if data.val_mask[pid]])
        coord_comb = np.concatenate((coord_train, coord_val), axis=0)
        coord_test = np.asarray([(coord[pid, 0], coord[pid, 1]) for pid in range(nodes)
                      if data.test_mask[pid]])
        bw = Sel_BW(coord_comb, Y_comb, X_comb, fixed=False, kernel='bisquare', multi=False).search(criterion='AICc')
        model = GWR(coord_comb, Y_comb, X_comb, bw=bw, fixed=False, kernel='bisquare')
        gwrres = model.predict(coord_test, X_test)
        Y_pred = gwrres.predictions
        print(f"{bw} {RMSE(Y_test, Y_pred):.4f} {MAE(Y_test, Y_pred):.4f} {Rsquare(Y_test, Y_pred):.4f}")
    elif model_type == 'XGB':  # XGBoost
        model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03)  # default: n_est=100, lr=0.3
        model.fit(X_train, Y_train)
        if validation:
            Y_pred = model.predict(X_val)
            rmse, mae, r2 = RMSE(Y_val, Y_pred), MAE(Y_val, Y_pred), Rsquare(Y_val, Y_pred)
        else:
            Y_pred = model.predict(X_test)
            rmse, mae, r2 = RMSE(Y_test, Y_pred), MAE(Y_test, Y_pred), Rsquare(Y_test, Y_pred)
        print(f"{rmse:.4f} {mae:.4f} {r2:.4f}")

