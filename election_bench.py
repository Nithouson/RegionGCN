import numpy as np
from torch_geometric.data import Data
from sklearn import linear_model, metrics, model_selection, preprocessing
from torch_geometric.transforms import RandomNodeSplit
import torch
import pandas as pd
from copy import copy
from libpysal import weights
import xgboost as xgb


def RMSE(Y_true, Y_pred):
    return metrics.mean_squared_error(Y_true, Y_pred, squared=False)


def MAE(Y_true, Y_pred):
    return metrics.mean_absolute_error(Y_true, Y_pred)


def Rsquare(Y_true, Y_pred):
    return metrics.r2_score(Y_true, Y_pred)


def standardized_adj(mx):
    degree = np.diag(np.sum(mx, axis=1))
    D_inv = np.linalg.inv(degree)
    return np.dot(D_inv, mx)


seed_st, seed_ed = 24121, 24130
model_type = 'XGB'  # ['LR', 'XGB']

db = pd.read_csv('../data/election/county_attr_recollect.csv', index_col='FID')
nodes = len(db.index.values)

varnames = ['Sexratio', 'Pct1829', 'Pct65', 'PctBlack', 'PctHispanic', 'MedIncome',
            'PctBach', 'Gini', 'PctManuf', 'lnPopden', 'Pct3party', 'Turnout',
            'PctFB', 'PctInsured']
n_var = len(varnames)

rook = weights.Rook.from_shapefile('../data/election/cb_2016_cus_county_500k.shp', idVariable='GEOID')
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
A = w.full()[0]
norm_adj = standardized_adj(A)

x = np.asarray(db[varnames].values)
x_lag = np.matmul(norm_adj, x)
x_arr = np.concatenate((x, x_lag), axis=1)
print(x.shape, x_arr.shape)
x_scaler = preprocessing.StandardScaler()
x_scaled = x_scaler.fit_transform(x_arr)
x = torch.FloatTensor(x_scaled)
y = torch.FloatTensor(db[['pct_dem_dr']].values.tolist())

adj = w.to_adjlist(remove_symmetric=False)
edges_list = adj[['focal', 'neighbor']].values.tolist()
edges = torch.LongTensor(edges_list)
edges_t = torch.t(edges)
data = Data(x=x, edge_index=edges_t, y=y)

for seed in range(seed_st, seed_ed+1):
    torch.manual_seed(seed)  # Data split
    randomsplit = RandomNodeSplit("train_rest", num_val=0.2, num_test=0.2)  # default: 6:2:2
    data = randomsplit(data)
    train_size, val_size, test_size = list(data.train_mask).count(True), \
                                      list(data.val_mask).count(True), list(data.test_mask).count(True)
    # print(data, train_size, val_size, test_size)
    # Data(x=[3108, 28], edge_index=[2, 17482], y=[3108, 1], train_mask=[3108],
    # val_mask=[3108], test_mask=[3108]) 1864 622 622

    X_train, X_test = np.asarray(data.x[data.train_mask]), np.asarray(data.x[data.test_mask])
    Y_train, Y_test = np.asarray(data.y[data.train_mask]), np.asarray(data.y[data.test_mask])
    if model_type == 'LR':  # Linear Regression
        reg = linear_model.LinearRegression()
        reg.fit(X_train, Y_train)
        # print(reg.intercept_, reg.coef_)
        Y_pred = reg.predict(X_test)
        print(f"RMSE: {RMSE(Y_test,Y_pred):.4f} MAE: {MAE(Y_test,Y_pred):.4f} R^2: {Rsquare(Y_test,Y_pred):.4f}")
    elif model_type == 'XGB':  # XGBoost
        model = xgb.XGBRegressor()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        print(f"RMSE: {RMSE(Y_test,Y_pred):.4f} MAE: {MAE(Y_test,Y_pred):.4f} R^2: {Rsquare(Y_test,Y_pred):.4f}")



