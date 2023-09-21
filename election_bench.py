import numpy as np
from torch_geometric.data import Data
from sklearn import linear_model, metrics, model_selection, preprocessing
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


torch.manual_seed(1234)  # Data split & Adam
np.random.seed(1234)  # Initial Region


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
adj = w.to_adjlist(remove_symmetric=False)
edges_list = adj[['focal', 'neighbor']].values.tolist()
edges = torch.LongTensor(edges_list)
edges_t = torch.t(edges)

data = Data(x=x, edge_index=edges_t, y=y)
split = np.asarray(db['split'].values)
data.train_mask = [True if split[i] == 0 else False for i in range(len(x_arr))]
data.val_mask = [True if split[i] == 1 else False for i in range(len(x_arr))]
data.test_mask = [True if split[i] == 2 else False for i in range(len(x_arr))]
train_size = list(data.train_mask).count(True)
print(data, train_size)
# Data(x=[3107, 18], y=[3107, 1]) 2330

X_train, X_test = np.asarray(data.x[data.train_mask]), np.asarray(data.x[data.test_mask])
Y_train, Y_test = np.asarray(data.y[data.train_mask]), np.asarray(data.y[data.test_mask])


# Linear Regression
reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)
# print(reg.intercept_, reg.coef_)
Y_pred = reg.predict(X_test)
print(f"LR RMSE: {RMSE(Y_test,Y_pred)} MAE:{MAE(Y_test,Y_pred)} R^2: {Rsquare(Y_test,Y_pred)}")

# XGBoost
model = xgb.XGBRegressor()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(f"XGBoost RMSE: {RMSE(Y_test,Y_pred)} MAE:{MAE(Y_test,Y_pred)} R^2: {Rsquare(Y_test,Y_pred)}")



