import sys
import torch
import copy
import pickle
import pandas as pd
import numpy as np
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
from sklearn import preprocessing
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from cycler import cycler
from libpysal import weights
from models_6 import standardized_adj

seed = 243
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers = 4 if sys.platform == 'linux' else 0

ds_prefix = 'uselec'
db = pd.read_csv('../data/election/county_attr_selected.csv', index_col='GEOID')
nodes = len(db.index.values)

varnames = ['PST045214L', 'PST120214', 'AGE135214', 'AGE295214', 'AGE775214', 'SEX255214',
            'RHI125214', 'RHI225214', 'RHI325214', 'RHI425214', 'RHI725214', 'POP645213',
            'EDU635213', 'EDU685213', 'HSG445213', 'INC910213L', 'PVY020213', 'POP060210L']
n_var = len(varnames)

x_arr = np.asarray(db[varnames].values)
x_scaler = preprocessing.StandardScaler()
x_scaled = x_scaler.fit_transform(x_arr)
x = torch.FloatTensor(x_scaled)
y = db['state_abbr'].values  # 'AL', 'CA', 'TX', ...

rook = weights.Rook.from_shapefile('../data/election/cb_2014_cus_county_dropmz.shp', idVariable='GEOID')
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

model = Node2Vec(data.edge_index, embedding_dim=18, walk_length=20, context_size=10,
    walks_per_node=10, num_negative_samples=1, p=1.0, q=1.0, sparse=True).to(device)
# context_size: number of positive nodes used (an l-length walk can produce l-k positive node chains)
# p,q: bias parameters

loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


colors = [hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1]) for i in range(100)]
plt.rcParams["axes.prop_cycle"] = cycler('color', colors)
print(len(list(set(data.y))))
exit()

def plot_points():
    model.eval()
    z = model().cpu().numpy()
    z = TSNE(n_components=2).fit_transform(z)
    y = data.y
    states = list(set(y))
    ncolors = len(states)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    #cm = plt.get_cmap('gist_rainbow')
    #ax.set_prop_cycle(color=[cm(1. * i / ncolors) for i in range(ncolors)])
    for i in range(len(states)):
        state = states[i]
        ax.scatter(z[y == state, 0], z[y == state, 1], s=8)
    plt.axis('off')
    plt.savefig(f"e{epoch}.svg")


for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch % 10 == 0:
        with torch.no_grad():
            plot_points()

with torch.no_grad():
    z = model().cpu().numpy()
    emb_file = open("uselec_emb_new.pkl", "wb")
    pickle.dump(z, emb_file)
    emb_file.close()
