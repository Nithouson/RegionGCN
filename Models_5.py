import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import numpy as np
import networkx
from copy import copy


class ANN(nn.Module):
    def __init__(self, d_in, d_out=1, hidden=16, outfunc=None):
        super(ANN, self).__init__()
        self.lin1 = nn.Linear(d_in, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.out_layer = nn.Linear(hidden, d_out)
        self.out_func = outfunc

    def forward(self, h, mx=None):
        h = self.lin1(h)
        h = h.relu()
        h = self.lin2(h)
        h = h.relu()
        h = self.out_layer(h)
        if self.out_func is not None:
            return self.out_func(h)
        return h


class GCN(torch.nn.Module):
    def __init__(self, d_in, d_out=1, hidden=16, outfunc=None):
        super().__init__()
        self.conv1 = GCNConv(d_in, hidden, improved=False)
        self.conv2 = GCNConv(hidden, hidden, improved=False)
        self.out_layer = nn.Linear(hidden, d_out)
        self.out_func = outfunc

    def forward(self, h, edge_index):
        h = self.conv1(h, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.out_layer(h)
        if self.out_func is not None:
            return self.out_func(h)
        return h


class GAT(torch.nn.Module):
    def __init__(self, d_in, d_out=1, hidden=16, heads=1, outfunc=None):
        super().__init__()
        self.conv1 = GATConv(d_in, hidden, heads)
        self.conv2 = GATConv(hidden * heads, hidden, heads)
        self.out_layer = nn.Linear(hidden * heads, d_out)
        self.out_func = outfunc

    def forward(self, h, edge_index):
        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.out_layer(h)
        if self.out_func is not None:
            return self.out_func(h)
        return h


# Fixed by Di Zhu.
def renormalized_laplacian(mx):
    mx_tilde = mx + np.eye(mx.shape[0])
    degree_tilde = np.diag(np.sum(mx_tilde, axis=1))
    D_tilde_inv_sqrt = np.linalg.inv(np.sqrt(degree_tilde))
    return np.dot(D_tilde_inv_sqrt, mx_tilde).dot(D_tilde_inv_sqrt)


def standardized_adj(mx):
    degree = np.diag(np.sum(mx, axis=1))
    D_inv = np.linalg.inv(degree)
    return np.dot(D_inv, mx)


class SRGCNConv(nn.Module):
    def __init__(self, d_in, d_out, use_bias=True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.FloatTensor(d_in, d_out), requires_grad=True)
        self.lag_weight = nn.Parameter(torch.FloatTensor(d_in, d_out), requires_grad=True)
        self.bias = nn.Parameter(torch.FloatTensor(d_out), requires_grad=True) if use_bias else None
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.constant_(self.weight, 1/self.d_in)
        nn.init.constant_(self.lag_weight, 1 / self.d_in)
        if self.use_bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, h, norm_adj):
        ra = torch.mm(h, self.weight)
        rb = torch.mm(norm_adj, torch.mm(h, self.lag_weight))
        out = torch.add(ra, rb)
        if self.use_bias:
            out.add_(self.bias)
        return out


class SRGCN(nn.Module):
    def __init__(self, d_in, d_out, hidden=16, outfunc=None):
        super().__init__()
        self.conv1 = SRGCNConv(d_in, hidden)
        self.conv2 = SRGCNConv(hidden, hidden)
        self.out_layer = nn.Linear(hidden, d_out)
        self.out_func = outfunc

    def forward(self, h, norm_adj):
        h = self.conv1(h, norm_adj)
        h = h.relu()
        h = self.conv2(h, norm_adj)
        h = h.relu()
        h = self.out_layer(h)
        if self.out_func is not None:
            return self.out_func(h)
        return h


class GWGCNConv(nn.Module):
    def __init__(self, d_in, d_out, num_vertex, use_bias=True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.FloatTensor(d_in, d_out), requires_grad=True)
        self.lag_weight = nn.Parameter(torch.FloatTensor(d_in, d_out), requires_grad=True)
        self.gwr_weight = nn.Parameter(torch.FloatTensor(num_vertex, d_in),
                                       requires_grad=True)  # Geographically local parameters
        self.bias = nn.Parameter(torch.FloatTensor(d_out), requires_grad=True) if use_bias else None
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.constant_(self.weight, 1/self.d_in)
        nn.init.constant_(self.lag_weight, 1/self.d_in)
        nn.init.constant_(self.gwr_weight, 1)
        if self.use_bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, h, norm_adj):
        gwh = torch.mul(h, self.gwr_weight)  # use torch.mul to enable element-wise product
        ra = torch.mm(gwh, self.weight)
        rb = torch.mm(norm_adj, torch.mm(gwh, self.lag_weight))
        out = torch.add(ra, rb)
        if self.use_bias:
            out.add_(self.bias)
        return out


class GWGCN(torch.nn.Module):
    def __init__(self, d_in, d_out, num_vertex, hidden=16, outfunc=None):
        super().__init__()
        self.conv1 = GWGCNConv(d_in, hidden, num_vertex)
        self.conv2 = GWGCNConv(hidden, hidden, num_vertex)
        self.out_layer = nn.Linear(hidden, d_out)
        self.out_func = outfunc

    def forward(self, h, norm_adj):
        h = self.conv1(h, norm_adj)
        h = h.relu()
        h = self.conv2(h, norm_adj)
        h = h.relu()
        h = self.out_layer(h)
        if self.out_func is not None:
            return self.out_func(h)
        return h


class RegGCNConv(nn.Module):
    def __init__(self, d_in, d_out, num_vertex, num_regions, use_bias=True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.use_bias = use_bias
        self.n_vertex = num_vertex
        self.n_regions = num_regions
        self.weight = nn.Parameter(torch.FloatTensor(d_in, d_out), requires_grad=True)
        self.lag_weight = nn.Parameter(torch.FloatTensor(d_in, d_out), requires_grad=True)
        self.reg_param = nn.Parameter(torch.FloatTensor(num_regions, d_in),
                                      requires_grad=True)  # regional parameters
        self.bias = nn.Parameter(torch.FloatTensor(d_out), requires_grad=True) if use_bias else None
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.constant_(self.weight, 1/self.d_in)
        nn.init.constant_(self.lag_weight, 1/self.d_in)
        nn.init.constant_(self.reg_param, 1)
        if self.use_bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, h, norm_adj, region_index):
        reg_weight = torch.stack([self.reg_param[region_index[v]] for v in range(self.n_vertex)])
        rwh = torch.mul(h, reg_weight)  # use torch.mul to enable element-wise product
        ra = torch.mm(rwh, self.weight)
        rb = torch.mm(norm_adj, torch.mm(rwh, self.lag_weight))
        out = torch.add(ra, rb)
        if self.use_bias:
            out.add_(self.bias)
        return out


class RegGCN(torch.nn.Module):
    '''
    def __init__(self, d_in, d_out, num_vertex, num_regions, w, init_regions, hidden=16, outfunc=None):
        super().__init__()
        self.conv = RegGCNConv(d_in, hidden, num_vertex, num_regions)
        self.out_layer = RegGCNConv(hidden, d_out, num_vertex, num_regions)
        self.out_func = outfunc
        self.n_vertex = num_vertex
        self.n_regions = num_regions
        self.adjlist = w
        self.region_index = init_regions
        self.border = [self.is_border(vertex_id) for vertex_id in range(self.n_vertex)]

    def forward(self, h, norm_adj, region_index=None):
        if region_index is None:
            region_index = self.region_index
        h = self.conv(h, norm_adj, region_index)
        h = h.relu()
        h = self.out_layer(h, norm_adj, self.region_index)
        if self.out_func is not None:
            return self.out_func(h)
        return h
    '''

    def __init__(self, d_in, d_out, num_vertex, num_regions, w, init_regions, hidden=16, outfunc=None):
        super().__init__()
        self.conv1 = RegGCNConv(d_in, hidden, num_vertex, num_regions)
        self.conv2 = RegGCNConv(hidden, hidden, num_vertex, num_regions)
        self.out_layer = nn.Linear(hidden, d_out)
        self.out_func = outfunc
        self.n_vertex = num_vertex
        self.n_regions = num_regions
        self.adjlist = w
        self.region_index = init_regions
        self.border = [self.is_border(vertex_id) for vertex_id in range(self.n_vertex)]

    def forward(self, h, norm_adj, region_index=None):
        if region_index is None:
            region_index = self.region_index
        h = self.conv1(h, norm_adj, region_index)
        h = h.relu()
        h = self.conv2(h, norm_adj, region_index)
        h = h.relu()
        h = self.out_layer(h)
        if self.out_func is not None:
            return self.out_func(h)
        return h

    def is_border(self, vertex_id):
        regv = self.region_index[vertex_id]
        for neighbor in self.adjlist[vertex_id]:
            if not self.region_index[neighbor] == regv:
                return True
        return False

    def update_regions(self, h, norm_adj, y_label, labeled_mask, criterion):
        self.eval()
        reg_index = copy(self.region_index)
        for v in range(self.n_vertex):
            if not self.border[v]:
                continue
            reg_cand = list(set([self.region_index[v]]+[self.region_index[vn] for vn in self.adjlist[v]]))
            loss_list = []
            for r in reg_cand:
                reg_index[v] = r
                with torch.no_grad():
                    out = self.forward(h, norm_adj, reg_index)
                loss_list.append(criterion(out[labeled_mask],y_label))
            reg_new = reg_cand[loss_list.index(min(loss_list))]
            reg_index[v] = reg_new
            if not reg_new == self.region_index[v]:
                self.region_index[v] = reg_new
                self.border[v] = self.is_border(v)
                for neighbor in self.adjlist[v]:
                    self.border[neighbor] = self.is_border(neighbor)
        return


def region_neighbors(region, w):
    # Get neighboring units for members of a region.
    n_list = []
    for member in region:
        n_list.extend(w[member])
    n_set = list(set(n_list))
    for u in n_set:
        if u in region:
            n_set.remove(u)
    return n_set


def weights_to_graph(w):
    # transform a PySAL W to a networkx graph
    g = networkx.Graph()
    for ego, alters in w.neighbors.items():
        for alter in alters:
            g.add_edge(ego, alter)
    return g


def initial_regions(w, n_regions, stoc_step=False):
    # w: libpysal.weights.W
    units = np.arange(w.n).astype(int)
    # if not connected, each component must have seeds
    g = weights_to_graph(w)
    if not networkx.is_connected(g):
        branches = list(networkx.connected_components(g))
        print([len(br) for br in branches])
        if len(branches) > n_regions:
            raise ValueError("The number of disconnected components exceeds the number of regions.")
        seeds = []
        for br in branches:
            s = np.random.choice(list(br), size=1)
            seeds.append(s[0])
        add = np.random.choice(list(set(units).difference(set(seeds))), size=n_regions - len(branches), replace=False)
        seeds = seeds + list(add)
    else:
        seeds = np.random.choice(units, size=n_regions, replace=False)

    label = np.array([-1] * w.n).astype(int)
    for i, seed in enumerate(seeds):
        label[seed] = i
    to_assign = units[label == -1]

    while to_assign.size > 0:
        for rid in range(n_regions):
            region = units[label == rid]
            neighbors = region_neighbors(region, w)
            neighbors = [j for j in neighbors if j in to_assign]
            if len(neighbors) > 0:
                if stoc_step:
                    u = np.random.choice(neighbors)
                    label[u] = rid
                else:
                    for u in neighbors:
                        label[u] = rid
        prev_size = to_assign.size
        to_assign = units[label == -1]
        assert to_assign.size < prev_size
    return label
