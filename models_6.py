import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from math import sqrt
import numpy as np
import networkx
import copy


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


# accelerated version (Do not call np.inv and np.matmul directly!!!)
def standardized_adj(mx):
    rowsum_inv = (1.0/np.sum(mx, axis=1)).reshape((-1, 1))  # axis=1: row sum
    return np.multiply(rowsum_inv, mx)


class SRGCNConv(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.FloatTensor(d_in, d_out), requires_grad=True)
        self.lag_weight = nn.Parameter(torch.FloatTensor(d_in, d_out), requires_grad=True)
        self.bias = nn.Parameter(torch.FloatTensor(d_out), requires_grad=True)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.uniform_(self.weight, -1/sqrt(self.d_in), 1/sqrt(self.d_in))
        nn.init.uniform_(self.lag_weight, -1/sqrt(self.d_in), 1/sqrt(self.d_in))
        nn.init.uniform_(self.bias, -1/sqrt(self.d_in), 1/sqrt(self.d_in))

    def forward(self, h, norm_adj):
        ra = torch.mm(h, self.weight)
        rb = torch.mm(norm_adj, torch.mm(h, self.lag_weight))
        out = torch.add(ra, rb)
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
    def __init__(self, d_in, d_out, num_vertex):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.FloatTensor(d_in, d_out), requires_grad=True)
        self.lag_weight = nn.Parameter(torch.FloatTensor(d_in, d_out), requires_grad=True)
        self.gwr_weight = nn.Parameter(torch.FloatTensor(num_vertex, d_in),
                                       requires_grad=True)  # Geographically local parameters
        self.bias = nn.Parameter(torch.FloatTensor(d_out), requires_grad=True)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.uniform_(self.weight, -1/sqrt(self.d_in), 1/sqrt(self.d_in))
        nn.init.uniform_(self.lag_weight, -1/sqrt(self.d_in), 1/sqrt(self.d_in))
        nn.init.constant_(self.gwr_weight, 1)
        nn.init.uniform_(self.bias, -1/sqrt(self.d_in), 1/sqrt(self.d_in))

    def forward(self, h, norm_adj):
        gwh = torch.mul(h, self.gwr_weight)  # use torch.mul to enable element-wise product
        ra = torch.mm(gwh, self.weight)
        rb = torch.mm(norm_adj, torch.mm(gwh, self.lag_weight))
        out = torch.add(ra, rb)
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
    def __init__(self, d_in, d_out, num_vertex, num_regions, weight, lag_weight, bias):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_vertex = num_vertex
        self.n_regions = num_regions
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.lag_weight = nn.Parameter(lag_weight, requires_grad=False)
        self.reg_weight = nn.Parameter(torch.FloatTensor(num_regions, d_out),
                                       requires_grad=True)  # regional parameters
        nn.init.constant_(self.reg_weight, 1)
        self.reg_bias = nn.Parameter(torch.stack([bias for v in range(self.n_regions)]), requires_grad=True)

    def forward(self, h, norm_adj, region_index):
        ra = torch.mm(h, self.weight)
        rb = torch.mm(norm_adj, torch.mm(h, self.lag_weight))
        h = torch.add(ra, rb)
        rws = torch.stack([self.reg_weight[region_index[v]] for v in range(self.n_vertex)])
        out = torch.mul(h, rws)  # use torch.mul to enable element-wise product
        rwb = torch.stack([self.reg_bias[region_index[v]] for v in range(self.n_vertex)])
        out.add_(rwb)
        return out


class RegGCN(torch.nn.Module):
    def __init__(self, d_in, d_out, num_vertex, num_regions, w, global_params, init_regions, hidden=16, outfunc=None):
        super().__init__()
        self.conv1 = RegGCNConv(d_in, hidden, num_vertex, num_regions, global_params["conv1.weight"],
                                global_params["conv1.lag_weight"], global_params["conv1.bias"])
        self.conv2 = RegGCNConv(hidden, hidden, num_vertex, num_regions, global_params["conv2.weight"],
                                global_params["conv2.lag_weight"], global_params["conv2.bias"])

        self.out_layer = nn.Linear(hidden, d_out)
        self.out_layer.weight = nn.Parameter(global_params["out_layer.weight"], requires_grad=False)
        self.out_layer.bias = nn.Parameter(global_params["out_layer.bias"], requires_grad=False)

        self.out_func = outfunc
        self.n_vertex = num_vertex
        self.n_regions = num_regions
        self.adjlist = w
        self.adjgraph = weights_to_graph(w)
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
            h = self.out_func(h)
        return h

    def is_border(self, vertex_id):
        regv = self.region_index[vertex_id]
        for neighbor in self.adjlist[vertex_id]:
            if not self.region_index[neighbor] == regv:
                return True
        return False

    def update_regions(self, h, norm_adj, y_label, labeled_mask, criterion):
        self.eval()
        reg_index = copy.copy(self.region_index)
        loop = 1
        need_check = [True for u in range(self.n_vertex)]
        while True:
            moved = checked = 0
            for v in range(self.n_vertex):
                if (not self.border[v]) or (not need_check[v]):
                    continue
                reg_cand = list(set([self.region_index[v]] + [self.region_index[vn] for vn in self.adjlist[v]]))
                reg_cand.sort()
                loss_list = []
                for r in reg_cand:
                    reg_index[v] = r
                    with torch.no_grad():
                        out = self.forward(h, norm_adj, reg_index)
                    loss_list.append(criterion(out[labeled_mask], y_label))
                reg_new = reg_cand[loss_list.index(min(loss_list))]
                reg_index[v] = reg_new
                checked += 1
                need_check[v] = False
                if not reg_new == self.region_index[v]:
                    moved += 1
                    self.region_index[v] = reg_new
                    self.border[v] = self.is_border(v)
                    for neighbor in self.adjlist[v]:
                        need_check[neighbor] = True
                        self.border[neighbor] = self.is_border(neighbor)
            print(f"Round {loop}: {moved} units moved; {checked} units checked; "
                  f"regions:{[list(self.region_index).count(i) for i in range(self.n_regions)]}")
            if moved == 0:
                break
            loop += 1
        return

    def update_connected_regions(self, h, norm_adj, y_label, labeled_mask, criterion, enclave_dict):
        self.eval()
        enclaves = []
        for v in enclave_dict.keys():
            enclaves += enclave_dict[v]

        reg_index = copy.copy(self.region_index)
        loop = 1
        need_check = [True for u in range(self.n_vertex)]
        while True:
            moved = checked = 0
            for v in range(self.n_vertex):
                if (not self.border[v]) or (not need_check[v]) or v in enclaves:
                    continue
                if v in enclave_dict.keys():  # v surrounds one or more enclaves
                    source = [i for i in range(self.n_vertex) if self.region_index[i] == self.region_index[v]]
                    if not contiguity_check([v]+enclave_dict[v], source, g=self.adjgraph):
                        # move the enclaves together with v
                        continue
                    reg_cand = list(set([self.region_index[v]] + [self.region_index[vn] for vn in self.adjlist[v]]))
                    reg_cand.sort()
                    loss_list = []
                    for r in reg_cand:
                        reg_index[v] = r
                        for enc in enclave_dict[v]:
                            reg_index[enc] = r  # move the enclaves together with v
                        with torch.no_grad():
                            out = self.forward(h, norm_adj, reg_index)
                        loss_list.append(criterion(out[labeled_mask], y_label))
                    reg_new = reg_cand[loss_list.index(min(loss_list))]
                    reg_index[v] = reg_new
                    for enc in enclave_dict[v]:
                        reg_index[enc] = reg_new  # move the enclaves together with v
                    checked += 1
                    need_check[v] = False
                    if not reg_new == self.region_index[v]:
                        moved += 1
                        self.region_index[v] = reg_new
                        for enc in enclave_dict[v]:
                            self.region_index[enc] = reg_new  # move the enclaves together with v
                        self.border[v] = self.is_border(v)
                        for neighbor in self.adjlist[v]:
                            need_check[neighbor] = True
                            self.border[neighbor] = self.is_border(neighbor)
                else:
                    source = [i for i in range(self.n_vertex) if self.region_index[i] == self.region_index[v]]
                    if not contiguity_check([v], source, g=self.adjgraph):
                        continue
                    reg_cand = list(set([self.region_index[v]] + [self.region_index[vn] for vn in self.adjlist[v]]))
                    reg_cand.sort()
                    loss_list = []
                    for r in reg_cand:
                        reg_index[v] = r
                        with torch.no_grad():
                            out = self.forward(h, norm_adj, reg_index)
                        loss_list.append(criterion(out[labeled_mask], y_label))
                    reg_new = reg_cand[loss_list.index(min(loss_list))]
                    reg_index[v] = reg_new
                    checked += 1
                    need_check[v] = False
                    if not reg_new == self.region_index[v]:
                        moved += 1
                        self.region_index[v] = reg_new
                        self.border[v] = self.is_border(v)
                        for neighbor in self.adjlist[v]:
                            need_check[neighbor] = True
                            self.border[neighbor] = self.is_border(neighbor)
            print(f"Round {loop}: {moved} units moved; {checked} units checked; "
                  f"regions:{[list(self.region_index).count(i) for i in range(self.n_regions)]}")
            if moved == 0:
                break
            loop += 1
        return


def region_neighbors(region, w):
    # Get neighboring units for members of a region.
    n_list = []
    for member in region:
        n_list.extend(w[member])
    n_set = set(n_list).difference(set(region))
    return list(n_set)


def weights_to_graph(w):
    # transform a PySAL W to a networkx graph
    g = networkx.Graph()
    for ego, alters in w.neighbors.items():
        for alter in alters:
            g.add_edge(ego, alter)
    return g


def contiguity_check(units, from_region, g):
    # check if moving area would break source connectivity
    new_source = [j for j in from_region if not j in units]
    if len(new_source) == 0 or networkx.is_connected(g.subgraph(new_source)):
        return True
    else:
        return False


def find_cut_enclave(g):
    enclave_dict = dict()
    for v in g.nodes:
        v_rem = [j for j in g.nodes if j != v]
        if not networkx.is_connected(g.subgraph(v_rem)):
            enclaves = []
            max_comp_size = max([len(c) for c in networkx.connected_components(g.subgraph(v_rem))])
            comp_list = networkx.connected_components(g.subgraph(v_rem))  # largest first
            for cc in comp_list:
                if len(cc) < max_comp_size:
                    enclaves += cc
            enclave_dict[v] = enclaves
    return enclave_dict


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
