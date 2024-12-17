from libpysal import weights
import pickle
from models_6 import *

rook = weights.Rook.from_shapefile('../data/election/cb_2016_cus_county_500k.shp', idVariable='GEOID')
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
g = weights_to_graph(w)


enclave_dict = find_cut_enclave(g)
enclaves = set()
for v in enclave_dict.keys():
    enclaves = enclaves.union(set(enclave_dict[v]))
cut_vertex = list(enclave_dict.keys())
for v in cut_vertex:
    if v in enclaves:
        enclave_dict.pop(v)
for v in enclave_dict.keys():
    print(rook.id_order[v], [rook.id_order[e] for e in enclave_dict[v]])
print(len(enclave_dict), len(enclaves))

enc_file = open("uselec_enc_3.pkl", "wb")
pickle.dump(enclave_dict, enc_file)
enc_file.close()

encf = open("uselec_enc_2.pkl", "rb")
encd = pickle.load(encf)
encl = []
for v in encd.keys():
    encl += encd[v]
print(len(encd), len(encl))
