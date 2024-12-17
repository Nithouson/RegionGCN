import pickle
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances

seed = 23901
runid = 62
n_regions = 5

params_file = open(f"log_reggcn_{seed}_{runid}.pkl", "rb")
params = pickle.load(params_file)
params_file.close()
print(params.keys())

regw = np.asarray(torch.cat((params['conv1.reg_weight'], params['conv2.reg_weight']), 1))
regb = np.asarray(torch.cat((params['conv1.reg_bias'], params['conv2.reg_bias']), 1))
print(regw.shape, regb.shape)

cos_wmat, cos_bmat = cosine_similarity(regw), cosine_similarity(regb)
print(cos_wmat, cos_bmat)
l1_wmat, l1_bmat = manhattan_distances(regw), manhattan_distances(regb)
print(l1_wmat, l1_bmat)
