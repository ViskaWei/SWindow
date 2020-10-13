import numpy as np
import torch
from tqdm import tqdm
from util.norm import norm_function
from evals.csnorm import CSNorm
from evals.sketchsUpdate import kept_sketchs_id


# def create_hashes(r):
#     torch.random.manual_seed(42)
#     # rand_state = torch.random.get_rng_state()
#     hashes = torch.randint(0, LARGEPRIME, (r, 6),
#                             dtype=torch.int64, device="cpu")
#     # torch.random.set_rng_state(rand_state) 
#     print('len',len(hashes))   
#     return hashes

def create_csv(id, norm_fn, c,r, device):
    csv = CSNorm(id, norm_fn, c, r, device=device)
    return csv

def update_norm(csv, item):
    csv.accumulateVec(item)
    csv.get_norm()

def update_norms(csvs, c,r, device, item):
    norms = torch.tensor([], device = device)
    for csv in csvs:
        update_norm(csv, item)    
        norms = torch.cat((norms, csv.norm.view(1)), 0)        
    return norms

def update_sketchs(id, norm_fn, csvs, item, c,r,device):
    csv0 = create_csv(id, norm_fn, c,r,device)
    csvs.append(csv0)
    norms = update_norms(csvs, c,r, device, item)
    idxs = kept_sketchs_id(norms)
    csvsLeft = [csvs[i] for i in idxs]
    del csvs
    normsLeft = norms[list(idxs)]
    # print(normsLeft)
    return csvsLeft, normsLeft 

# def kept_sketchs_id(norms):
#     l= len(norms)
#     keep = []
#     i = 0
#     while i < l:
#         keep.append(i)
#         if i == l-2: return keep +[i+1]
#         found = False
#         j = i+1
#         while j < l:
#             if (abs(norms[j]-norms[i]) >= (norms[i] / 2.0)):
#                 i = j if j-1 == i else j-1                
#                 found = True
#                 break
#             j+=1
#         if not found: return keep + [l-1]
#     return keep

def get_windowed_id(csvs, w, size =2):
    ids = np.array([])
    for csv in csvs:
        ids  = np.append(ids, csv.id)
        del csv
    wId = ids[-1] - w
    closeIds=np.argsort(abs(ids- wId))[:size]
    # print('ids',ids,'closet', closeIds)
    return closeIds 


def get_sketched_norm(normType, stream, w, m, c, r, device, isNearest = True, toNumpy=True):
    csvs = []
    streamTr=torch.tensor(stream[:m], dtype=torch.int64)
    assert len(streamTr) == m
    norm_fn = norm_function(normType, isTorch=True)
    # for i in range(m):
    for i in tqdm(range(m)):
        csvs, norms = update_sketchs(i,norm_fn, csvs, streamTr[i], c,r,device)
    closeIds = get_windowed_id(csvs, w)
    # print(norms)
    if isNearest:
        norm = norms[closeIds[0]]
    else:
        norm = norms[closeIds].mean()
    if toNumpy: norm = float(norm.cpu().detach().numpy())
    return norm
