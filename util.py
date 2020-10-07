import numpy as np
import torch
from csnorm import CSNorm
from tqdm import tqdm
from norm import norm_function


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

def kept_sketchs_id(norms):
    l= len(norms)
    if l <3: return list(range(l))
    result = set([])
    i = 0
    while i < l:
        result.add(i)
        found = False
        j = i+1
        while j < l:
            if norms[j] > norms[i]:
                i = j -1
                found = True
                break
            if (norms[j] < (norms[i] / 2.0)):
                if j != i+1:
                    result.add(j-1)                    
                i = j - 1
                found = True
                break;
            j+=1
        if not found and i != l-1: 
            result.add(l-1)
            return result
        i+=1
    return result

def get_windowed_id(csvs, wId, size =2):
    ids = np.array([])
    for csv in csvs:
        ids  = np.append(ids, csv.id)
        del csv
    closeIds=np.argsort(abs(ids- wId))[:size]
    print('ids',ids,'closet', closeIds)
    return closeIds 

def get_sketched_norm(normType, streamTr, c,r,device ,wId):
    csvs = []
    norm_fn = norm_function(normType, isTorch=True)
    for i in tqdm(range(len(streamTr))):
        csvs, norms = update_sketchs(i,norm_fn, csvs, streamTr[i], c,r,device)
    closeIds = get_windowed_id(csvs, wId)
    print(norms)
    return norms[closeIds].mean()