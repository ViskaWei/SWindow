import numpy as np
import torch
from csnorm import CSNorm

def create_csv(c,r, device):
    csv = CSNorm(c, r, device=device)
    return csv

def update_norm(csv, item):
    csv.accumulateVec(item)
    norm = csv.get_norm()
    return norm

def update_norms(csvs, c,r, device, item):
    csv0 = create_csv(c,r,device)
    csvs.append(csv0)
    norms = torch.tensor([], device = device)
    for csv in csvs:
        norm = update_norm(csv, item)    
        norms = torch.cat((norms, norm.view(1)), 0)        
    return norms

def update_item(csvs, item, c,r,device):
    norms = update_norms(csvs, c,r, device, item)
    idxs = update_sketchs(norms)
#     print(idxs)
    csvsLeft = [csvs[i] for i in idxs]
    del csvs
    normsLeft = norms[list(idxs)]
    print(normsLeft)
    return csvsLeft, normsLeft 

def update_sketchs(norms):
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

def run(streamTr, c,r,device):
    csvs = []
    for i in range(len(streamTr)):
        csvs, norms = update_item(csvs, streamTr[i], c,r,device)
    return norms
