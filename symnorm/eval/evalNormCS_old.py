import numpy as np
import torch
import logging
from tqdm import tqdm
from time import time

from util.norm import norm_function
from evals.csnorm import CSNorm
from evals.sketchsUpdate import kept_sketchs_id

def create_csv(id, norm_fn, c,r, device):
    csv = CSNorm(id, norm_fn, c, r, device=device)
    return csv

def update_norm(csv, item):
    csv.accumulateVec(item)
    csv.get_norm()
    # print('item{} | id{} | norm {} | table{}'.format(item, csv.id, csv.norm, csv.table))

def update_norms(csvs, c,r, device, item):
    norms = torch.tensor([], device = device)
    for csv in csvs:
        update_norm(csv, item)    
        norms = torch.cat((norms, csv.norm.view(1)), 0)   
        # print('norms', norms)     
    return norms

def update_sketchs(id, norm_fn, csvs, item, c,r,device):
    csv0 = create_csv(id, norm_fn, c,r,device)
    csvs.append(csv0)
    norms = update_norms(csvs, c,r, device, item)
    # print(f'===============csvs: {len(csvs)}============================')
    idxs = kept_sketchs_id(norms)
    csvsLeft = [csvs[i] for i in idxs]
    del csvs
    normsLeft = norms[list(idxs)]
    return csvsLeft, normsLeft 


def get_windowed_id(csvs, w, size =2):
    ids = np.array([])
    for csv in csvs:
        ids  = np.append(ids, csv.id)
        del csv
    # print(ids)
    wId = ids[-1]+1 - w
    closeIds=np.argsort(abs(ids- wId))[:size]
    id1,id2 = ids[closeIds]
    diff = id2-id1
    c2 = (wId-id1)/(diff)
    c1 = (id2-wId)/(diff)
    # print(id1,id2,wId, c1,c2)
    # print('ids',ids,'closet', closeIds)
    return closeIds, c1,c2

def get_averaged_sketched_norm(aveNum, normType, stream, w, m, c, r, device, isNearest = True, isRatio=False, toNumpy=True):
    normCsAvg = np.array([])
    for j in range(aveNum):
    # for j in tqdm(range(aveNum)):
        normCs = get_sketched_norm(normType, stream,w, m, int(c),int(r),device, \
                                                isNearest=isNearest, isRatio=isRatio, toNumpy=toNumpy)
        normCsAvg = np.append(normCsAvg, normCs)
    normCs = normCsAvg.mean().round(3)
    normCsStd = normCsAvg.std().round(3)
    print(normCs, normCsStd)

    return normCs, normCsStd

def get_sketched_norm(normType, stream, w, m, c, r, device, isNearest = False, isRatio=True, toNumpy=True):
    streamTr=torch.tensor(stream[:m], dtype=torch.int64)
    assert len(streamTr) == m
    norm_fn = norm_function(normType, isTorch=True)
    csvs = []
    for i in range(m):
    # for i in tqdm(range(m)):
        # t0 = time()
        csvs, norms = update_sketchs(i,norm_fn, csvs, streamTr[i], c,r,device)
        # print(time()-t0, len(csvs), norms)
    closeIds, c1,c2 = get_windowed_id(csvs, w)
    # print(norms, closeIds)
    logging.debug(norms)
    if isNearest:
        norm = norms[closeIds[0]]
    elif isRatio:
        # print(norms[closeIds])
        norm = c1*norms[closeIds[0]] + c2*norms[closeIds[1]]
    else:
        norm = norms[closeIds].mean()
    if toNumpy: norm = float(norm.cpu().detach().numpy())
    return norm
