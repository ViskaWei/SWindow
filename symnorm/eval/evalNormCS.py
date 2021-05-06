import numpy as np
import torch
import logging
from tqdm import tqdm
from time import time

from symnorm.util.norm import norm_function
from symnorm.eval.csnorm import CSNorm
from symnorm.eval.sketchsUpdate import kept_sketchs_id


def update_norms(csvs, norm_fn, item):
    norms = torch.tensor([], device = "cpu")
    for csv in csvs:
        csv.accumulateVec(item)
        norm = torch.mean(norm_fn(csv.table))
#     # print('item{} | id{} | norm {} | table{}'.format(item, csv.id, csv.norm, csv.table))
        norms = torch.cat((norms, norm.view(1)), 0)   # view(1) for concatenation
        # print('norms', norms)     
    return norms

def get_sketched_norm_pairs(streamTr, r, c, norm_fn):
    csvs = []
    for id, item in enumerate(streamTr):
        csv0 = CSNorm(id, r, c)
        csvs.append(csv0)
        norms = update_norms(csvs, norm_fn, item)
        # print(f'===============csvs: {len(csvs)}============================')
        idxs = kept_sketchs_id(norms)
        csvs = [csvs[i] for i in idxs]
        norms = norms[list(idxs)]
        # logging.debug(f"normsLeft: {norms}")
    ids = np.array([csv.id for csv in csvs])
    return ids, norms




# def get_windowed_id(ids, w, size =2):
#     wId = ids[-1]+1 - w
#     closeIds = np.argsort(abs(ids- wId))[:size]
#     id1,id2 = ids[closeIds]
#     diff = id2-id1
#     c2 = (wId-id1)/(diff)
#     c1 = (id2-wId)/(diff)
#     # print(id1,id2,wId, c1,c2)
#     # print('ids',ids,'closet', closeIds)
#     return closeIds, c1,c2

# def get_averaged_sketched_norm(aveNum, normType, stream, w, m, c, r, isNearest = True, isRatio=False, toNumpy=True):
#     normCsAvg = np.array([])
#     for j in range(aveNum):
#     # for j in tqdm(range(aveNum)):
#         normCs = get_sketched_norm(normType, stream,w, m, int(c),int(r), \
#                                                 isNearest=isNearest, isRatio=isRatio, toNumpy=toNumpy)
#         normCsAvg = np.append(normCsAvg, normCs)
#     normCs = normCsAvg.mean().round(3)
#     normCsStd = normCsAvg.std().round(3)
#     print(normCs, normCsStd)

#     return normCs, normCsStd

# def get_sketched_norm(normType, stream, w, m, c, r, isNearest = False, isRatio=True, toNumpy=True):
#     streamTr = torch.tensor(stream[:m], dtype=torch.int64)
#     assert len(streamTr) == m
#     norm_fn = norm_function(normType, isTorch=True)
#     ids, norms = update_sketchs(streamTr, r, c, norm_fn)
#         # print(time()-t0, len(csvs), norms)
#     closeIds, c1,c2 = get_windowed_id(ids, w)
#     # print(norms, closeIds)
#     logging.debug(norms)
#     norm = norms[closeIds[0]]
#     # if isNearest:
#     #     norm = norms[closeIds[0]]
#     # elif isRatio:
#     #     # print(norms[closeIds])
#     #     norm = c1*norms[closeIds[0]] + c2*norms[closeIds[1]]
#     # else:
#     #     norm = norms[closeIds].mean()
#     if toNumpy: norm = float(norm.cpu().detach().numpy())
#     return norm
