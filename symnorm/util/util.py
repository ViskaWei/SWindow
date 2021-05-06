import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from datetime import datetime
from dataset.randomstream import create_random_stream
from dataset.dataloader import load_traffic_stream, get_stream_range
from evals.evalNorm import get_estimated_norm
from evals.evalNormCS import get_sketched_norm, get_averaged_sketched_norm

def get_stream(ftr = None, n = None, m = None, HH = True, pckPath = None, isLoad = True, isTest = False):
    m = int(m)
    if ftr == 'rd':
        stream = create_random_stream(n,m, HH=HH, HH3=None)
    else:
        stream = load_traffic_stream(ftr, isTest, isLoad, m, pckPath)
    # n = get_stream_range(stream, n=n, ftr=ftr)
    n = n or 0
    return stream, n

def get_norms(mList, rList, cList, normType, stream, n, \
                w=None, wRate=0.9,sRate=0.1, device=None, aveNum=None, \
                mode=None, isUniSampled=False):
    results = []
    for m in mList:
    # for m in tqdm(mList):
        # m = int(m)    
        stream0 = stream[:m]
        # if w is None:
        w = int(m*wRate)
        # assert (w- (m//wRate)) < 0.1
        assert w <= m
        # logging.debug(f'stream:{stream0}|w:{w}')
        # w = min(int(m*wRate), wmin+1)
        if rList is None: rList = get_rList(m,delta=0.05, l=2, fac=False,gap=4)
        if cList is None: cList = get_cList(m,rList[0])
        normEx, normUn,errUn,unStd = get_estimated_norm(normType, stream0, n, w, sRate=sRate,isUniSampled=isUniSampled,aveNum=aveNum)
        print(normEx, normUn, errUn)
        for r in rList:
        # for r in tqdm(rList):
            for c in cList:
                cr = int(np.log2(c*r))
                if False:
                    normCs, normStd = get_averaged_sketched_norm(aveNum, normType, stream0, w, m, int(c),int(r),device, \
                                                mode=mode, toNumpy=True)
                    errCs = np.round(abs(normEx - normCs)/normEx,3)
                    print(normCs, normCsStd)
                    normStd= np.round(normStd/normEx,3)
                else:
                    errCs, normCs=0,0
                    normStd = unStd
                output = [aveNum, errCs,normStd, m,w,c,r,cr, normEx,normCs, normUn, errUn, n]
                # logging.info(output)
                # print(output)
                results.append(output)
        print(normUn, errUn, unStd)
    return results


def get_analyze_pd(outputs, outName, colName, outDir='./out/'):
    resultPd = pd.DataFrame(data = outputs, columns = colName)
    # print(resultPd)
    resultPd.to_csv(f'{outDir}{outName}.csv', index = False)
    return resultPd


def get_name(normType, ftr, isClosest=None, add='',logdir='./log/'):
    name = normType + '_'
    if ftr is not None:
        name = name + ftr + '_'
    if isClosest:
        name = name + 'c_'
    now = datetime.now()
    name = name + add + now.strftime("%m%d_%H:%M")
    logName = logdir + name
    return name, logName

def get_rList(m,delta=0.05, l=3, fac =False, gap=4):
    rr = int(np.log10(m/delta))
    rList = [rr]
    i = 0
    while i <= l:
        if fac:
            rrNew = int(rr/2)
        else:
            rrNew = rr - gap
        if rrNew >3:
            rList.append(rrNew)
        else:
            break
        rr = rrNew
        i+=1
    return rList

def get_cList(m,r,epsilon=0.05):
    c2 = [np.floor(np.log(m/r)), np.ceil(np.log(m/r))]
    return [int(2**cc) for cc in c2]