import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from dataset.randomstream import create_random_stream
# from dataset.traffic import get_packet_stream
from dataset.dataloader import load_traffic_stream, get_stream_range
from evals.evalNorm import get_estimated_norm
from evals.evalNormCS import get_sketched_norm

def get_stream(NAME, ftr=None, n=None,m=None,HH=True, pckPath = None, isLoad = True, isTest=False):
    m = int(m)
    if ftr =='rd':
        stream = create_random_stream(n,m, HH=HH, HH3=None)
    else:
        stream = load_traffic_stream(ftr, isTest, isLoad, m, pckPath)
    n = get_stream_range(stream, n=n, ftr=ftr)
    return stream, n

# def get_norms(normType, stream, w, m, n, sRate, c,r=None, device=None, isNearest=True):
#     device='cuda'
#     logging.info('ex {:0.2f}, un {:0.2f}, cs {:0.2f}'.format(exactNorm, uniformNorm,sketchNorm))
#     logging.info('n{}|m{}|w{}|sRate{}|table_c{}r{}'.format(n,m,w,sRate,c,r))   
#     errCs = abs(exactNorm-sketchNorm)/sketchNorm
#     errUn = abs(exactNorm-uniformNorm)/uniformNorm 
#     cr = np.log2(c*r)
#     output = [n,m,w,sRate,c,r, cr, exactNorm, uniformNorm,sketchNorm, errCs, errUn]
#     logging.info(output)
#     return output

def get_analyze_pd(outputs, outName, colName, outDir='./out/'):
    resultPd = pd.DataFrame(data = outputs, columns = colName)
    print(resultPd)
    resultPd.to_csv(f'{outDir}{outName}.csv', index = False)
    return None


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