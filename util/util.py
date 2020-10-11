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

def get_stream(NAME, ftr=None, n=None,m=None,pckPath = None, isLoad = True, isRand = False, isTest=False):
    if isRand:
        stream = create_random_stream(n,m)
    else:
        stream = load_traffic_stream(ftr, isTest, isLoad, m, pckPath)
    n = get_stream_range(stream, n=n, ftr=ftr)
    m = len(stream)
    # print(m, n)
    return stream, m, n

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

def get_analyze_pd(outputs, outName, colName):
    out = pd.DataFrame(data = outputs, columns = colName)
    print(out)
    out.to_csv(f'./out/{outName}.csv', index = False)


def get_name(isRand, ftr=None,isClosest=None, add=''):
    if isRand:
        name ='rand_'
    elif ftr is not None:
        name = 'trff_' + ftr + '_'
    if isClosest:
        name = name + 'c_'
    now = datetime.now()
    name = name + add + now.strftime("%m%d_%H:%M")
    logName = './log/' + name
    return name, logName