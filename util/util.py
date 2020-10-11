import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from dataset.randomstream import create_random_stream
# from dataset.traffic import get_packet_stream
from dataset.dataloader import load_traffic_stream, get_stream_range
from eval.evalNorm import get_estimated_norm
from eval.evalNormCS import get_sketched_norm


def get_stream(NAME, ftr=None, n=None,m=None,path = None, isLoad = True, isRand = False, isTest=False):
    if isLoad:
        stream = load_traffic_stream(ftr,isTest, m)
    else:
        raise 'to do sniff in commandline'
    n = get_stream_range(stream, n=n, ftr=ftr)
    m = len(stream)
    # print(m, n)
    return stream, m, n

def get_norms(normType, stream, w, m, n, sRate, c,r=None, device=None, isNearest=True):
    exactNorm, uniformNorm = get_estimated_norm(normType, stream, n, w, sRate)
    if r is None: r = int(np.log(m/0.05))
    device='cuda'
    sketchNorm = get_sketched_norm(normType, stream,w, m, c,r,device, isNearest)
    sketchNorm = sketchNorm.data.cpu().numpy()
    logging.info('exact {:0.2f}, uniform {:0.2f}, sketch{:0.2f}'.format(exactNorm, uniformNorm,sketchNorm))
    logging.info('n{}|m{}|w{}|sRate{}|table_c{}r{}'.format(n,m,w,sRate,c,r))    
    output = [n,m,w,sRate,c,r, exactNorm, uniformNorm,sketchNorm]
    return output

def get_analyze_pd(outputs, NAME):
    out = pd.DataFrame(data = outputs, columns = ['n','m','w','sRate','c','r', 'exact', 'uniform','sketch'])
    out['sketchErr'] = abs(out['exact'] - out['sketch'])/out['exact']
    out['uniformErr'] = abs(out['exact'] - out['uniform'])/out['exact']
    out.to_csv(f'./out/{NAME}.csv', index = False)

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