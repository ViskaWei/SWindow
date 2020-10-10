import numpy as np
import pandas as pd
import logging
from dataset.randomstream import create_random_stream
from dataset.traffic import get_packet_stream
from eval.evalNorm import get_estimated_norm
from eval.evalNormCS import get_sketched_norm

def get_stream(NAME, n=None,m=None,path = None, isLoad = True, isRand = True, isTest=False):
    if isLoad:
        stream = np.loadtxt(f'./data/stream/{NAME}.txt')
    else:
        if isRand:
            stream = create_random_stream(n,m)
        else:
            if isTest: m = 10
            stream, m = get_packet_stream(path, 'len', m=m)
            n = max(stream)
        np.savetxt(f'./data/stream/{NAME}.txt', stream)
    return stream, m, n

def get_norms(normType, stream, w, m, n, sRate, c,r=None, device=None, isNearest=True):
    exactNorm, uniformNorm = get_estimated_norm(normType, stream, n, w, sRate)
    if r is None: r = int(np.log10(m))
    device='cuda'
    sketchNorm = get_sketched_norm(normType, stream,w, m, c,r,device, isNearest)
    logging.info('exact {:0.2f}, uniform {:0.2f}, sketch{:0.2f}'.format(exactNorm, uniformNorm,sketchNorm))
    logging.info('n{}|m{}|w{}|sRate{}|table_c{}r{}'.format(n,m,w,sRate,c,r))    
    output = [n,m,w,sRate,c,r, exactNorm, uniformNorm,sketchNorm.cpu().numpy()]
    return output


