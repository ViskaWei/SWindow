import numpy as np
import logging
from collections import Counter
from util.norm import norm_function

def get_estimated_norm(normType, stream, n, w, sRate=None, getUniform=True, round =True):
    norm_fn = norm_function(normType)
    normEx = get_exact_norm(norm_fn, stream,n, w)
    if round: normEx = np.round(normEx,3)
    if getUniform:
        normUn =get_uniform_sampled_norm(norm_fn, stream, n, w, sRate=sRate)
        errUn = np.round(abs(normEx-normUn)/normUn,3)
        if round: 
            normUn = np.round(normUn,2)
            errUn =  np.round(errUn,2)
    else:
        normUn, errUn = 0, 0
    return normEx, normUn, errUn

def get_freqList(stream, n=None, m=None):
    # print(n)
    c = Counter(stream)   
    if n is not None and m is not None and m>n:
        freqList=[]
        for i in range(1, n+1):
            freqList.append(c[i])
        assert len(freqList) == n
    else:
        freqList = list(c.values())
    # logging.info('Freqlist{}'.format(freqList))
    return freqList


def get_exact_norm(norm_fn, stream, n, w):
    freqList = get_freqList(stream[-w:], n=n)
    normEx = norm_fn(freqList) 
    logging.info('normEx in w{}: {:0.2f}'.format(w,normEx))
    return normEx

def get_uniform_sampled_norm(norm_fn, stream, n, w, sRate=0.1):
    # np.random.seed(926)
    samples = np.random.choice(stream, size=int(w*sRate))
    freqList = get_freqList(samples, n=n)
    samplesNorm = norm_fn(freqList)
    # samplesNorm = np.linalg.norm(freqList, ord=2)
    uniformNorm = np.sqrt(samplesNorm**2 / sRate)
    logging.info('{:0.2f}-sampled uniform Norm in window {:0.2f}'.format(sRate, uniformNorm))
    return uniformNorm