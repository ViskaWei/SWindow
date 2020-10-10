import numpy as np
import logging
from collections import Counter
from util.norm import norm_function

def get_estimated_norm(normType, stream, n, w, sRate):
    norm_fn = norm_function(normType)
    exactNorm = get_exact_norm(norm_fn, stream,n, w)
    uniformNorm =get_uniform_sampled_norm(norm_fn, stream, n, w, sRate=sRate)
    return exactNorm, uniformNorm

def get_freqList(stream, n):
    c = Counter(stream)    
    freqList=[]
    for i in range(1, n+1):
        freqList.append(c[i])
    assert len(freqList) == n
    return freqList

def get_exact_norm(norm_fn, stream, n, w):
    freqList = get_freqList(stream[-w:], n)
    exactNorm = norm_fn(freqList) 
    logging.info('exact Norm of windowed stream {:0.2f}'.format(exactNorm))
    return exactNorm

def get_uniform_sampled_norm(norm_fn, stream, n, w, sRate=0.1):
    # np.random.seed(926)
    samples = np.random.choice(stream, size=int(w*sRate))
    freqList = get_freqList(samples, n)
    samplesNorm = norm_fn(freqList)
    # samplesNorm = np.linalg.norm(freqList, ord=2)
    uniformNorm = np.sqrt(samplesNorm**2 / sRate)
    logging.info('uniform Norm of {:0.2f}-sampled windowed stream {:0.2f}'.format(sRate, uniformNorm))
    return uniformNorm