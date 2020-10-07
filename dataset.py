import numpy as np
import logging
from collections import Counter
from norm import norm_function

def create_stream(normType, n,m,w, sRate=0.1):
    # assert m > w
    # np.random.seed(42)
    stream = np.random.randint(1,n+1,size=m)
    norm_fn = norm_function(normType)
    exactNorm = exact_counting(norm_fn, stream,n, w)
    uniformNorm =uniform_sampling(norm_fn, stream, n, w, sRate=sRate)
    return stream, exactNorm, uniformNorm

def get_freqList(stream, n):
    c = Counter(stream)    
    freqList=[]
    for i in range(1, n+1):
        freqList.append(c[i])
    assert len(freqList) == n
    return freqList

def exact_counting(norm_fn, stream, n, w):
    freqList = get_freqList(stream[-w:], n)
    exactNorm = norm_fn(freqList) 
    logging.info('exact Norm of windowed stream {:0.2f}'.format(exactNorm))
    return exactNorm

def uniform_sampling(norm_fn, stream, n, w, sRate=0.1):
    # np.random.seed(926)
    samples = np.random.choice(stream, size=int(w*sRate))
    freqList = get_freqList(samples, n)
    samplesNorm = norm_fn(freqList)
    # samplesNorm = np.linalg.norm(freqList, ord=2)
    uniformNorm = np.sqrt(samplesNorm**2 / sRate)
    logging.info('uniform Norm of windowed stream {:0.2f}'.format(uniformNorm))
    return uniformNorm