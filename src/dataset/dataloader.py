import numpy as np
from dataset.traffic import get_sniffed_stream


def load_traffic_stream(ftr, isTest, isLoad, m, pckPath):
    if isLoad:
        return preload_traffic_stream(ftr, isTest, m)
    else:
        return get_sniffed_stream(ftr, pckPath, m = m, save=False)
        

def preload_traffic_stream(ftr, isTest, m):
    path = get_traffic_stream_path(ftr, isTest, m=m, sDir='data/stream/traffic_')
    stream = np.loadtxt(path)
    return stream


def get_traffic_stream_path(ftr, isTest, m, sDir='data/stream/traffic_'):
    suffix=''
    if isTest:
        prefix = './test/' 
        if m is not None: suffix = f'_m{m}.txt'
    else:
        prefix = './'
        suffix = '.txt'
    path = prefix + sDir + ftr + suffix
    return path

def get_stream_range(stream, n=None, ftr=None):
    if n is not None: 
        return n
    elif ftr[-4:] == 'port':
        return 2**16
    elif (ftr =='src' or ftr == 'dst'):
        return 2**32
    elif ftr == 'len':
        return 1500
    else:
        return max(stream)