import os
import numpy as np
import logging
from dataset.traffic import get_packet_stream, get_sniffed_stream

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
    if isTest:
        prefix = './test/' 
        if m is not None: suffix = f'_m{m}.txt'
    else:
        prefix = './'
        suffix = '.txt'
    path = prefix + sDir + ftr + suffix
    return path

def get_stream_range(stream, n=None, ftr=None):
    if ftr is None:
        return n if n is not None else max(stream)
    elif ftr[-4:] == 'port':
        return 2**16
    elif (ftr =='src' or ftr == 'dst'):
        return 2**32
    elif ftr == 'len':
        return 1500
    else:
        return max(stream)

        # else:
        # if isRand:
        #     stream = create_random_stream(n,m)
        # else:
        #     if isTest: 
        #         m = 10
        #         # path ='/home/swei20/SymNormSlidingWindows/data/testdata/test100.pcap'
        #         logging.info('test on 10 packets') 
        #     elif m is None:
        #         m = -1               
        #     stream, m = get_packet_stream(path, ftr, m=m)
        # np.savetxt(f'./data/stream/{streamPath}.txt', stream)