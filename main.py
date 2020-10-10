import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import cProfile
from util.util import get_stream, get_norms

# DATASET ='testdata/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'
DATADIR ='/home/swei20/SymNormSlidingWindows/data/' 
DATASET ='testdata/test100.pcap'
path = os.path.join(DATADIR, DATASET)
logging.basicConfig(level=logging.INFO)
device = 'cuda'
normType=['L2','T10'][0]
LOADSTREAM, RANDSTREAM, TEST,  = 0,1,1
EVALNORM = 1
# NAME = 'randMLoop'
NAME = 'test'
# mList = [100, 1000, 10000, 100000, 1000000]
mList= [10, 100]
cList =[100]

def main():
    wRate, sRate = 0.2, 0.2    
    assert wRate < 1
    results = []

    for m in tqdm(mList):
        n = m
        stream, m, n = get_stream(NAME, n=n,m=m,path = path,\
                 isLoad = LOADSTREAM, isRand = RANDSTREAM, isTest=TEST)
        w = int(m*wRate)
        for c in cList:
            output = get_norms(normType, stream, w, m, n, sRate,\
                        c,r=None, device=device, isNearest=True)
            results.append(output)
    out = pd.DataFrame(data = results, columns = ['n','m','w','sRate','c','r', 'exact', 'uniform','sketch'])
    out.to_csv(f'./out/{NAME}.csv')






if __name__ == "__main__":
    main()
 