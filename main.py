import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import cProfile
from util.util import get_stream, get_norms, get_analyze_pd, get_name
DATADIR ='/home/swei20/SymNormSlidingWindows/data/' 
DATASET ='testdata/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'
STREAMPATH = 'traffic'
# DATASET ='testdata/test100.pcap'
path = os.path.join(DATADIR, DATASET)
device = 'cuda'
normType=['L2','T10'][0]
LOADSTREAM, RANDSTREAM = 1,0
TEST = 0

# w = min(int(m*wRate), 1000)
# assert wRate < 1

def main():
    ftr = 'sport'
    wRate, w, sRate = 0.1, 10000, 0.1    
    NAME, logName = get_name(RANDSTREAM, ftr=ftr,  add='_csLoop_')
    logging.basicConfig(filename = f'{logName}.log', level=logging.INFO)
    if TEST:
        mList= [100]
        cList =[10]
    else:
        mList=[100]
        # mList = [100, 500, 1000, 5000, 10000, 50000]
        cList = [2**6, 2**7, 2**8, 2**9, 2**10]
        rList = [2**2, 2**3, 2**4]
        # mList = [100000, 500000, 100000, 500000, 974000]        
        # cList =[10]
    results = []
    n=None
    stream, m,n = get_stream(NAME, ftr=ftr, n=n,m=mList[-1],path = path,\
                 isLoad = LOADSTREAM, isRand = RANDSTREAM, isTest=TEST)

    for m in tqdm(mList):    
        stream0 = stream[:m]
        w = min(int(m*wRate), 10000)

        for c in cList:
            for r in rList:
                output = get_norms(normType, stream0, w, m, n, sRate,\
                        int(c),r=int(r), device=device, isNearest=True)
            results.append(output)
        # print(results)

    get_analyze_pd(results, NAME)




if __name__ == "__main__":
    main()
 