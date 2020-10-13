import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import cProfile
from util.util import get_name, get_stream, get_analyze_pd 
from dataset.traffic import test_sniff
from evals.evalNorm import get_estimated_norm
from evals.evalNormCS import get_sketched_norm
DATADIR ='/home/swei20/SymNormSlidingWindows/data' 
PCKSET ='/home/swei20/SymNormSlidingWindows/test/data/packets/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'
TESTSET = '/home/swei20/SymNormSlidingWindows/test/data/packets/test100.pcap'
STREAMPATH = 'traffic'
# path = os.path.join(DATADIR, DATASET)
device = 'cuda'

def main():
    LOAD, RAND = 1,0
    TEST = 0
    CSLOOP = (not TEST) and 1
    MLOOP = (not CSLOOP)
    if TEST:
        mList ,cList, rList= [100], [10],[2]
        suffix = 'test'
        path = TESTSET
        MLOOP = 1
        cr = np.log2(cList[0]*rList[0])
        colName = ['n','m','w','sRate','c','r', 'cr', 'ex', 'un','cs','errCs','errUn']
    else:
        path = PCKSET
        if CSLOOP:
            mList=[10000]
            # cList = [2**8]
            cList = [2**4, 2**5, 2**6, 2**7, 2**8, 2**9]
            # cList = [2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**13]
            rList = None
            # rList = [4,8,12,16,20]
            suffix = f'csL_m{mList[-1]}_'
            colName = ['n','m','w','c','r','cr', 'ex','cs','errCs']
        elif MLOOP:
            # mList = [100,300]
            mList = [100, 500, 1000, 5000, 10000, 50000]
            # mList = [5000, 10000, 50000, 100000, 500000]        
            cList =[2**4]
            # rList = [2**3]
            rList = None
            if rList is None:
                suffix = f'mL_c{cList[0]}_'
            else:
                cr = np.log2(cList[0]*rList[0])
                suffix = f'mL_t{cr}_'
            colName = ['n','m','w','sRate','c','r', 'cr', 'ex', 'un','cs','errCs','errUn']
        else:
            pass
    if RAND: 
        n = 100
        mList = [1000]
    else:
        n = None
    wRate, w, sRate = 0.1, 50000, 0.1    
    k = 2**5
    normType=['L2',f'T{k}'][0]
    ftr = ['sport', 'src'][0]
    NAME, logName = get_name(RAND, normType, ftr=ftr, add=suffix)
    logging.basicConfig(filename = f'{logName}.log', level=logging.INFO)

    results = []
    stream, m,n = get_stream(NAME, ftr=ftr, n=n,m=mList[-1],pckPath = path,\
                 isLoad = LOAD, isRand = RAND, isTest=TEST)
    logging.info('Stream Prepared. Estimating Norms...')
    for m in tqdm(mList):    
        stream0 = stream[:m]
        w = min(int(m*wRate), 1000)
        normEx, normUn,errUn = get_estimated_norm(normType, stream, n, w, sRate=sRate,getUniform=MLOOP)
        
        for c in tqdm(cList):
            if c > m: continue
            if rList is None: 
                r = int(np.log2(m/0.05))
                sketchSize = c*r
                # if sketchSize > m: continue
                cr = int(np.log2(sketchSize))
                rList=[r]
            for r in rList:
                normCs = get_sketched_norm(normType, stream,w, m, int(c),int(r),device, \
                                                isNearest=False, toNumpy=True)
                errCs = abs(normEx - normCs)/normEx
                if CSLOOP:
                    cr = int(np.log2(c*r))
                    output = [n,m,w,c,r, cr, normEx,normCs, errCs]
                elif MLOOP:
                    output = [n,m,w,sRate,c,r,cr, normEx, normUn,normCs, errCs, errUn]
                logging.info(output)
                results.append(output)

    get_analyze_pd(results, NAME, colName)




if __name__ == "__main__":
    main()
 