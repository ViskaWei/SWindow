import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
# import cProfile
from util.util import get_name, get_stream, get_analyze_pd, get_rList,get_cList
from dataset.traffic import test_sniff
from evals.evalNorm import get_estimated_norm
from evals.evalNormCS import get_sketched_norm
DATADIR ='/home/swei20/SymNormSlidingWindows/data' 
PCKSET ='/home/swei20/SymNormSlidingWindows/test/data/packets/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'
TESTSET = '/home/swei20/SymNormSlidingWindows/test/data/packets/test100.pcap'
STREAMPATH = 'traffic'
# path = os.path.join(DATADIR, DATASET)
device = 'cpu'
# outIdx = '_sport_r16'
# outIdx = '_src_final'
outIdx = 'test'
# outIdx = '_rd_r12'
import torch

torch.random.manual_seed(42)

def main():
    outIdx = '_test'

    try: 
        os.mkdir(f'./out{outIdx}/')
        os.mkdir(f'./log{outIdx}/')
    except:
        # print('not creating dir')
        pass
    LOAD, TEST = 1,0
    CSLOOP = (not TEST) and 1
    MLOOP = (not CSLOOP)
    colName = ['errCs','n','m','w','c','r', 'cr', 'ex', 'cs','std','un','errUn']

    if TEST:
        mList ,cList, rList= [100], [10],[2]
        suffix = 'test'
        path = TESTSET
        MLOOP = 1
        cr = np.log2(cList[0]*rList[0])
    else:
        path = PCKSET
        if CSLOOP:
            # mList=[2**13]
            mList=[2**4]
            cList = [2**(int(2) + mm) for mm in range(2)]  
            # cList=[16]
            # cList = [2**(int(5) + mm) for mm in range(6)]  
            rList=[1,2]
            # rList = get_rList(mList[0],delta=0.05, l=1, fac=False,gap=4)
            # print(rList)
            suffix = f'csL_m{mList[-1]}_'
            # colName = ['errCs','n','m','w','c','r','cr', 'ex','cs','std']
        elif MLOOP:
            mList = [2**(int(10) + mm) for mm in range(7)]  
            cList, rList = None, None
            rList =[16]
            cList = [1024]
            suffix = f'mL_'
            if cList is not None:
                if rList is not None:
                    cr = int(np.log2(cList[0]*rList[0]))
                    suffix = suffix + f'cr{cr}_'
                else:
                    suffix = suffix + f'c{rList[0]}_'
        else:
            pass
    normK = [8, 16, 4][1]
    normType=['L2',f'T{normK}'][0]
    ftr = ['rd', 'src'][0]
    NAME, logName = get_name(normType, ftr, add=suffix,logdir=f'./log{outIdx}/')
    logging.root.setLevel(logging.DEBUG)
    logging.basicConfig(filename = f'{logName}.log', level=logging.DEBUG)
    n=2**5 if ftr == 'rd' else None
    stream, n = get_stream(ftr=ftr, n=n,m=mList[-1],HH = False, pckPath = path, isLoad = LOAD, isTest=TEST)
    logging.info(f'{normType}-norm of {ftr} Stream {mList[-1]} with dict {n}.')
    results = []
    for m in tqdm(mList):
        m = int(m)    
        stream0 = stream[:m]
        w = int(m*wRate)
        logging.debug(f'stream:{stream0}|w:{w}')
        # w = min(int(m*wRate), wmin+1)
        if rList is None: rList = get_rList(m,delta=0.05, l=2, fac=False,gap=4)
        if cList is None: cList = get_cList(m,rList[0])
        normEx, normUn,errUn = get_estimated_norm(normType, stream0, n, w, sRate=sRate,isUniSampled=MLOOP)
        for r in rList:
        # for r in tqdm(rList):
            for c in cList:
                # if c > m: continue
                csSize = c*r
                # if csSize > 2*w: continue
                cr = int(np.log2(csSize))
                normCsStd = 0
                normCs = get_sketched_norm(normType, stream, w, m, int(c),int(r),device, \
                                                isNearest=True, toNumpy=True)
                errCs = np.round(abs(normEx - normCs)/normEx,3)
                output = [errCs, n,m,w,c,r,cr, normEx,normCs,normCsStd, normUn, errUn]
                logging.info(output)
                results.append(output)

    get_analyze_pd(results, NAME, colName, outDir=f'./out{outIdx}/')



if __name__ == "__main__":
    main()
 