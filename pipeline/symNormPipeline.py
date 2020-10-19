import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from pipeline.cmdPipeline import CmdPipeline

# DATADIR ='/home/swei20/SymNormSlidingWindows/data' 
# PCKSET ='/home/swei20/SymNormSlidingWindows/test/data/packets/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'
# TESTSET = '/home/swei20/SymNormSlidingWindows/test/data/packets/test100.pcap'
# STREAMPATH = 'traffic'
# # path = os.path.join(DATADIR, DATASET)
# device = 'cpu'
# # outIdx = '_sport_r16'
# # outIdx = '_src_final'
# outIdx = 'test'
# # outIdx = '_rd_r12'
# import torch

# torch.random.manual_seed(42)

class SymNormPipeline(CmdPipeline):
    def __init__(self, logging=True):
        super().__init__()
        self.mList=None
        self.cList=None
        self.rList=None

        self.dim=None
        self.smooth=None
        self.cutoff=None
        self.base=None
        self.dtype='uint64'
        self.save={'mat': False, 'mask':False, 'stream':False, 'HHs':False, 'maskId':None}
        self.idx=None
        self.sketchMode='exact'
        
    def add_args(self, parser):
        super().add_args(parser)
        # ===========================  LOAD  ================================
        parser.add_argument('--mList', type=int, nargs=3, default=None, help='stream length \n')
        parser.add_argument('--cList', type=int, nargs=3, default=None, help='sketch table column\n')
        parser.add_argument('--rList', type=int, nargs=3, default=None, help='sketch table row\n')
        
        parser.add_argument('--test', type=bool, help='Test or original size\n')
        parser.add_argument('--saveMat', type=bool, help='Saving mat\n')

        # ===========================  PREPRO  ================================
        parser.add_argument('--cutoff', type=str, default=None, help='Bg cutoff\n')
        parser.add_argument('--base', type=int, default=None, help='Base\n')
        parser.add_argument('--dtype', type=str, default=None, help='dtype\n')

        
    def prepare(self):
        super().prepare()
        # self.apply_dataset_args()
        # self.apply_prepro_args()
        self.apply_loop_args()

    def apply_loop_args(self):
        self.mList = self.get_loop_from_arg('mList')
        self.cList = self.get_loop_from_arg('cList')
        self.rList = self.get_loop_from_arg('rList')



    # outIdx = '_test'

    # try: 
    #     os.mkdir(f'./out{outIdx}/')
    #     os.mkdir(f'./log{outIdx}/')
    # except:
    #     # print('not creating dir')
    #     pass
    # LOAD, TEST = 1,0
    # CSLOOP = (not TEST) and 1
    # MLOOP = (not CSLOOP)
    # colName = ['errCs','n','m','w','c','r', 'cr', 'ex', 'cs','std','un','errUn']

    # if TEST:
    #     mList ,cList, rList= [100], [10],[2]
    #     suffix = 'test'
    #     path = TESTSET
    #     MLOOP = 1
    #     cr = np.log2(cList[0]*rList[0])
    # else:
    #     path = PCKSET
    #     if CSLOOP:
    #         # mList=[2**13]
    #         mList=[2**4]
    #         cList = [2**(int(2) + mm) for mm in range(2)]  
    #         # cList=[16]
    #         # cList = [2**(int(5) + mm) for mm in range(6)]  
    #         rList=[1,2]
    #         # rList = get_rList(mList[0],delta=0.05, l=1, fac=False,gap=4)
    #         # print(rList)
    #         suffix = f'csL_m{mList[-1]}_'
    #         # colName = ['errCs','n','m','w','c','r','cr', 'ex','cs','std']
    #     elif MLOOP:
    #         mList = [2**(int(10) + mm) for mm in range(7)]  
    #         cList, rList = None, None
    #         rList =[16]
    #         cList = [1024]
    #         suffix = f'mL_'
    #         if cList is not None:
    #             if rList is not None:
    #                 cr = int(np.log2(cList[0]*rList[0]))
    #                 suffix = suffix + f'cr{cr}_'
    #             else:
    #                 suffix = suffix + f'c{rList[0]}_'
    #     else:
    #         pass
    # normK = [8, 16, 4][1]
    # normType=['L2',f'T{normK}'][0]
    # ftr = ['rd', 'src'][0]
    # NAME, logName = get_name(normType, ftr, add=suffix,logdir=f'./log{outIdx}/')
    # logging.root.setLevel(logging.DEBUG)
    # logging.basicConfig(filename = f'{logName}.log', level=logging.DEBUG)
    # n=2**5 if ftr == 'rd' else None
    # stream, n = get_stream(NAME, ftr=ftr, n=n,m=mList[-1],HH = False, pckPath = path, isLoad = LOAD, isTest=TEST)
    # logging.info(f'{normType}-norm of {ftr} Stream {mList[-1]} with dict {n}.')
    # results = get_norms(mList, rList, cList, normType, stream, w, m, n,\
    #          wRate=0.9,sRate=0.1, device=None, isNearest=True, MLOOP=MLOOP)

    # get_analyze_pd(results, NAME, colName, outDir=f'./out{outIdx}/')
