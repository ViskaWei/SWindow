import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
from pipeline.cmdPipeline import CmdPipeline
from dataset.randomstream import create_random_stream
from util.util import get_stream, get_norms, get_analyze_pd, get_rList,get_cList

# DATADIR ='/home/swei20/SymNormSlidingWindows/data' 
# PCKSET ='/home/swei20/SymNormSlidingWindows/test/data/packets/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'
# TESTSET = '/home/swei20/SymNormSlidingWindows/test/data/packets/test100.pcap'
# STREAMPATH = 'traffic'
# # path = os.path.join(DATADIR, DATASET)

import torch
torch.random.manual_seed(42)

class SymNormPipeline(CmdPipeline):
    def __init__(self, logging=True):
        super().__init__()
        self.mList=None
        self.cList=None
        self.rList=None
        self.n = None
        self.loop=None
        self.normType = None
        self.ftr = None
        self.isUniSampled=None
        self.w=None
        self.wRate = None
        self.pdCol = ['ave','errCs','std','m','w','c','r', 'cr', 'ex', 'cs','un','errUn','n']
        self.save={'stream':False}
        self.aveNum=None
        self.mode=None
        self.isPrint=None


    def add_args(self, parser):
        super().add_args(parser)
        # ===========================  LOOP  ================================
        parser.add_argument('--mList', type=int, nargs=3, default=None, help='stream length \n')
        parser.add_argument('--cList', type=int, nargs=3, default=None, help='sketch table column\n')
        parser.add_argument('--rList', type=int, nargs=3, default=None, help='sketch table row\n')
        parser.add_argument('--wRate', type=float, default=None, help='sliding window 1/rate\n')
        parser.add_argument('--aveNum', type=int, default=None, help='averaged Number \n')
        # parser.add_argument('--w', type=int, default=None, help='sliding window size\n')
        # ===========================  LOOP  ================================
        parser.add_argument('--load', action = 'store_true', help='Sniff or load packets\n')
        parser.add_argument('--test', action = 'store_true', help='Test or original size\n')
        parser.add_argument('--saveStream', action = 'store_true', help='Saving stream\n')
        # ===========================  NORM  ================================
        parser.add_argument('--norm', type=str, choices=['L','T'], help='Lp-norm or Topk-norm\n')
        parser.add_argument('--normDim', type=int, help='norm dimension\n')
        parser.add_argument('--ftr', type=str, choices=['rd','src'], help='rd or src \n')
        parser.add_argument('--mode', type=str, choices=['nearest','ratio', 'mean'], help='nearest or interpolated or mean\n')
        parser.add_argument('--print', action='store_true', help='printing results\n')

################################################# PREPARE ##############################################
    def prepare(self):
        super().prepare()
        self.apply_loop_args()
        self.apply_dataset_args()
        self.apply_norm_args()
        self.apply_name_args()

    def apply_loop_args(self):
        self.mList = self.get_loop_from_arg('mList')
        self.cList = self.get_loop_from_arg('cList')
        self.rList = self.get_loop_from_arg('rList')
        # self.w = self.get_arg('w',default = None)
        self.aveNum = self.get_arg('aveNum', default = 1)
        self.wRate = self.get_arg('wRate', default = 4)
        self.loop = self.get_arg('loop')
        if self.loop == 'csL':
            m = self.mList[-1]
            self.isUniSampled = False
            self.name ='_m' + str(m)
        elif self.loop == 'mL':
            self.isUniSampled = True
            self.cr = int(np.log2(self.cList[0]*self.rList[0]))
            self.name ='_cr' + str(self.cr)

    def apply_norm_args(self):
        normStr = self.get_arg('norm')
        normInt = self.get_arg('normDim')
        self.normType=normStr + str(normInt)
        self.mode = self.get_arg('mode')

    def apply_dataset_args(self):
        self.ftr = self.get_arg('ftr')
        if self.ftr == 'rd': self.n = int(2**6)
        self.isTest = self.get_arg('test')
        self.isLoad = self.get_arg('load')


    def apply_name_args(self):
        name = self.ftr + '_' + self.loop + '_' + self.normType + '_' + self.name + '_'
        now = datetime.now()
        self.name = name + now.strftime("%m%d_%H:%M")
        self.isPrint = self.get_arg('print')
        if self.isPrint: print(self.name)
        # self.logName = self.logdir + name
    

################################################# RUN ##############################################

    def run(self):
        super().run()
        stream = self.run_step_stream()
        # stream = list(range(1, self.mList[-1]+1))
        assert (len(stream) >= self.mList[-1])
        results = self.run_step_loop(stream)
        self.run_step_analyze(results)

    def run_step_stream(self):
        stream, self.n =get_stream(ftr=self.ftr, n=self.n,m=self.mList[-1],HH=True,\
             pckPath = None, isLoad = self.isLoad, isTest=self.isTest)
        # print(stream)
        # logging.info(f'{self.normType}-norm of {self.ftr} Stream {self.mList[-1]} with dict {self.n}.')
        return stream

    def run_step_loop(self,stream):
        results = get_norms(self.mList, self.rList, self.cList, self.normType, stream,  self.n,\
                                aveNum=self.aveNum, w=self.w, wRate=self.wRate,sRate=0.1, device=self.device, \
                                mode=self.mode, isUniSampled=self.isUniSampled)
        return results

    def run_step_analyze(self, results):
        resultPd= get_analyze_pd(results, self.name, self.pdCol, outDir=self.outDir)
        if self.isPrint: print(resultPd)

