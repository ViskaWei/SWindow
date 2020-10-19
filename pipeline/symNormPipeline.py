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
# device = 'cpu'
# # outIdx = '_sport_r16'
# # outIdx = '_src_final'
# outIdx = 'test'
# # outIdx = '_rd_r12'
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
        self.wRate = None
        self.pdCol = ['errCs','n','m','w','c','r', 'cr', 'ex', 'cs','std','un','errUn']
        
        self.save={'stream':False}


    def add_args(self, parser):
        super().add_args(parser)
        # ===========================  LOOP  ================================
        parser.add_argument('--mList', type=int, nargs=3, default=None, help='stream length \n')
        parser.add_argument('--cList', type=int, nargs=3, default=None, help='sketch table column\n')
        parser.add_argument('--rList', type=int, nargs=3, default=None, help='sketch table row\n')
        parser.add_argument('--wRate', type=float, default=None, help='sliding window size\n')
        
        # ===========================  LOOP  ================================
        parser.add_argument('--load', action = 'store_true', help='Sniff or load packets\n')
        parser.add_argument('--test', action = 'store_true', help='Test or original size\n')
        parser.add_argument('--saveStream', action = 'store_true', help='Saving stream\n')

        # ===========================  NORM  ================================
        parser.add_argument('--norm', type=str, choices=['L','T'], help='Lp-norm or Topk-norm\n')
        parser.add_argument('--normDim', type=int, help='norm dimension\n')

        parser.add_argument('--ftr', type=int, choices=['rd','src'], help='rd or src \n')

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
        self.wRate = self.get_arg('wRate',default =0.9)
        self.loop = self.get_arg('loop')
        if self.loop == 'csL':
            m = self.mList[0]
            self.isUniSampled = False
            self.name = self.loop + 'm' + str(m)
        elif self.loop == 'mL':
            self.isUniSampled = True
            self.cr = int(np.log2(self.cList[0]*self.rList[0]))
            self.name = self.loop + 'cr' + str(self.cr)

    def apply_norm_args(self):
        normStr = self.get_arg('norm')
        normInt = self.get_arg('normDim')
        self.normType=normStr + str(normInt)

    def apply_dataset_args(self):
        self.ftr = self.get_arg('ftr')
        if self.ftr == 'rd': self.n = int(2**6)
        self.isTest = self.get_arg('test')
        self.isLoad = self.get_arg('load')


    def apply_name_args(self):
        name = self.ftr + '_' + self.normType + '_' + self.name
        now = datetime.now()
        self.name = name + now.strftime("%m%d_%H:%M")
        # self.logName = self.logdir + name
    

################################################# RUN ##############################################

    def run(self):
        super().run()
        stream = self.run_step_stream()
        results = self.run_step_loop(stream)
        self.run_step_analyze(results)


    def run_step_stream(self):
        stream, self.n =get_stream(ftr=self.ftr, n=self.n,m=self.mList[-1],HH=True,\
             pckPath = None, isLoad = self.isLoad, isTest=self.isTest)
        logging.info(f'{self.normType}-norm of {self.ftr} Stream {self.mList[-1]} with dict {self.n}.')
        return stream

    def run_step_loop(self,stream):
        results = get_norms(self.mList, self.rList, self.cList, self.normType, stream,  self.n,\
             wRate=self.wRate,sRate=0.1, device=self.device, isNearest=True, isUniSampled=self.isUniSampled)
        return results

    def run_step_analyze(self, results):
        get_analyze_pd(results, self.name, self.pdCol, outDir=self.outDir)
