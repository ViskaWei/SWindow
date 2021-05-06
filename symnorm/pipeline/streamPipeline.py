import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
from src.pipeline.basePipeline import basePipeline
# from dataset.randomstream import create_random_stream
# from util.util import get_stream, get_norms, get_analyze_pd, get_rList,get_cList

# DATADIR ='/home/swei20/SymNormSlidingWindows/data' 
# PCKSET ='/home/swei20/SymNormSlidingWindows/test/data/packets/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'
# TESTSET = '/home/swei20/SymNormSlidingWindows/test/data/packets/test100.pcap'
# STREAMPATH = 'traffic'
# # path = os.path.join(DATADIR, DATASET)

import torch
torch.random.manual_seed(42)

class StreamPipeline(basePipeline):
    def __init__(self, logging=True):
        super().__init__()
        self.mList = None
        self.m = None
        self.n = None
        self.ftr = None
        self.isUniSampled = None
        self.w = None
        self.wRate = None
        self.pdCol = ['ave','errCs','std','m','w','c','r', 'cr', 'ex', 'cs','un','errUn','n']
        self.save = {'stream':False}
        self.aveNum = None
        self.mode = None
        self.isPrint = None


    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument('--mList', type=int, nargs=3, default=None, help='stream length \n')
        parser.add_argument('--load', action = 'store_true', help='Sniff or load packets\n')
        parser.add_argument('--saveStream', action = 'store_true', help='Saving stream\n')
        parser.add_argument('--ftr', type=str, choices=['rd','src'], help='rd or src \n')
        parser.add_argument('--mode', type=str, choices=['nearest','ratio', 'mean'], help='nearest or interpolated or mean\n')
        parser.add_argument('--print', action='store_true', help='printing results\n')

################################################# PREPARE ##############################################
    def prepare(self):
        super().prepare()
        self.apply_dataset_args()

    def apply_dataset_args(self):
        self.mList = self.get_loop_from_arg('mList')
        self.m = int(self.mList[-1])
        self.ftr = self.get_arg('ftr')
        if self.ftr == 'rd': self.n = int(2**6)
        self.isTest = self.get_arg('test')
        self.isLoad = self.get_arg('load')

################################################# RUN ##############################################

    def run(self):
        super().run()
        stream = self.run_step_stream()
        # stream = list(range(1, self.mList[-1]+1))
        assert (len(stream) >= self.mList[-1])

    def run_step_stream(self):
        if self.ftr == 'rd':
            stream = self.create_test_stream()
            # stream = self.create_random_stream(HH = True, HH3=None)
        else:
            # stream = load_traffic_stream(ftr, isTest, isLoad, m, pckPath)
            # n = get_stream_range(stream, n=n, ftr=ftr)
            n = n or 0
            # print(stream)
            # logging.info(f'{self.normType}-norm of {self.ftr} Stream {self.mList[-1]} with dict {self.n}.')
        return stream

    def create_test_stream(self):
        stream = np.append(np.arange(1, 3201), np.ones(70)*3201)
        return stream

    def create_random_stream(self, HH=True, HH3=None):
        np.random.seed(922)
        if HH:
            stream = []
            d = int(np.log2(m))
            assert d < n
            for i in range(1,d):
                [stream.append(i) for j in range(int(2**(d-i-1)))]
            l = len(stream)
            stream = np.array(stream)
            rands = np.random.randint(d+2,high=n+1,size = m-len(stream))
            stream = np.append(stream,rands)
        else: 
            stream = np.random.randint(1,high=n+1,size = m)
            if HH3 is not None:
                stream[:int(m//8)] = HH3[2]
                stream[int(m//8):int(m//4)] = HH3[1]
                stream[int(m//4):] = HH3[0]
        assert len(stream) == m
        np.random.shuffle(stream)
        return stream

    # n , m = 100, 2**5
    # print(create_random_stream(n,m))
