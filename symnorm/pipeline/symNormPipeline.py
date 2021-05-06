import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
from collections import Counter
from symnorm.util.norm import norm_function
from symnorm.pipeline.basePipeline import BasePipeline
from symnorm.eval.evalNormCS import get_sketched_norm_pairs
from symnorm.dataset.dataloader import load_traffic_stream
# from dataset.randomstream import create_random_stream
# from util.util import get_stream, get_norms, get_analyze_pd, get_rList,get_cList

# DATADIR ='/home/swei20/SymNormSlidingWindows/data' 
# PCKSET ='/home/swei20/SymNormSlidingWindows/test/data/packets/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'
# TESTSET = '/home/swei20/SymNormSlidingWindows/test/data/packets/test100.pcap'
# STREAMPATH = 'traffic'
# # path = os.path.join(DATADIR, DATASET)

import torch
torch.random.manual_seed(42)

class SymNormPipeline(BasePipeline):
    def __init__(self, logging=True):
        super().__init__()
        self.test = None
        self.mList = None
        self.cList = None
        self.rList = None
        self.wRList = None
        self.mMax = None
        self.n = None
        self.loop=None
        self.normType = None
        self.norm_fn = None
        self.norm_fn_tr = None
        self.ftr = None
        self.sRate = 0.1
        self.w=None
        self.save={'stream':False}
        self.aveNum= None
        self.mode=None
        self.eval_mat = []
        self.rDigit = 2


    def add_args(self, parser):
        super().add_args(parser)
        # ===========================  LOOP  ================================
        parser.add_argument('--mList', type=int, nargs=3, default=None, help='stream length \n')
        parser.add_argument('--cList', type=int, nargs=3, default=None, help='sketch table column\n')
        parser.add_argument('--rList', type=int, nargs=3, default=None, help='sketch table row\n')
        parser.add_argument('--wRList', type=int, nargs=3, default=None, help='window size\n')

        parser.add_argument('--wRate', type=float, default=None, help='sliding window 1/rate\n')
        parser.add_argument('--aveNum', type=int, default=None, help='averaged Number \n')
        # parser.add_argument('--w', type=int, default=None, help='sliding window size\n')
        # ===========================  LOOP  ================================
        parser.add_argument('--load', action = 'store_true', help='Sniff or load packets\n')
        parser.add_argument('--saveStream', action = 'store_true', help='Saving stream\n')
        # ===========================  NORM  ================================
        parser.add_argument('--norm', type=str, choices=['L','T'], help='Lp-norm or Topk-norm\n')
        parser.add_argument('--normDim', type=int, help='norm dimension\n')
        parser.add_argument('--ftr', type=str, choices=['rd','src'], help='rd or src \n')
        # parser.add_argument('--mode', type=str, choices=['nearest','ratio', 'mean'], help='nearest or interpolated or mean\n')

################################################# PREPARE ##############################################
    def prepare(self):
        super().prepare()
        self.apply_loop_args()
        self.apply_dataset_args()
        self.apply_norm_args()
        self.apply_name_args()

    def apply_loop_args(self):
        self.test = self.get_arg('test')
        if self.test:
            logging.info('==========TESTING=========')
            self.mList, self.rList, self.cList, self.wRList = [3270], [8], [16], [1, 2]
        else:
            self.mList = self.get_loop_from_arg('mList')
            self.rList = self.get_loop_from_arg('rList')
            self.cList = self.get_loop_from_arg('cList')
            self.wRList = self.get_loop_from_arg('wRList')
        # self.w = self.get_arg('w',default = None)
        self.aveNum = self.get_arg('aveNum', default = self.rList[0])
        self.loop = self.get_arg('loop')
        self.mMax = self.mList[-1]
        if self.loop == 'sL':
            self.name ='_m' + str(self.mMax)
        elif self.loop == 'mL':
            self.rc = int(np.log2(self.cList[0]*self.rList[0]))
            self.name ='_rc' + str(self.rc)

    def apply_norm_args(self):
        normStr = self.get_arg('norm')
        normInt = self.get_arg('normDim')
        self.normType=normStr + str(normInt)
        self.norm_fn = norm_function(self.normType)
        self.norm_fn_tr = norm_function(self.normType, isTorch = True)
        # self.mode = self.get_arg('mode')

    def apply_dataset_args(self):
        self.ftr = self.get_arg('ftr')
        # if self.ftr == 'rd': self.n = int(2**6)
        self.isLoad = self.get_arg('load')


    def apply_name_args(self):
        name = self.ftr + '_' + self.loop + '_' + self.normType + '_' + self.name + '_'
        now = datetime.now()
        self.name = name + now.strftime("%m%d_%H:%M")
        logging.debug(self.name)

################################################# RUN ##############################################

    def run(self):
        super().run()
        stream = self.run_step_stream()
        # stream = list(range(1, self.mList[-1]+1))
        assert (len(stream) >= self.mList[-1])
        if self.debug: self.stream = stream
        self.run_step_eval(stream)
        self.run_step_save()

    def run_step_stream(self):
        logging.info('Creating Coord Update Stream...')
        if self.test:
            stream = self.create_test_stream()
        elif self.ftr == "rd":
            stream = self.create_HH_stream(self.mMax, self.n, shuffle=True)
        elif self.ftr == "src":
            pass
        # logging.info(f'{self.normType}-norm of {self.ftr} Stream {self.mList[-1]} with dict {self.n}.')
        return stream[:self.mMax]

    def run_step_eval_float(self, stream):
        logging.info('Eval Stream Norms...')

        streamTr = torch.tensor(stream, dtype=torch.int64)
        # for m in tqdm(self.mList):
        for m in self.mList:
            stream0 = stream[:m]
            for r in self.rList:
                for c in tqdm(self.cList):
                    rc = int(np.log2(c*r))
                    logging.info('sketching')
                    ids, norms = get_sketched_norm_pairs(streamTr, r, c, self.norm_fn_tr)
                    # window_normCs = self.sketch_norm_from_coord(streamTr, r, c)

                    for wId, wRate in enumerate(self.wRList):
                        w = int(m / wRate)
                        stream0 = stream0[-w:]
                        normEx, normUS, errUS, stdUS = self.eval_sampled_norm(stream0, normEx = None)
                        normCs, errCs = self.eval_sketched_norm(norms, ids, w, normEx)
                        logging.info(f'|errCs: {errCs} | errUS: {errUS} | normEx: {normEx} | normCs: {normCs} | normUS: {normUS} | stdUS: {stdUS}')
                        output = [errCs, errUS, m, w, c, r, rc, normEx, normCs, normUS, stdUS, self.n]
                        self.eval_mat.append(output)

    def run_step_eval(self, stream):
        if self.loop == 'sL':
            self.run_step_sL(stream)
        elif self.loop == 'mL':
            self.run_step_mL(stream)

    def run_step_mL(self, stream):
        logging.info('Eval Stream Norms...')
        streamTr = torch.tensor(stream, dtype=torch.int64, device = 'cpu')
        # for m in tqdm(self.mList):
        r = self.rList[0]
        c = self.cList[0]
        for m in self.mList:
            stream0 = stream[:m]
            normEx, normUS, errUS, stdUS, normUU, errUU, stdUU = self.eval_sampled_norm(stream0, normEx = None)
            logging.info('sketching')
            ids, norms = get_sketched_norm_pairs(streamTr, r, c, self.norm_fn_tr)

            for wId, wRate in enumerate(self.wRList):
                w = int(m / wRate)
                normCs, errCs = self.eval_sketched_norm(norms, ids, w, normEx)
                logging.info(f'|errCs: {errCs} | errUS: {errUS} | errUU: {errUU} | normEx: {normEx} | normCs: {normCs} | normUS: {normUS} | stdUS: {stdUS}')
                output = [errCs, errUS, errUU, m, wRate, normEx, normCs, normUS, normUU, stdUS, stdUU, self.n, r, c, self.rc]
                self.eval_mat.append(output)

    def run_step_sL(self, stream):
        logging.info('Eval Stream Norms...')
        streamTr = torch.tensor(stream, dtype=torch.int64, device = 'cpu')
        # for m in tqdm(self.mList):
        m = self.mList[0]
        stream0 = stream[:m]
        normEx = self.get_norm_from_coord(stream0)
        normEx = np.around(normEx, self.rDigit + 1)
        # normEx, normUS, errUS, stdUS, normUU, errUU, stdUU = self.eval_sampled_norm(stream0, normEx = None)
        logging.info('sketching')
        for r in self.rList:
            for c in tqdm(self.cList):
                rc = int(np.log2(c * r))
                ids, norms = get_sketched_norm_pairs(streamTr, r, c, self.norm_fn_tr)

                for wId, wRate in enumerate(self.wRList):
                    w = int(m / wRate)
                    normCs, errCs = self.eval_sketched_norm(norms, ids, w, normEx)
                    logging.info(f'|errCs: {errCs} | normEx: {normEx} | normCs: {normCs} | rc: {rc}|')
                    output = [errCs, m, wRate, r, c, rc, normEx, normCs]
                    self.eval_mat.append(output)

    def run_step_save(self, out = True):
        if self.loop == 'sL':
            columns = ['errCs','m','w','r', 'c', 'rc', 'ex', 'cs', '']
        elif self.loop == 'mL':
            columns = ['errCs','errUS','errUU','m','w','ex', 'cs','us','uu','stdUS','stdUU','n','r','c','rc']

        dfEval = pd.DataFrame(data = self.eval_mat,columns = columns)
        logging.info(f"\n {dfEval.round(2)}")
        if self.debug:
            self.dfEval = dfEval
        else:
            dfEval.to_csv(f'{self.outDir}{self.name}.csv', index = False)


################################################# Stream ##############################################

    def create_test_stream(self):
        '''
        return: [1,2,3,4,5...,3199, 3200, 3201, 3201, 3201,.... 3201]
        len = 3270 = "1...3200" + 70 x "3201"
        '''
        stream = np.append(np.arange(1, 3201), np.ones(70)*3201)
        self.mMax = len(stream)
        self.n = 80
        self.mList = [self.mMax]
        return stream

    def create_HH_stream(self, m, n = None, shuffle=True):
        '''
        for m = 40, n = None, 
        coord = [  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,   1,   2,   3,
         4,   5,   6,   7,   8,   9,  10,  93,  96,  65,  81, 329, 138,
        61, 102, 145,  62, 180, 108, 312,  78, 345, 101, 400, 400, 400, 400]
        '''
        cut = int(m / 4)
        s1 = np.arange(1, cut + 1)
        if n is None: n = 10 * m
        rd = np.random.randint(cut + 2, high=n + 1, size=m - 2 * cut)
        coord = np.concatenate((s1, s1, rd))
        # create heaviness = 0.1 Heavy Hitters
        nHH = np.min([0.05 * m, np.sqrt(m)])
        coord[-int(nHH):] = n
        self.n = n
        if shuffle: np.random.shuffle(coord)
        return coord

    # def create_HH_stream(self, m, n):
    #     HH  = np.concatenate((np.ones(m//2), np.ones(m//4)*2, np.ones(m//8)*4))
    #     rd = np.random.randint(5,high=n+1,size = m-len(HH))
    #     stream = np.concatenate((HH, rd))
    #     np.random.shuffle(stream)
    #     return stream

################################################# GET NORMS ##############################################
    def get_norm_from_coord(self, coord, out=False):
        counter = Counter(coord)
        freqVec = list(counter.values())
        norm = self.norm_fn(freqVec) 
        if out: return norm, counter
        return norm

    def get_norm_from_sampled_universe(self, counter):
        sampleSize = int(self.n * self.sRate)
        sampled_universe = np.random.choice(np.arange(1, self.n + 1), size=sampleSize, replace = False)
        freqVec = [counter[item] for item in sampled_universe]
        normUU = self.norm_fn(freqVec) 
        return normUU

    def get_norm_from_sampled_coord(self, coord):
        sampleSize = int(len(coord) * self.sRate)
        sampleCoord = np.random.choice(coord, size=sampleSize, replace = False)
        sampleNorm = self.get_norm_from_coord(sampleCoord, out=False)
        norm = sampleNorm / self.sRate 
        return norm

    # def sketch_norm_from_coord(self, streamTr, r, c):
    #     logging.info('sketching')
    #     ids, norms = get_sketched_norm_pairs(streamTr, r, c, self.norm_fn_tr)
    #         # print(time()-t0, len(csvs), norms)
    #     norm_ids = [self.get_norm_id_from_window(ids, w) for w in self.wList]
    #     window_norms = norms[norm_ids]
    #     return window_norms.cpu().detach().numpy()

    def get_norm_id_from_window(self, ids, w):
        wId = ids[-1]+1 - w
        id = (np.abs(ids - wId)).argmin()
        return id
 
    def get_stats_from_norms(self, norms, normEx):
        norm, std = norms.mean(), norms.std()
        err = abs(normEx - norm) / normEx
        std = std / normEx
        return np.around(norm, self.rDigit), \
                np.around(err, self.rDigit), np.around(std, self.rDigit)
################################################# EVAL NORMS ##############################################

    def eval_sampled_norm(self, coord, normEx = None, counter = None):
        if normEx is None or counter is None: 
            normEx, counter = self.get_norm_from_coord(coord, out=True)
        normEx = np.around(normEx, self.rDigit + 1)

        normUSs = np.array([self.get_norm_from_sampled_coord(coord) for i in range(self.aveNum)])
        normUS, errUS, stdUS = self.get_stats_from_norms(normUSs, normEx)

        normUUs = np.array([self.get_norm_from_sampled_universe(counter) for i in range(self.aveNum)])
        normUU, errUU, stdUU = self.get_stats_from_norms(normUUs, normEx)

        return normEx, normUS, errUS, stdUS, normUU, errUU, stdUU

    def eval_sketched_norm(self, norms, ids, w, normEx):
        normCs = norms[self.get_norm_id_from_window(ids, w)]
        normCs = normCs.cpu().detach().numpy()        
        errCs = abs(normEx - normCs) / normEx
        return np.around(normCs, self.rDigit), np.around(errCs, self.rDigit)
                    
    def get_rList(self, m, delta=0.05, l=3, fac =False, gap=4):
        rr = int(np.log10(m/delta))
        rList = [rr]
        i = 0
        while i <= l:
            if fac:
                rrNew = int(rr/2)
            else:
                rrNew = rr - gap
            if rrNew >3:
                rList.append(rrNew)
            else:
                break
            rr = rrNew
            i+=1
        return rList

    def get_cList(self, m, r, epsilon=0.05):
        c2 = [np.floor(np.log(m/r)), np.ceil(np.log(m/r))]
        return [int(2**cc) for cc in c2]


    # def run_step_eval(self, stream):
    #     logging.info('Eval Stream Norms...')

    #     streamTr = torch.tensor(stream, dtype=torch.int64)
    #     # for m in tqdm(self.mList):
    #     if self.mList is None: self.mList = [len(stream)]
    #     for m in self.mList:
    #         stream0 = stream[:m]
    #         if self.rList is None: self.rList = self.get_rList(m, delta=0.05, l=2, fac=False, gap=4)
    #         if self.cList is None: self.cList = self.get_cList(m, self.rList[0])
    #         if self.wRList is None: self.wRList = [1]
    #         normEx, normUS, errUS, stdUS = self.eval_sampled_norm(stream0, normEx = None)

    #         for r in self.rList:
    #             for c in tqdm(self.cList):
    #                 cr = int(np.log2(c*r))
    #                 logging.info('sketching')
    #                 ids, norms = get_sketched_norm_pairs(streamTr, r, c, self.norm_fn_tr)
    #                 # window_normCs = self.sketch_norm_from_coord(streamTr, r, c)

    #                 for wId, wRate in enumerate(self.wRList):
    #                     w = int(m / wRate)
    #                     normCs, errCs = self.eval_sketched_norm(norms, ids, w, normEx)
    #                     logging.info(f'|errCs: {errCs} | errUS: {errUS} | normEx: {normEx} | normCs: {normCs} | normUS: {normUS} | stdUS: {stdUS}')
    #                     output = [errCs, errUS, m, wRate, r, c, cr, normEx, normCs, normUS, stdUS, self.n]
    #                     self.eval_mat.append(output)






        