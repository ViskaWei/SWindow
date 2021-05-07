import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
from collections import Counter
from symnorm.dataset.stream import Stream
from symnorm.eval.uniform import Uniform

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
        parser.add_argument('--ftr', type=str, choices=['tst', 'rd','src', 'sport'], help='test, rd or CAIDA src, sport \n')
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
            self.ftr = 'tst'
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
        if self.ftr != 'tst':
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
        coord = self.run_step_stream()
        self.run_step_eval(coord)
        self.run_step_save()

    def run_step_stream(self):
        strm = Stream(self.mMax, self.n, self.ftr)
        strm.run()
        self.mMax = strm.mMax
        self.n = strm.n
        assert (len(strm.coord) >= self.mList[-1])
        return strm.coord

    def run_step_eval(self, coord):
        uni = Uniform(self.norm_fn, self.sRate, aveNum=self.aveNum, rDigit=self.rDigit)      
        if self.loop == 'sL':
            self.run_step_sL(coord, uni)
        elif self.loop == 'mL':
            self.run_step_mL(coord, uni)

    def run_step_mL(self, stream, uni):
        logging.info('Eval Stream Norms...')
        streamTr = torch.tensor(stream, dtype=torch.int64, device = 'cpu')
        # for m in tqdm(self.mList):
        r = self.rList[0]
        c = self.cList[0]
        for m in self.mList:
            if m > self.mMax: break
            coord = stream[:m]
            # if (self.ftr == 'src') or (self.ftr == 'sport'): 
            #     self.n = 10000
            normEx, US3, UU3 = uni.run(coord, self.n)

            logging.info('sketching')
            ids, norms = get_sketched_norm_pairs(streamTr, r, c, self.norm_fn_tr)

            for wId, wRate in enumerate(self.wRList):
                w = int(m / wRate)
                normCs, errCs = self.eval_sketched_norm(norms, ids, w, normEx)
                logging.info(f'|errCs: {errCs} | errUS: {US3[1]} | errUU: {UU3[1]} | normEx: {normEx} | normCs: {normCs}')
                output = [errCs, normEx, normCs, m, wRate, self.n, r, c, self.rc, *US3, *UU3]
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
            columns = ['errCs','ex', 'cs', 'm','w','n','r','c','rc', \
                        'us','errUS','stdUS','uu','errUU','stdUU']

        dfEval = pd.DataFrame(data = self.eval_mat,columns = columns)
        logging.info(f"\n {dfEval.round(2)}")
        if self.debug:
            self.dfEval = dfEval
        else:
            dfEval.to_csv(f'{self.outDir}{self.name}.csv', index = False)



################################################# GET NORMS ##############################################

    def eval_uniform_norm(self, coord, normEx = None, counter = None):
        uniform = Uniform(self.mList)

    def eval_sketched_norm(self, norms, ids, w, normEx):
        normCs = norms[SymNormPipeline.get_norm_id_from_window(ids, w)]
        normCs = normCs.cpu().detach().numpy()        
        errCs = abs(normEx - normCs) / normEx
        return np.around(normCs, self.rDigit), np.around(errCs, self.rDigit)

    @staticmethod
    def get_norm_id_from_window(ids, w):
        wId = ids[-1]+1 - w
        id = (np.abs(ids - wId)).argmin()
        return id
                    
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



    # def run_step_eval_float(self, stream):
    #     logging.info('Eval Stream Norms...')

    #     streamTr = torch.tensor(stream, dtype=torch.int64)
    #     # for m in tqdm(self.mList):
    #     for m in self.mList:
    #         stream0 = stream[:m]
    #         for r in self.rList:
    #             for c in tqdm(self.cList):
    #                 rc = int(np.log2(c*r))
    #                 logging.info('sketching')
    #                 ids, norms = get_sketched_norm_pairs(streamTr, r, c, self.norm_fn_tr)
    #                 # window_normCs = self.sketch_norm_from_coord(streamTr, r, c)

    #                 for wId, wRate in enumerate(self.wRList):
    #                     w = int(m / wRate)
    #                     stream0 = stream0[-w:]
    #                     normEx, normUS, errUS, stdUS = self.eval_sampled_norm(stream0, normEx = None)
    #                     normCs, errCs = self.eval_sketched_norm(norms, ids, w, normEx)
    #                     logging.info(f'|errCs: {errCs} | errUS: {errUS} | normEx: {normEx} | normCs: {normCs} | normUS: {normUS} | stdUS: {stdUS}')
    #                     output = [errCs, errUS, m, w, c, r, rc, normEx, normCs, normUS, stdUS, self.n]
    #                     self.eval_mat.append(output)




        