import logging
import numpy as np
# import scipy.stats
import pandas as pd
from collections import Counter



class StreamTrace():
    def __init__(self):
        self.need = None

class Stream():
    def __init__(self, mMax, n, ftr, alpha=0.05, trace=None):
        self.trace = trace

        self.mMax = int(mMax)
        self.n = n
        self.ftr = ftr
        self.coord = None
        self.alpha = alpha

    def run(self):
        logging.info('Creating Stream of Coordinate Updates...')

        if self.ftr == "tst":
            self.coord = self.create_test_stream()
            self.mMax = len(self.coord)
            self.n = 3201
            # self.mList = [self.mMax]

        elif self.ftr == "rd":
            if self.n is None: 
                self.n = int(10 * self.mMax)

            if self.trace is not None:
                shuffle = False
            else:
                shuffle = True 

            self.coord = self.create_HH_stream(self.mMax, self.n, alpha=self.alpha, shuffle=shuffle)

        elif self.ftr == "src" or self.ftr[-4:] == "port":
            self.coord = self.load_CAIDA()
            self.mMax = len(self.coord)
            self.n = self.get_n_from_ftr()
        # logging.info(f'{self.normType}-norm of {self.ftr} Stream {self.mList[-1]} with dict {self.n}.')

################################################# TEST ##############################################
    @staticmethod
    def create_test_stream():
        '''
        return: [1,2,3,4,5...,3199, 3200, 3201, 3201, 3201,.... 3201]
        len = 3270 = "1...3200" + 70 x "3201"
        '''
        return np.append(np.arange(1, 3201), np.ones(70)*3201)

################################################# RANDOM ##############################################

    @staticmethod
    def create_HH_stream(m, n, alpha=0.05, shuffle=True):
        '''
        for m = 40, n = 10 * m, 
        return: [  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,   1,   2,   3,
         4,   5,   6,   7,   8,   9,  10,  93,  96,  65,  81, 329, 138,
        61, 102, 145,  62, 180, 108, 312,  78, 345, 101, 400, 400, 400, 400]
        '''
        cut = int(m / 4)
        s1 = np.arange(2, cut + 2)
        rd = np.random.randint(cut + 3, high=n + 1, size=m - 2 * cut)
        coord = np.concatenate((s1, s1, rd))

        # create alpha-heaviness Heavy Hitters
        nHH = np.min([alpha * m, np.sqrt(m)])
        coord[-int(nHH):] = 1

        if shuffle: 
            np.random.shuffle(coord)

        return coord

    # def create_HH_stream(self, m, n):
    #     HH  = np.concatenate((np.ones(m//2), np.ones(m//4)*2, np.ones(m//8)*4))
    #     rd = np.random.randint(5,high=n+1,size = m-len(HH))
    #     stream = np.concatenate((HH, rd))
    #     np.random.shuffle(stream)
    #     return stream

################################################# SOURCE ##############################################

    def load_CAIDA(self):
        dataDir = '/home/swei20/SymNormSlidingWindows/data/stream/'
        data = np.loadtxt(f'{dataDir}traffic_{self.ftr}.txt')
        return data

    def get_n_from_ftr(self):
        if self.ftr[-4:] == 'port':
            # return 2 ** 16
            return 10000
        elif (self.ftr =='src' or self.ftr == 'dst'):
            # return 2 ** 32
            return 10000
        elif self.ftr == 'len':
            return 1500
        else:
            return None