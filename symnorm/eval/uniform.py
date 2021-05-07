import logging
import numpy as np
# import scipy.stats
import pandas as pd
from collections import Counter



class UniformTrace():
    def __init__(self):
        self.freqVecUU = None
        self.normEx = None
        self.US3 = None
        self.UU3 = None
    

class Uniform():
    def __init__(self, norm_fns, sRate, aveNum=1, rDigit=2, trace=None):
        self.trace = trace

        self.norm_fns = norm_fns
        self.sRate = sRate
        self.rDigit = rDigit
        self.aveNum = aveNum
        self.fill_values = np.zeros(3)

        if self.trace is not None: np.random.seed(1178)

    def get_norms_from_coord(self, coord):
        counter = Counter(coord)
        freqVec = list(counter.values())
        norms = self.get_norms_from_freqVec(freqVec)
        return norms, counter

    def get_norms_from_freqVec(self, freqVec):
        if isinstance(self.norm_fns, list):
            return np.array([norm_fn(freqVec) for norm_fn in self.norm_fns])
        else:
            return self.norm_fns(freqVec)

    def get_norms_from_sampled_universe(self, counter, n):
        sampleSize = int(n * self.sRate)
        sampledDict = np.random.choice(np.arange(1, n + 1), size=sampleSize, replace = False)
        
        freqVec = [counter[item] for item in sampledDict]

        if self.trace is not None:
            self.trace.cUU = counter
            self.trace.sDict = sampledDict
            self.trace.freqVecUU = freqVec

        sampleNorm = self.get_norms_from_freqVec(freqVec)
        return sampleNorm / self.sRate 

    def get_norms_from_sampled_coord(self, coord):
        sampleSize = int(len(coord) * self.sRate)
        sampleCoord = np.random.choice(coord, size=sampleSize, replace = False)
        freqVec = list(Counter(sampleCoord).values())

        if self.trace is not None:
            self.trace.freqVecUS = freqVec

        sampleNorm = self.get_norms_from_freqVec(freqVec)
        return sampleNorm / self.sRate 
    
 
    def get_stats_from_norms(self, norms, normEx):
        norm, std = norms.mean(), norms.std()
        err = abs(normEx - norm) / normEx
        std = std / normEx
        return np.around([norm, err, std], self.rDigit)
################################################# EVAL NORMS ##############################################

    def run_MLoop(self, coord_full, mList, n):
        out =[]
        for m in mList:
            coord = coord_full[:m]
            normEx, US3, UU3 = self.run(coord, n)
            out.append([normEx, *US3, *UU3])
        return np.array(out)

    def run(self, coord, n):
        normEx, counter = self.get_norms_from_coord(coord)
        normEx = np.around(normEx, self.rDigit + 1)
        
        normUSs = np.array([self.get_norms_from_sampled_coord(coord) for i in range(self.aveNum)])
        US3 = self.get_stats_from_norms(normUSs, normEx)

        normUUs = np.array([self.get_norms_from_sampled_universe(counter, n) for i in range(self.aveNum)])
        UU3 = self.get_stats_from_norms(normUUs, normEx)


        if self.trace is not None:
            self.trace.normEx = normEx
            self.trace.US3 = US3
            self.trace.UU3 = UU3

        return normEx, US3, UU3


    