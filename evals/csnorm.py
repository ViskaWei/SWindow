# import numpy as np
import torch
import logging
LARGEPRIME = 2**61-1

# torch.random.manual_seed(42)

class CSNorm(object):
    def __init__(self, id, norm_fn, c, r, device=None):
        self.r = r # num of rows
        self.c = c # num of columns
#         self.d = int(d) # vector dimensionality
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            if (not isinstance(device, torch.device) and
                    not ("cuda" in device or device == "cpu")):
                msg = "Expected a valid device, got {}"
                raise ValueError(msg.format(device))
        self.device = device
        self.table = torch.zeros((r, c), device=self.device)
        self.id = id
        self.norm = None
        # torch.random.manual_seed(42)
        # rand_state = torch.random.get_rng_state()
        # torch.random.set_rng_state(rand_state)
        self.hashes = torch.randint(0, LARGEPRIME, (self.r, 6),
                               dtype=torch.int64, device="cpu")
        self.h1 = self.hashes[:,0:1]
        self.h2 = self.hashes[:,1:2]
        self.h3 = self.hashes[:,2:3]
        self.h4 = self.hashes[:,3:4]
        self.h5 = self.hashes[:,4:5]
        self.h6 = self.hashes[:,5:6]
        
        self.norm = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.norm_fn = norm_fn
#         self.topk = torch.zeros((k,2), dtype=torch.int64, device=self.device)        
    
    def accumulateVec(self, vec):
        assert(len(vec.size()) == 1 or vec.size()==torch.Size([]))
        signs = (((self.h1 * vec + self.h2) * vec + self.h3) * vec + self.h4)
        signs = ((signs % LARGEPRIME % 2) * 2 - 1).float()
        signs = signs.to(self.device)
        # computing bucket hashes (2-wise independence)
        buckets = ((self.h5 * vec) + self.h6) % LARGEPRIME % self.c
        buckets = buckets.to(self.device)
        for r in range(self.r):
            bucket = buckets[r,:]
            sign = signs[r,:]
            self.table[r,:] += torch.bincount(input=bucket,
                                               weights=sign,
                                                minlength=self.c)
        # print(f'=================={self.id}=={vec}===============')
        # logging.
        #  print(f'{self.table}')
    # @staticmethod
    def get_norm(self):
        norms = self.norm_fn(self.table)
        logging.debug(f'norms: {norms}')
        # print(norms)
        assert(len(norms)==self.r)
        if self.r<4:
            self.norm = torch.mean(norms)
        else:
            self.norm = torch.median(norms)

    