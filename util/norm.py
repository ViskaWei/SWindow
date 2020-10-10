import numpy as np
import torch

def norm_function(name, isTorch=False):
    """
    choose between 'L1', 'L2'...'Lp', 'Tk'
    """
    if name[0] == 'L':
        try: 
            p = int(name[1:]) 
        except: 
            raise 'enter # for p in Lp'
        if isTorch:
            return lambda x: torch.norm(x, p=p, dim=1, keepdim=False, out=None)
        else:
            return lambda x: sum([abs(i)**p for i in x])**(1.0/p)
    elif name[0] == 'T':
        try: 
            k = int(name[1:])
        except: 
            raise 'enter # for k in Tk'
        if isTorch:
            return lambda x: torch.norm(x, p=p, dim=1, keepdim=False, out=None)
        else:
            return lambda x: sum(np.sort([abs(i) for i in x])[::-1][:k])