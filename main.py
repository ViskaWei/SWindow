from util import *
from dataset import *
import numpy as np
import logging
import cProfile
# n, m, w =100, 10000, 2000
n, m, w, sRate =100, 100, 20, 0.2
assert m >= w
r=int(np.log10(m))
c=50
device='cuda'
normType=['L2','T10'][0]
stream,  exactNorm, uniformNorm = create_stream(normType, n,m,w, sRate)
streamTr=torch.tensor(stream, dtype=torch.int64)
# streamTr0=streamTr[:10]
def main():
    sketchNorm = run(normType, streamTr,c,r,device, m-w)
    print('exact {:0.2f}, uniform {:0.2f}, sketch{:0.2f}'.format(exactNorm, uniformNorm,sketchNorm))
    print('n{}|m{}|w{}|sRate{}|table_c{}r{}'.format(n,m,w,sRate,c,r))
main()
