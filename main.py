from util import *
from dataset import *
from evalNorm import *
import numpy as np
import logging
import cProfile
n, m, w, sRate =100, 10000, 2000, 0.2
assert m >= w
r=int(np.log10(m))
c=100
device='cuda'
normType=['L2','T10'][0]
streamTr=torch.tensor(stream, dtype=torch.int64)
# streamTr0=streamTr[:10]

def main():
    stream = create_random_stream(n,m)
    exactNorm, uniformNorm = get_estimated_norm(normType, stream, n, w, sRate)
    sketchNorm = get_sketched_norm(normType, streamTr,c,r,device, m-w)
    print('exact {:0.2f}, uniform {:0.2f}, sketch{:0.2f}'.format(exactNorm, uniformNorm,sketchNorm))
    print('n{}|m{}|w{}|sRate{}|table_c{}r{}'.format(n,m,w,sRate,c,r))


if __name__ == "__main__":
    main()
