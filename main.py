import numpy as np
import logging
import cProfile

from dataset.randomstream import create_random_stream
from dataset.traffic import get_packet_stream
from eval.evalNorm import get_estimated_norm
from eval.evalNormCS import get_sketched_norm

# DATASET ='/home/swei20/SymNormSlidingWindows/data/testdata/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'
DATASET ='/home/swei20/SymNormSlidingWindows/data/testdata/test100.pcap'
n, m, w, sRate =100, 10000, 2000, 0.2
assert m >= w
r=int(np.log10(m))
c=100
device='cuda'
normType=['L2','T10'][0]
RANDSTREAM, TEST, EVALNORM = 0,0,1


def main():
    num = 10 if TEST else None
    if RANDSTREAM:
        stream = create_random_stream(n,m)
    else:
        stream = get_packet_stream(DATASET, 'len', num=num)
        n
    print(stream)
    if EVALNORM:
        exactNorm, uniformNorm = get_estimated_norm(normType, stream, n, w, sRate)
        sketchNorm = get_sketched_norm(normType, stream,w,c,r,device)
        print('exact {:0.2f}, uniform {:0.2f}, sketch{:0.2f}'.format(exactNorm, uniformNorm,sketchNorm))
        print('n{}|m{}|w{}|sRate{}|table_c{}r{}'.format(n,m,w,sRate,c,r))


if __name__ == "__main__":
    main()
