from util import *
import numpy as np

c=2
r=100
device='cuda'
n=100
np.random.seed(42)
stream = np.random.randint(1,n,size=n)
# exactL2 = np.linalg.norm(stream, ord=2)
streamTr=torch.tensor(stream, dtype=torch.int64)
item=streamTr[2]
streamTr0=streamTr[:10]

def main():
    run(streamTr0,c,r,device)


main()
