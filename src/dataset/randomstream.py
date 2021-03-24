import numpy as np
# import logging
# from collections import Counter

def create_random_stream(n,m, HH=True, HH3=None):
    np.random.seed(42)
    if HH:
        stream = []
        d = int(np.log2(m))
        assert d < n
        for i in range(1,d):
            [stream.append(i) for j in range(int(2**(d-i-1)))]
        l = len(stream)
        stream = np.array(stream)
        rands = np.random.randint(d+2,high=n+1,size = m-len(stream))
        stream = np.append(stream,rands)
    else: 
        stream = np.random.randint(1,high=n+1,size = m)
        if HH3 is not None:
            stream[:int(m//8)] = HH3[2]
            stream[int(m//8):int(m//4)] = HH3[1]
            stream[int(m//4):] = HH3[0]
    assert len(stream) == m
    np.random.shuffle(stream)
    return stream

# n , m = 100, 2**5
# print(create_random_stream(n,m))