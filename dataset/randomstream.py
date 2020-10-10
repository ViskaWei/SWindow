import numpy as np
# import logging
# from collections import Counter

def create_random_stream(n,m):
    # np.random.seed(42)
    stream = np.random.randint(1,high=n+1,size = m)
    assert len(stream) > 1 
    return stream

# def 