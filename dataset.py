import numpy as np
import logging

def create_stream(n,m,w):
    assert m > w
    np.random.seed(42)
    stream = np.random.randint(1,n,size=m)
    exactL2 = np.linalg.norm(stream, ord=2)
    exactL2w = np.linalg.norm(stream[-w:], ord=2)    
    logging.info('exact L2 of whole stream {:0.2f}'.format(exactL2))
    logging.info('exact L2 of last windowed stream {:0.2f}'.format(exactL2w))
    return stream, exactL2, exactL2w

def uniform_sampling(stream, w, sRate=0.1):
    np.random.seed(926)
    samples = np.random.choice(stream, size=int(w*sRate))
    samplesNorm = np.linalg.norm(samples, ord=2)
    uniformL2 = np.sqrt(samplesNorm**2 / sRate)
    return uniformL2