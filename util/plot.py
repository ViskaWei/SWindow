import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(['ieee','no-latex'])

DATADIR = '/home/swei20/SymNormSlidingWindows/'
def get_path(filename,DATADIR=DATADIR):
    return os.path.join(DATADIR, filename)
def get_pd(filename, mLoop=True):
    path = get_path(filename)    
    out = pd.read_csv(path)
    if mLoop:
        plot_error(out)  
    else:
        plot_size_error(out)
    return out
def plot_error(out, cols=['errUn','errCs'], labels=['uniform','sketch']):
    for i, col in enumerate(cols): 
        plt.scatter(np.log10(out.m), out[col], label = f'{labels[i]}')
    plt.legend(frameon=True, facecolor='lightgrey')
    plt.grid()
    plt.ylabel('error')
    plt.xlabel('log stream size')
def plot_size_error(out):
    rList = out['r'].unique()
    for r in rList:
        outR = out[out['r']==r]
        plt.scatter(outR['cr'], outR['errCs'], label=f'r={r}')
    plt.legend(frameon=True, facecolor='lightgrey')    
    plt.grid()
    plt.ylabel('error')
    plt.xlabel('log sketch size')