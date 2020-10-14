import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

plt.style.use(['ieee','no-latex'])

DATADIR = '/home/swei20/SymNormSlidingWindows/'
def get_path(filename,DATADIR=DATADIR):
    return os.path.join(DATADIR, filename)
def get_pd(filename, mLoop=True, normType='L2'):
    ftr = filename.split('_')[1]
    path = get_path(filename)    
    out = pd.read_csv(path)
    if mLoop:
        plot_error(out, ftr, normType)  
    else:
        plot_size_error(out, ftr, normType)
    return out
def plot_error(out, ftr, normType, cols=['errUn','errCs'], labels=['uniform','sketch']):
    for i, col in enumerate(cols): 
        plt.scatter(np.log10(out['m']), out[col], label = f'{labels[i]}')
    plt.legend(frameon=True, facecolor='lightgrey')
    plt.grid()
    plt.ylabel(f'{normType} error')
    plt.xlabel('log stream size')
    plt.title(f'CAIDA {ftr}')
def plot_size_error(out, ftr, normType):
    rList = out['r'].unique()
    colors = cm.get_cmap('viridis')(np.linspace(1,0,len(rList)))
    for i, r in enumerate(rList):
        outR = out[out['r']==r]
        plt.scatter(outR['cr'], outR['errCs'], color = colors[i] ,label=f'r={r}')
    plt.legend(frameon=True, facecolor='lightgrey')    
    plt.grid()
    plt.ylabel(f'{normType} error')
    plt.xlabel('log sketch size')
    plt.title(f'CAIDA {ftr}')