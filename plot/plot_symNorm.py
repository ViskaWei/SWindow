import os
import sys
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt


# def plot_symNorm(self):

def plot_mL(self, ax=None):
    color = ['r', 'salmon']
    if ax is None: ax=plt.gca()
    df = self.dfEval
    ax.plot(df['m'], df['errUS'], 'bo-', label = 'Uniform (Stream)')
    lbUS = np.clip(df['errUS'] - df['stdUS'], 0, None)
    ubUS = df['errUS'] + df['stdUS']
    ax.fill_between(df['m'], lbUS, ubUS, alpha=0.3, color = 'lightblue', label = "1$\sigma$-CI")

    ax.plot(df['m'], df['errUU'], 'go-', label = 'Uniform (Universe)')
    lbUU = np.clip(df['errUU'] - df['stdUU'], 0, None)
    ubUU = df['errUU'] + df['stdUU']
    ax.fill_between(df['m'], lbUU, ubUU, alpha=0.3, color = 'lightgreen', label = "1$\sigma$-CI")

    for widx, w in enumerate(self.wRList):
        df0 = df[df['w'] == w]
        ax.plot(df0['m'], df0['errCs'], 'o-', c=color[widx], label = f'Sketch (w = m/{w})')
    
    ax.set_xlabel('stream size m')
    ax.set_ylabel('% error')
    ax.set_title(f"{self.normType} | {self.ftr} | rc = {df['rc'][0]} |")
    ax.set_xscale('log', basex=2)
    # plt.yscale('log', basey=2)
    ax.legend(bbox_to_anchor=(1.5, 0.5), loc='center right', ncol=1)

def plot_sL(self):
    df = self.dfEval    
    # plt.plot(df['w'], df['ex'], 'ko', label = 'exact')
    plt.plot(df['rc'], df['errCs'], 'ro-', label = 'CS')
#     plt.plot(df['cr'], df['errUn'], 'bo', label = 'Uniform')
    plt.xlabel('log2(table size)')
    plt.ylabel('abs error of norm')
    plt.title(f"| m = {df['m'][0]} | w = {df['w'][0]} |")
    plt.legend()