# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 13:46:52 2021

@author: james hazelden
"""

from scipy.fft import fft, fftfreq
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import freqz
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

def get_ticks(n_smpls, log_min, log_max, nticks = 5):
    # Evenly spaced ticks in log space. The third line removes trailing zeros.
    ticks = np.linspace(0, n_smpls - 1, nticks)
    labels = np.round(np.linspace(log_min, log_max, nticks), 1) 
    labels = np.array(["%g" % t for t in labels])
    return ticks, labels

def run():
    # Define timestep (ms), sample rate (Hz), and number of seconds. 
    dt = 0.25
    fs = 1000 / dt
    fs = int(fs)
    log_min = -2.0
    log_max = 0.0
    
    cutoff_buckets = [4, 10, 30, 80, 200, 600]
#    cutoff_buckets = np.linspace(1, 600, 100)

    # Read in sweep results.
    sweep = pd.read_csv("..\coupling_speed_test\SWEEP_OUT_RUN3.txt", delimiter=',', header=None).to_numpy()[:, :-1]
    sweep_params = sweep[:, :2]
    sweep = sweep[:, 2:]   
    dim = int(round(math.sqrt(sweep.shape[0])))
    sweep = sweep.reshape(dim, dim, sweep.shape[1])
    N = sweep.shape[2]
    
    plt.figure(dpi=500)
    means = np.mean(sweep, axis = -1)
    plt.imshow(means, interpolation='quadric', cmap='gist_rainbow', origin="lower")
    plt.axis('on')
    plt.box(on=None)
    plt.tick_params(axis=u'both', which=u'both',length=0)
    plt.xlabel('$\log$ $g_{grey}$', fontsize=12)
    plt.ylabel('$\log$ $g_{white}$', fontsize=12)
    ticks,labels = get_ticks(dim, log_min, log_max, 2)
    plt.xticks(ticks, labels=labels)
    plt.yticks(ticks, labels=labels)
    plt.colorbar()
    plt.show()
    
    grid = np.zeros((len(cutoff_buckets), dim, dim))
    for i in range(dim):
        for j in range(dim):
            yf = fft(sweep[i, j, :])
            xf = fftfreq(N, dt / 1000)[:N//2]
        
            # Note that we remove zero frequency. Also, we skip first transient second.
            xf = xf[1:]
            yf = 2.0 / N * np.abs(yf[1:N//2])
            
            # Bucket fft output.
            k = 0
            for s in range(len(cutoff_buckets)):
                split = cutoff_buckets[s]
                k_prev = k
                while k < xf.size and xf[k] <= split:
                    grid[s, i, j] += yf[k]
                    k += 1
                    
                # Divide by number of samples for bucket.
                if k-k_prev > 0:
                    grid[s, i, j] /= (k - k_prev)
           
    plt.figure(dpi=600)
    titles = ['Delta', 'Theta', 'Beta', 'Gamma', 'Fast', 'Ultra Fast']
    vmin = np.min(grid)
    vmax = np.max(grid)
    vmax = 0.3
    
    plt.figure(dpi=500)
    plt.plot(sweep[5, -1, 12500:15000])
    plt.show()
    
    plt.figure(dpi=500)
    plt.plot(sweep[0, -1, :])
    plt.show()
    
    fig, axs = plt.subplots(2, 3)      
    for s in range(grid.shape[0]):
        ax = axs[s // 3][s % 3]
        plt.subplot(ax)
        plt.imshow(grid[s,:,:], interpolation='quadric', cmap='seismic', vmin=vmin, vmax=vmax, origin="lower")
        ax.axis('off')
        plt.title(titles[s])
        
    # Display color bar and axis labels
    plt.colorbar(ax = axs[:2, :])
    plt.subplot(axs[1][0])
    plt.axis('on')
    plt.box(on=None)
    plt.tick_params(axis=u'both', which=u'both',length=0)
    plt.xlabel('$\log$ $g_{grey}$', fontsize=12)
    plt.ylabel('$\log$ $g_{white}$', fontsize=12)
    ticks,labels = get_ticks(dim, log_min, log_max, 2)
    plt.xticks(ticks, labels=labels)
    plt.yticks(ticks, labels=labels)
    plt.savefig('sweep_fig4.pdf')
    plt.show()
run()