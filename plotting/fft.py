# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 01:40:39 2021

@author: jhazelde
"""
import os 
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import pandas as pd

def get_ticks(n_smpls, log_min, log_max, nticks = 5):
    # Evenly spaced ticks in log space. The third line removes trailing zeros.
    ticks = np.linspace(0, n_smpls - 1, nticks)
    labels = np.round(np.linspace(log_min, log_max, nticks), 1) 
    labels = np.array(["%g" % t for t in labels])
    return ticks, labels
    
def run():
    log_min = -3.0
    log_max = -2.0
    n_smpls = 20
    smpls = np.logspace(log_min, log_max, n_smpls)
    print(smpls)
           
    # Define timestep (ms), sample rate (Hz), and number of seconds. 
    dt = 0.4
    fs = 1000 / dt
    fs = int(fs)
                
    # desired split frequencies
    splitcuts = [4.0, 10.0, 30.0, 80.0, 200.0, 600.0, fs / 2.0]

    # Grids to display results for each split frequency.
    # grids = np.zeros((n_smpls, n_smpls, len(splitcuts)))
    # for s1, g_white in enumerate(smpls):
    #     for s2, g_grey in enumerate(smpls):
    #         print(s1, s2)
    #         print(grids[:, :, 0])

    #         # Read in SAN trace.
    #         full = pd.read_csv("HH_OUT_%d_%d.txt" % (s1, s2), delimiter=',').to_numpy()
    #         y = full[:, 0]
    #         y = y[fs:]
    #         N = y.size
        
    #         yf = fft(y)
    #         xf = fftfreq(N, dt / 1000)[:N//2]
            
    #         # Note that we remove zero frequency. Also, we skip first transient second.
    #         xf = xf[1:]
    #         yf = 2.0 / N * np.abs(yf[1:N//2])
    #        # plt.plot(xf[1:], 2.0/N * np.abs(yf[1:N//2]))
            
    #         buckets = grids[s1, s2, :]
    #         # Bucket fft output.
    #         j = 0
    #         for i in range(len(splitcuts)):
    #             split = splitcuts[i]
    #             j_prev = j
    #             while j < xf.size and xf[j] <= split:
    #                 buckets[i] += yf[j]
    #                 j += 1
                    
    #             # Divide by number of samples for bucket.
    #             buckets[i] /= (j - j_prev)
                
    #        # if(s1 == 0):
    #        #     plt.figure(s1 * n_smpls + s2)
    #        #     plt.plot(splitcuts, buckets)
    #        #     plt.show()

    with open('grids.npy', 'rb') as f:    
        grids = np.load(f)
    with open('grids.npy', 'wb') as f:
        np.save(f, grids)
        
    vmin = np.min(grids)
    vmax = np.max(grids)
        
    titles = ['Delta', 'Theta', 'Beta', 'Gamma', 'Fast', 'Ultra Fast']

    
    fig, axs = plt.subplots(2, 3)  
    for f in range(6):
        ax = axs[f // 3][f % 3]
        grid = grids[:, :, f]
        plt.subplot(ax)
        plt.imshow(grid, interpolation='quadric', cmap='seismic', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title(titles[f])
    
    # Display color bar and axis labels
    plt.colorbar(ax = axs[:2, :])
    plt.subplot(axs[1][0])
    plt.axis('on')
    plt.box(on=None)
    plt.tick_params(axis=u'both', which=u'both',length=0)
    plt.xlabel('$\log$ $g_{grey}$', fontsize=12)
    plt.ylabel('$\log$ $g_{white}$', fontsize=12)
    ticks,labels = get_ticks(n_smpls, log_min, log_max, 2)
    plt.xticks(ticks, labels=labels)
    plt.yticks(ticks, labels=labels)
    
    plt.subplots(1,1)
    plt.imshow(grids[:, :, 3], interpolation='quadric', cmap='hot')
    plt.title('Gamma Waves Variation')
    plt.xlabel('$\log$ $g_{grey}$', fontsize=12)
    plt.ylabel('$\log$ $g_{white}$', fontsize=12)
    ticks,labels = get_ticks(n_smpls, log_min, log_max, 5)
    plt.xticks(ticks, labels=labels)
    plt.yticks(ticks, labels=labels)

    plt.show()
    
def test():
    from scipy.fft import fft, fftfreq
    import numpy as np
    # Number of sample points
    N = 600
    # sample spacing
    T = 1.0 / 200.0
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    import matplotlib.pyplot as plt
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.grid()
    plt.show()
    
run()
