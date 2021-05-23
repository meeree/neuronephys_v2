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
from matplotlib import colors, cm

def get_ticks(n_smpls, log_min, log_max, nticks = 5):
    # Evenly spaced ticks in log space. The third line removes trailing zeros.
    ticks = np.linspace(0, n_smpls - 1, nticks)
    labels = np.round(np.linspace(log_min, log_max, nticks), 1) 
    labels = np.array(["%g" % t for t in labels])
    return ticks, labels
    

def plot_individual(grids, idx, name, n_smpls, log_min, log_max):
    plt.subplots(1,1)
    plt.imshow(grids[:, :, idx], interpolation='quadric', cmap='seismic')
    plt.title('%s Waves Variation' % name)
    plt.xlabel('$\log$ $g_{grey}$', fontsize=12)
    plt.ylabel('$\log$ $g_{white}$', fontsize=12)
    ticks,labels = get_ticks(n_smpls, log_min, log_max, 5)
    plt.colorbar()
    plt.xticks(ticks, labels=labels)
    plt.yticks(ticks, labels=labels) 

# Usage:
# log_min specifies minimum value of log(g_grey), log(g_white)
# log_max specifies maximum value of log(g_grey), log(g_white)
# n_smpls is specifies the granularity for the parameter sweep along each axis
# inp_fl_name 
#    is file name with the param sweep results. If it is empty, we read 
#    HH output and write param sweep to file grids_<LOG_MIN>_<LOG_MAX>_<N_SMPLS>.npy
def run(log_min, log_max, n_smpls = 20, inp_fl_name = ''):
    smpls = np.logspace(log_min, log_max, n_smpls)
    print(smpls)
           
    # Define timestep (ms), sample rate (Hz), and number of seconds. 
    dt = 0.4
    fs = 1000 / dt
    fs = int(fs)
                
    # desired split frequencies
    splitcuts = [4.0, 10.0, 30.0, 80.0, 200.0, 600.0, fs / 2.0]
    
    if inp_fl_name != '':
        with open(inp_fl_name, 'rb') as inp_file:
            grids = np.load(inp_file)
    else:
        # Grids to display results for each split frequency.
        grids = np.zeros((n_smpls, n_smpls, len(splitcuts) + 1))
        for s1_raw, g_white in enumerate(smpls):
            for s2_raw, g_grey in enumerate(smpls):
                s1, s2 = s1_raw, s2_raw
                buckets = grids[s1, s2, :]
                print('(%d, %d)' % (s1, s2), end = ',')

                # Read in SAN trace.
                full = pd.read_csv("HH_OUT_%d_%d.txt" % (s1, s2), delimiter=',').to_numpy()
                y = full[:, 0]
                y = y[4*fs:]
                N = y.size
                
                chi = pd.read_csv("HH_CHI_OUT_%d_%d.txt" % (s1, s2), delimiter=',').to_numpy()
                chi = chi[0]
            
                yf = fft(y)
                xf = fftfreq(N, dt / 1000)[:N//2]
                
                # Note that we remove zero frequency. Also, we skip first transient second.
                xf = xf[1:]
                yf = 2.0 / N * np.abs(yf[1:N//2])
                # plt.plot(xf[1:], 2.0/N * np.abs(yf[1:N//2]))
                
                # Bucket fft output.
                j = 0
                for i in range(len(splitcuts)):
                    split = splitcuts[i]
                    j_prev = j
                    while j < xf.size and xf[j] <= split:
                        buckets[i] += yf[j]
                        j += 1
                        
                    # Divide by number of samples for bucket.
                    buckets[i] /= (j - j_prev)
                    
                # Add synchrony.
                buckets[-1] = chi
                    
                if(s1 == 0):
                    plt.figure(s1 * n_smpls + s2)
                    plt.plot(y)
                    plt.show()

    out_fl_name = 'grids_%g_%g_%d.npy' % (log_min, log_max, n_smpls)
    print('Writing to %s' % out_fl_name)
    with open(out_fl_name, 'wb') as f:
        np.save(f, grids)
            
    vmin = np.min(grids)
    vmax = np.max(grids)
    norm = colors.LogNorm(grids.mean() + 0.5 * grids.std(), grids.max(), clip='True')

    titles = ['Delta', 'Theta', 'Beta', 'Gamma', 'Fast', 'Ultra Fast']
    fig, axs = plt.subplots(2, 3)  
    for f in range(6):
        ax = axs[f // 3][f % 3]
        grid = grids[:, :, f]
        plt.subplot(ax)
        plt.imshow(grid, cmap='seismic', interpolation='quadric')
        #plt.plot([2, 20], [0, 18], '--', color=[0,1,0], linewidth=1.0)
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
        
    # Plot synchrony
    plt.subplots(1,1)
    plt.imshow(grids[:, :, -1], cmap='hot', interpolation='quadric')
    plt.colorbar()
    plt.tick_params(axis=u'both', which=u'both',length=0)
    plt.xlabel('$\log$ $g_{grey}$', fontsize=12)
    plt.ylabel('$\log$ $g_{white}$', fontsize=12)
    ticks,labels = get_ticks(n_smpls, log_min, log_max, 5)
    plt.xticks(ticks, labels=labels)
    plt.yticks(ticks, labels=labels)
    
    for i in range(6):
        plot_individual(grids, i, titles[i], n_smpls, log_min, log_max)
        plt.plot([2, 20], [0, 18], color=[0,1,0])
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
    
# NOTE: You need to run this code with coupling_speed_test path as root directory.
run(-2, 0, 20)
