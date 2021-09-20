# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 02:13:23 2021

@author: jhazelde
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_over_time(ls, label, n_bins=20, stretch=4):    
    # Get max and min of all weights over time.
    mn = 100000
    mx = -100000
    for i, name in enumerate(ls):
#        hf = h5py.File(name, 'r')
#        weights = hf.get('data').value
        weights = np.genfromtxt(name, delimiter=',')[:-1]
        if np.min(weights[:]) < mn:
            mn = np.min(weights[:])
        if np.max(weights[:]) > mx:
            mx = np.max(weights[:])    
    cmap = cm.get_cmap('seismic')

    ax = plt.subplot(111)
    vals_2d = np.zeros((len(ls), n_bins*stretch))    
    bins=[mn + j * (mx - mn) / n_bins for j in range(n_bins+1)]
    for i, name in enumerate(ls):
        weights = np.genfromtxt(name, delimiter=',')[:-1]
#        hf = h5py.File(name, 'r')
#        weights = hf.get('data').value
        
        # Histogram plot.
        kwargs = dict(alpha=0.6, density=True, bins=n_bins, ec="k", log=True)
        plt.hist(weights, facecolor=cmap(i / len(ls)), **kwargs)

        # Get row for imshow plot.
        h,_ = np.histogram(weights, n_bins)
        h = np.log10(h) # Log based scale
        vals_2d[i, :] = [h[i // stretch] for i in range(len(h) * stretch)]
         
    plt.xlabel('Weight')
    plt.ylabel('Density')   
    plt.title('Weights Histogram Over Time During %s' % label)

    # Add colorbar to temporal histogram.
    norm = mpl.colors.Normalize(vmin=0, vmax=60)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label('Time (s)')
     
    plt.figure(dpi=dpi)
    vals_2d = np.array(vals_2d)
    plt.imshow(vals_2d, cmap=cmap)
    plt.xlabel('Weight')
    plt.ylabel('Time')   
    plt.title(label)
    plt.xticks([0, n_bins*stretch-1], labels=["%.2g" % v for v in [mn, mx]]) 


def plot_weights(weights, case=''):
    plt.figure(dpi=dpi) 
    plt.hist(weights, bins=10)
    plt.title('Histogram: %s' % case)
    plt.xlabel('Weight')
    plt.ylabel('Count')
    
    plt.figure(dpi=dpi)
    plt.plot(weights)
    plt.title('Weights: %s' % case)
    plt.xlabel('Synapse index')
    plt.ylabel('Weight')
   

# 300 dpi: plain paper prints, 600: professional papers, 1200: posters, top quality.
dpi = 600
mpl.rcParams['agg.path.chunksize'] = 10000

plt.figure(dpi=dpi)
plot_over_time(["../coupling_speed_test/HEBBIAN_WEIGHTS_%d.txt" % (i * 1000) for i in range(25)], 'Sleep-Wake Transition', stretch=1)

#plt.figure(dpi=dpi)
#plot_over_time(["../coupling_speed_test/WEIGHTS_OUT_i=%d.h5" % (i * 25000) for i in range(1, 60)], 'Sleep', stretch=1)

# for t, lbl in zip([0.0, 0.5, 1.0, -1], ['Wake', 'Transition', 'Deep Sleep', 'Wake, long']):
#     hf = h5py.File("../coupling_speed_test/WEIGHTS_OUT_%g.h5" % t, 'r')
#     weights = hf.get('data').value
#     plot_weights(weights, lbl)

plt.show()