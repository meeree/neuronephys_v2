# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:44:34 2021

@author: james hazelden
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd    
    
def plot_eeg(eeg, ax, eeg_inds=None, label=False):
    N, M = eeg.shape 
    inc_scalar = 5
    incs = np.tile(np.arange(0, M), (N, 1)) * inc_scalar
    
    incremented = eeg + incs - np.mean(eeg, axis=0)
    plt.plot(incremented, linewidth=0.4)
    plt.xlim([0, N])
    plt.ylim([np.min(incs) - inc_scalar, np.max(incs) + inc_scalar])


    # Remove box and setup axis ticks.
    if not label:
        plt.box(False)

    ax.spines['left'].set_position(('outward', 1))
    ax.tick_params('y', length=1, pad=1, width=0.5, which='major')
    ax.tick_params('x', length=5, pad=1, width=1, which='major')

    plt.yticks(incs[0, :], labels=eeg_inds, fontsize=4)
    if label:
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
           
        x_labels = ["%g" % i for i in np.linspace(0.0, 1.0, 3)]
        plt.xticks(np.linspace(0, 2500, 3), labels=x_labels, fontsize=6)
        plt.xlabel('Time (s)', fontsize=8)
        #plt.ylabel('EEG Index')
    else:
        ax.axes.get_xaxis().set_visible(False)

def run():
    # 300 dpi: plain paper prints, 600: professional papers, 1200: posters, top quality.
    dpi = 600
    plt.figure(dpi=dpi)
    plt.tight_layout()

    # Read in EEG traces.
    y_real = pd.read_csv("../coupling_speed_test/OUT_HH_ALLEN.txt", delimiter=',').to_numpy()
    y_uniform_local = pd.read_csv("../coupling_speed_test/OUT_HH_UNIFORM_LOCAL.txt", delimiter=',').to_numpy()
    y_uniform = pd.read_csv("../coupling_speed_test/OUT_HH_UNIFORM.txt", delimiter=',').to_numpy()
    
    # Select eeg traces for regions containing at least 100 neurons. 
    inds = [0,1,2,4,6,7,8,9,10,14,15,16,17,18,19,20,21,22,25,26,27,29,32,33,34,35,37,38,39,42,44,45,48,50,51,52,53,54,56,57,58,62,63,65,67,70,71,72,73,74,75,76,77,78,79,81,82,83,84,85,86,88,89,90,91,92,94,95,96,98,99,100,101,102,104,105,106,109,112,113,114,116,118,119,120,121,122,124,125,127,128,129,131,132,135,136,137,139,140,141,143,144,145,146,147,148,150,151,152,153,154,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,175,176,177,178,181,182,184,185,186,187,188,189,190,194,195,196,197,198,199,200,201,202,203,204,206,207,208,209,212,213,214,215,218,220,224,225,226,227,229,230,232,233,235];
    y_real = y_real[:, inds];
    y_uniform_local = y_uniform_local[:, inds];
    y_uniform = y_uniform[:, inds];
    
    # Remove first second.
    y_real = y_real[2500:, :];
    y_uniform_local = y_uniform_local[2500:, :];
    y_uniform = y_uniform[2500:, :];

    # plt.subplot(121)
 #   plt.plot(y_real)
    eeg_idx = range(0, y_real.shape[1])[::5]

    ax = plt.subplot(131)
    plot_eeg(y_real[:, ::5], ax, eeg_idx, label=True)
   # plt.title('Grey and White')
    plt.title('A', fontweight='bold', loc='left')
    
    ax =plt.subplot(132)
    plot_eeg(y_uniform_local[:, ::5], ax, eeg_idx)
   # plt.title('Uniform and Grey')
    plt.title('B', fontweight='bold', loc='left')
    
    ax = plt.subplot(133)
    plot_eeg(y_uniform[:, ::5], ax, eeg_idx)
   # plt.title('Uniform')
    plt.title('C', fontweight='bold', loc='left')
    
    chi = np.array([
        [    
        0.00538548,
        0.00555288,
        0.00566938,
        0.00557819,
        0.00570637,
        0.00527938,
        0.0061578,
        0.00589833,
        0.00658946,
        0.0110413,
        0.0168005],
        [
        0.021131,
        0.0212917,
        0.0213907,
        0.0213078,
        0.0223403,
        0.0213981,
        0.0219327,
        0.0243542,
        0.0243875,
        0.0261889,
        0.0273361], 

        [
        0.0351544,
        0.0351806,
        0.0344181,
        0.0336295,
        0.0327805,
        0.0338595,
        0.0334795,
        0.0340333,
        0.0346071,
        0.033407,
        0.0306121],
    ])
    
    N = np.array([2**i for i in range(16, 5, -1)])
    N_12 = N ** (-1/2)
    plt.savefig('uniform_vs_real_eeg.png', dpi=dpi, bbox_inches='tight')

    plt.figure(dpi=dpi)
    colors = ['green', 'blue', 'red']
    markers = ['o', '^', 's']
    for chi_sub, color, marker in zip(chi, colors, markers):
        plt.plot(N_12, chi_sub, '-' + marker, linewidth=3, markersize=6, color=color)
    plt.yscale("log")
    plt.title('D', fontweight='bold', loc='left', fontsize=20)
    plt.xlabel('$N_s^{-1/2}$', fontsize=15)
    plt.ylabel('$\chi$', fontsize=15)
    plt.legend(['Gray + White', 'Gray + Uniform', 'Uniform Only'])
    plt.savefig('uniform_vs_real_synchrony.png', dpi=dpi, bbox_inches='tight')
    plt.show()    
    
run()