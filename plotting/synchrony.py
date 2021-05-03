# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 16:30:22 2021

@author: james hazelden
"""

def run():
    import numpy as np
    import matplotlib.pyplot as plt
#    import pandas as pd 

    chi = np.array([
        3.72835e-05,
        6.82504e-05,
        0.000113323,
        0.000221569,
        0.000505184,
        0.00103658,
        0.00180988,
        0.00376229,
        0.00803446,
        0.0158343
        ])
    
    N = np.array([2**i for i in range(15, 5, -1)])
    N_12 = N ** (-1/2)

    plt.plot(N_12, chi, '-o')
    plt.title('Synchrony of Desynchronized Case with White Noise, No Coupling')
    plt.xlabel('$N_s^{-1/2}$')
    plt.ylabel('$\chi$')
    plt.legend(['Desynchronized Network'])
    plt.show()    
    
run()