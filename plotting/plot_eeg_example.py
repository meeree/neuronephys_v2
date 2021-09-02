# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:57:05 2021

@author: jhazelde
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.colors import LogNorm

eeg = pd.read_csv("../coupling_speed_test/ELECTRODES_OUT.txt", delimiter=',', header=None).to_numpy()[:, :-1]
eeg = eeg[100:, :]

plt.figure(dpi=600)
plt.plot(eeg)
plt.title('Electrodes Example')
plt.show()