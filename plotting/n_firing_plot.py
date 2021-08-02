# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:07:02 2021

@author: james hazelden
"""

import matplotlib.pyplot as plt
import pandas as pd    
import numpy as np


plt.figure(dpi=600)
data = pd.read_csv('../coupling_speed_test/N_FIRING_APPCUR.csv', delimiter=',').to_numpy()
plt.plot((data[:, 1] - 4) * 1e4, linewidth=0.1)
plt.plot(data[:, 0], linewidth=0.2)
plt.show()