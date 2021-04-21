# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 13:46:52 2021

@author: james hazelden
"""

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='hp')
    return b, a


def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='lp')
    return b, a


def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def run():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd 
    
    # Define timestep (ms), sample rate (Hz), and number of seconds. 
    dt = 0.4
    fs = 1000 / dt
    fs = int(fs)

    # desired cutoff frequencies (Hz).
    highcut = 4.0

    # Read in SAN trace.
    full = pd.read_csv("..\coupling_speed_test\OUT_AN.txt", delimiter=',', nrows=fs * 5).to_numpy()
    
    # Select first column and remove initial transient second.
    real_dt = 0.4
    
    grid_dim = 100
    grid_N = grid_dim * grid_dim
    delays = np.zeros(grid_N, dtype=int)
    plt.figure(1)
        
    tol = 1.5
    for i in range(100):
        print(i)
        for n in range(grid_N):    
            x = full[int(1000 / real_dt):, n]
                
            # Downsample x so that timestep is equal to dt.
            downsample_rate = int(dt / real_dt + 0.5)
            x = x[0::downsample_rate]
            n_seconds_data = x.size * dt / 1000
            t = np.linspace(0.0, n_seconds_data, x.size)
        
            x = x[:int(fs * 3.5)]
            t = t[:int(fs * 3.5)]    
            y = butter_lowpass_filter(x, highcut, fs, order=3)
            
            # Remove first second of data. Filtering can mess up initial part of data.
            t, y = t[fs:], y[fs:]
    
            # Get index of first element of y satisfying y_i < mean(y) - 1.5 * std(y).
            delay = 0
            while delay == 0 and tol > 0:
                delay = (y < np.mean(y) - tol * np.std(y)).argmax()
                break
                tol -= 0.01
                if tol < 0.01:
                    print(tol)
    
            delays[n] = delay
        
        
        # replace occurences where no negative crossing was found with a delay of max_delay + 1.
        max_delay = max(delays)
        delays = np.array([d if d > 0 else max_delay + 1 for d in delays])
        
        # plt.figure(2)
        # plt.imshow(delays.reshape((grid_dim,grid_dim)), interpolation='quadric')
        # plt.colorbar()
        # plt.title("Delay Till Negative Peak")
        
        # unq = np.sort(np.unique(delays))
        # for i, d in enumerate(delays):
        #     delays[i] = np.where(unq == d)[0]
        
        plt.figure(2 + i)
        # Sort grid cells based on delays.
        # See: https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
        grid = np.arange(grid_N, dtype=int)
        print(delays.shape, grid.shape)
        grid = np.array([g for _, g in sorted(zip(delays, grid))] )
            
        plt.imshow(delays.reshape((grid_dim,grid_dim)), interpolation='quadric')
        plt.colorbar()
        plt.title("Relative Index of Negative Peak, tol=%f" % tol)
        
        tol -= 0.01
        
    plt.show()

run()