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
    from scipy.signal import freqz
    import pandas as pd

    # Define timestep (ms), sample rate (Hz), and number of seconds. 
    dt = 0.4
    fs = 1000 / dt
    fs = int(fs)

    # desired cutoff frequencies (in Hz).
    lowcut = 0.1
    highcut = 100.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_lowpass(highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Read in SAN trace.
    full = pd.read_csv("..\coupling_speed_test\OUT_AN.txt", delimiter=',').to_numpy()
    
    # Select first column and remove initial transient second.
    real_dt = 0.4
    x = full[int(1000 / real_dt):, 5000]
        
    # Downsample x so that timestep is equal to dt.
    downsample_rate = int(dt / real_dt + 0.5)
    x = x[0::downsample_rate]
    n_seconds_data = x.size * dt / 1000
    t = np.linspace(0.0, n_seconds_data, x.size)

    
    # Plot filtering with all data and with a subset. 
    for i in range(2):
        if i > 0:
            x = x[:int(fs * 2.5)]
            t = t[:int(fs * 2.5)]
            
        plt.figure(2 + i)
        plt.clf()
        plt.plot(t[fs:], x[fs:], label='SAN LFP Trace', linewidth=0.4)
    
        y = butter_lowpass_filter(x, highcut, fs, order=9)
        plt.plot(t[fs:], y[fs:], label='Filtered signal, %.0f Hz' % highcut, linewidth=0.8)
        plt.xlabel('time (seconds)')
        plt.ylabel('Voltage (mV)')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc='upper left')
        plt.title('Lowpass Filtered SAN; Sample Rate = %d Hz' % int(fs))
 #       plt.axhline(y=np.mean(y[fs:]) - 1.5*np.std(y[fs:]), color='r', linestyle='-')

    plt.show()


run()