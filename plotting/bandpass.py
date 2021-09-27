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

def lowpass_downsample_by_factor(data, fs, factor=10, downsample=True):
    for i in range(data.shape[1]):
        # Lowpass filter to fs/factor.
        data[:, i] = butter_lowpass_filter(data[:, i], fs / factor, fs, order=6)
    
    # Downsample to fs/factor.
    if downsample:
        return data[::factor, :], fs / factor
    return data, fs / factor

def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def run():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz
    import pandas as pd
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Define timestep (ms), sample rate (Hz), and number of seconds. 
    dt = 0.4
    fs = 1000 / dt
    fs = int(fs)

    # # desired cutoff frequencies (in Hz).
    # highcut = 250

    # # Plot the frequency response for a few different orders.
    # plt.figure(1)
    # plt.clf()
    # for order in [3, 6, 9]:
    #     b, a = butter_lowpass(highcut, fs, order=order)
    #     w, h = freqz(b, a, worN=2000)
    #     plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    # plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
    #          '--', label='sqrt(0.5)')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Gain')
    # plt.grid(True)
    # plt.legend(loc='best')

    # # Read in SAN trace.
    # full = pd.read_csv("..\coupling_speed_test\ERP_OUT_GRAY_WHITE_AI.csv", delimiter=',').to_numpy()[:, :-1]
    
    # # Read in region names.
    region_names = []
    with open("..\coupling_speed_test\REGION_NAMES.txt", "r") as fl:
        for name in fl:
            region_names.append(name.replace('\n', ''))
            
    # ys = np.zeros_like(full[fs//10:, :])
    # # Select first column and remove initial transient second.
    # for n in range(full.shape[1]):
    #     real_dt = 0.4
    #     x = full[:, n]
            
    #     # Downsample x so that timestep is equal to dt.
    #     downsample_rate = int(dt / real_dt + 0.5)
    #     x = x[0::downsample_rate]
    #     n_seconds_data = x.size * dt / 1000
    #     t = np.linspace(0.0, n_seconds_data, x.size)
        
    #     # Plot filtering with all data and with a subset. 
    #     for i in range(1):
    #         if i > 0:
    #             x = x[:int(fs * 2.5)]
    #             t = t[:int(fs * 2.5)]
                
    #         plt.figure(2 + i, dpi=600)
    #         plt.plot(t[fs:], x[fs:], label='SAN LFP Trace', linewidth=0.4)
        
    #         y = butter_lowpass_filter(x, highcut, fs, order=9)
    #         ys[:, n] = y[fs//10:]
    #         plt.plot(t[fs//10:], n * 2 + y[fs//10:], label='Filtered signal, %.0f Hz' % highcut, linewidth=0.8)
    #         plt.xlabel('time (seconds)')
    #         plt.ylabel('Voltage (mV)')
    #         plt.grid(True)
    #         plt.axis('tight')
    #        # plt.legend(loc='upper left')
    #         plt.title('Lowpass Filtered SAN; Sample Rate = %d Hz' % int(fs))
    #  #       plt.axhline(y=np.mean(y[fs:]) - 1.5*np.std(y[fs:]), color='r', linestyle='-')
     
    firing = pd.read_csv("..\coupling_speed_test\FIRING_OUT_GRAY_WHITE_VIS.csv", delimiter=',', header=None).to_numpy()[:, :-1]
    total_firing = np.sum(firing, axis=0)

    # Plot total firing.
    plt.figure(dpi=600)
    plt.barh(range(total_firing.shape[0]), total_firing)
    plt.yticks(range(total_firing.shape[0]), region_names)
    plt.title('Total Firing Per Region, Stimulating VIS Region')
    
    # Plot region sizes as bar chart. 
    plt.figure(dpi = 600)
    region_sizes = np.genfromtxt("..\coupling_speed_test\REGION_SIZES.csv")
    sorted_names = [name for _, name in sorted(zip(region_sizes, region_names))]
    region_sizes = np.array(sorted(region_sizes))
    plt.barh(range(len(region_sizes)), region_sizes)
    plt.yticks(range(len(region_sizes)), sorted_names)
    plt.show()
    
    # Plot physical connections as probability a connection is formed from region i to j. 
    plt.figure(dpi=600)
    physical_conns = pd.read_csv("..\coupling_speed_test\PHYSICAL_CONNECTIVITY.csv", delimiter=',', header=None).to_numpy()[:, :-1]    
    max_conns = np.outer(region_sizes, region_sizes) # entry i,j is region_sizes[i] * region_sizes[j]
    physical_conns /= max_conns # Compute probabilities
    axp = plt.imshow(physical_conns)
    plt.colorbar(axp,ax=[plt.gca()],label="Probability of Connection", location='left')
    plt.xticks(range(firing.shape[1]), region_names, rotation=90)
    plt.yticks(range(firing.shape[1]), region_names)
    plt.show()


    # ttl = np.zeros(functional.shape[0])
    # for w in windows:
    #     for i in range(firing.shape[1]):
    #             for j in range(firing.shape[1]):
    #                 if j == i:
    #                     break
    #                 functional[i, j] = np.corrcoef(firing[w:w+window_size, [i, j]], rowvar=False)[1,0]
    #     plt.barh(range(firing.shape[1]), np.sum(functional, axis=0))
    #     ttl += np.sum(functional, axis=0)
    #     plt.yticks(range(firing.shape[1]), region_names)
    #     plt.show()
    # plt.barh(range(firing.shape[1]), ttl)
    # plt.yticks(range(firing.shape[1]), region_names)
    # plt.show()
    
        
    # Subsample so that we only display specific regions, allowing us to compare to other papers.
    # desired_regions = ['RSP', 'VISC', 'MO', 'SS',  'VIS']
    # inds = [region_names.index(name) for name in desired_regions]
    # region_names = [region_names[index] for index in inds]
    # firing = firing[:, inds]
    
    # Plot firing traces with filtering to range [0.009, 0.08] Hz. This is tricky and 
    # we accomplish it by a sequence of downsampling and then filtering.
    plt.figure(dpi=600)
    plt.plot(firing[::100, 0], firing[::100, 1])
    plt.title("Phase Space %i, %i" % (0, 1))
           
    # First downsample from 2500 to 1000. This can be done by upsampling by a factor of 2 then downsample by a factor of 5.
    filtered_firing = firing
    filtered_firing = np.repeat(filtered_firing, 2, axis=0)
    filtered_firing, fs_filter = lowpass_downsample_by_factor(filtered_firing, fs * 2, 5)

    # Downsample 3 times by a factor of 10 to go from 1000 to 1.0. 
    for i in range(3):
        filtered_firing, fs_filter = lowpass_downsample_by_factor(filtered_firing, fs_filter)
    
    # Finally, lowpass filter from 1.0 to 0.08.
    for i in range(filtered_firing.shape[1]):
        # Lowpass filter to fs/factor.
        filtered_firing[:, i] = butter_lowpass_filter(filtered_firing[:, i], 0.08, 1.0, order=6)

    plt.subplot(211)
    plt.plot(filtered_firing[50:, :])
    plt.subplot(212)
    plt.plot(firing)
    plt.show()
    
    firing = filtered_firing[100:, :]
    fs = fs_filter
    
    # Functional connectivity over total timeframe.
    plt.figure(dpi = 600)
    functional = np.zeros((firing.shape[1], firing.shape[1]))
    vmax, vmin = (-0.3, 0.3)
    for i in range(firing.shape[1]):
           for j in range(firing.shape[1]):
               functional[i, j] = np.corrcoef(firing[:, [i, j]], rowvar=False)[1,0]
    plt.imshow(functional, cmap='hot', vmin=-0.3, vmax=0.3)
    plt.colorbar()
    plt.xticks(range(firing.shape[1]), region_names, rotation=90)
    plt.yticks(range(firing.shape[1]), region_names)
    plt.title("fMRI Functional Connnectivity, Stimulating VIS region")
    plt.show()
    
    # Plot sum across rows of functional connectivity.
    plt.figure(dpi = 600)
    plt.barh(range(firing.shape[1]), sorted(np.sum(functional, axis=0).reshape(-1)))
    plt.yticks(range(firing.shape[1]), region_names)
    plt.show()
   
    # # Animate the functional connectivity. 
            
    # # Sliding window analysis.
    # fig = plt.figure(4, dpi=600)
    # window_size = firing.shape[0] - 1
    # window_inc = window_size - 1
    # windows = range(0, 1 + firing.shape[0] - window_size, window_inc)
    
    # functional = np.zeros((firing.shape[1], firing.shape[1]))
    # im = plt.imshow(functional, vmin=vmin, vmax=vmax, cmap='jet',
    #                 #interpolation='quadric'
    # )
    # plt.xticks(range(firing.shape[1]), region_names, rotation=90)
    # plt.yticks(range(firing.shape[1]), region_names)
    
    # if firing.shape[0] - window_size < 0:
    #     print("ERROR: window size %d is too big for amount of data possible: %d." % (window_size, firing.shape[0]))
    #     return 1

    # def update(w):
    #     for i in range(firing.shape[1]):
    #         for j in range(firing.shape[1]):
    #             # if j == i:
    #             #     break
    #             functional[i, j] = np.corrcoef(firing[w:w+window_size, [i, j]], rowvar=False)[1,0]
                
    #     im.set_array(functional)
    #     plt.title("Window = [%.2f, %.2f), fMRI Functional Connnectivity" % (dt * w / 1000, dt *(w+window_size) / 1000))
    #     return [im]
        
    # # Create the animation
    # ani = FuncAnimation(fig, update, frames=windows, blit=True)
    # writer = PillowWriter(fps=2)
    # ani.save('functional_connectivity_sliding_window.gif', writer = writer)
    # plt.show()


run()