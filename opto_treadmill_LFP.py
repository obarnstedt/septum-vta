# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:01:58 2023

@author: Petra
"""
import os
simpleLFP_directory = "/path/to/simpleLFP/"
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert

#import data and functions
os.chdir(simpleLFP_directory)
import simpleLFP

#define variables
main_dir = "/path/to/data"
folder="data"
frame_rate=40
recording_length=24000
recording_time = 60.0
IGOR_folder = "/path/to/folder"
ad='ad6'
filetype='.ibw'
low_cutoff = 15
downsample = 2400
    
#define LFP data path
path_to_IGOR = os.path.join(main_dir, folder, "IGOR", "")
        
simpleLFP_signal = simpleLFP.signal_processing(recording_seconds=recording_time, notch_freq=50.0, downsample=downsample)
simpleLFP_spectral = simpleLFP.spectral_analysis(recording_seconds=recording_time)
simpleLFP_loader = simpleLFP.load(simpleLFP_directory, path_to_IGOR, IGOR_folder, ad, filetype)
files, Waves, TimeVec = simpleLFP_loader.ImportFolder()
  
#test wave
wave = 0
Waves[wave].shape

# plotting raw data
LFP_wave = np.arange(0, Waves[wave].shape[0])
plt.figure(figsize=(10,6))
plt.plot(LFP_wave, Waves[wave])
plt.title('LFP raw signal')

#filter signal
signal = Waves[wave]
signal_despike= simpleLFP_signal.filter_signal(signal, despike=True, use_hilbert=False, threshold_value=None,
                  low_cutoff=low_cutoff, high_cutoff=None, order_low=5, order_high=5, notch=False)

#plot original and filter signal
fig, (ax1, ax2) = plt.subplots (2,1, figsize=(10,6))
fig.suptitle('original and despiked signal with downsampling')
ax1.plot(signal, label='LFP')
ax2.plot (signal_despike, label='LFP despike')

#plot wavelet spectrum
wavelet_despike = simpleLFP_spectral.wavelet_transform(signal_despike, wavelet_freq=range(0,20), plot=True)

#calculate power spectrum density
_, psd = simpleLFP_spectral.power_spectrum_density(signal)
_, psd_despike = simpleLFP_spectral.power_spectrum_density(signal_despike)

#plot original and filtered psd
fig, (ax1, ax2) = plt.subplots (1,2, figsize=(10,6))
fig.suptitle('psd signal with downsampling')
ax1.plot(psd, label='psd')
ax1.set_title('Psd', loc='left')
ax2.plot (psd_despike, label='psd despike')
ax2.set_title('Psd_despike_downsampled', loc='left')

#calculate frequency amplitude in a range 
high_cutoff_9 = 2
low_cutoff_9 = 15
power = simpleLFP_signal.filter_signal(signal, despike=True, use_hilbert=True, threshold_value=None,
     low_cutoff=low_cutoff_9, high_cutoff=high_cutoff_9, order_low=5, order_high=5, notch=False)

power_env = hilbert(power)
amplitude = np.abs(power_env)

wavelet_amp = simpleLFP_spectral.wavelet_transform(power, wavelet_freq=range(0,20), plot=True)
