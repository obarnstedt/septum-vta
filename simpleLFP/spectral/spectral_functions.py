# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:35:11 2022

@author: kluxem
"""

import numpy as np
import quantities as pq
import elephant.signal_processing
from elephant.spectral import welch_psd
import matplotlib.pyplot as plt

class spectral_analysis():
    def __init__(self, recording_seconds=60.0):
        self.recording_seconds = recording_seconds
          
        
    def wavelet_transform(self, signal, n_cycles=9.0, wavelet_freq=np.arange(0,50,0.1).tolist(), plot=False): 
        sampling_frequency = signal.shape[0] / self.recording_seconds
        print("Sampling Frequency: ", sampling_frequency)
        
        wavelet_freq = wavelet_freq*pq.Hz
        wavelet_spec = elephant.signal_processing.wavelet_transform(signal, n_cycles=n_cycles, frequency=wavelet_freq, sampling_frequency=sampling_frequency)
        wavelet_spec = np.abs(wavelet_spec)
        
        # Plot Wavelet
        if plot == True:
            fig1, ax = plt.subplots()
            ax.imshow(wavelet_spec, origin='lower', aspect='auto', cmap='jet')
                
        return wavelet_spec
    
    def wavelet_transform_compare(self, signal1, signal2, n_cycles=9.0, wavelet_freq=np.arange(0,50,0.1).tolist(), plot=False):
        assert signal1.shape[0] == signal2.shape[0]
        
        sampling_frequency = signal1.shape[0] / self.recording_seconds
        print("Sampling Frequency: ", sampling_frequency)
        
        wavelet_freq = wavelet_freq*pq.Hz
        
        wavelet_spec_1 = elephant.signal_processing.wavelet_transform(signal1, n_cycles=n_cycles, frequency=wavelet_freq, sampling_frequency=sampling_frequency)
        wavelet_spec_1 = np.abs(wavelet_spec_1)
        
        wavelet_spec_2 = elephant.signal_processing.wavelet_transform(signal2, n_cycles=n_cycles, frequency=wavelet_freq, sampling_frequency=sampling_frequency)
        wavelet_spec_2 = np.abs(wavelet_spec_2)
        
        # Plot Wavelet
        if plot == True:
            plt.rcParams["figure.figsize"] = (20,10)
            fig1, ax = plt.subplots(1,2)
            ax[0].imshow(wavelet_spec_1, origin='lower', aspect='auto', cmap='jet')
            ax[1].imshow(wavelet_spec_2, origin='lower', aspect='auto', cmap='jet') 
            #TODO: Add colorbar
        
        return wavelet_spec_1, wavelet_spec_2
    
    
    def power_spectrum_density(self, signal):
        sampling_frequency = signal.shape[0] / self.recording_seconds
        freq, psd = welch_psd(signal, frequency_resolution=1, fs=sampling_frequency)

        return freq, psd[:60]
    
    
    
    def normalize_spectra(self, signal):
        pass
        
    
    
    
    
    
    
    