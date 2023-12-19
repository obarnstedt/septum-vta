# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:19:55 2022

@author: Kevin Luxem
"""

import numpy as np
import pandas as pd
from scipy.stats import iqr
from scipy import signal as dsp
from scipy.signal import hilbert
from scipy.signal import butter, lfilter

class signal_processing():
    def __init__(self, recording_seconds=60.0, notch_freq=50.0, downsample=None):
        self.recording_seconds = recording_seconds
        self.notch_freq = notch_freq
        self.downsample = downsample
        
    def _downsample_signal(self, signal):
        downsample_factor = int((len(signal) / self.downsample))
        signal_downsampled = signal[::downsample_factor]
        return signal_downsampled
    
    def _interpolation(self, signal, method="linear", order=None):
        signal_copy = signal.copy()
        signal_series = pd.Series(signal_copy)
        if np.isnan(signal_series[0]):
            signal_series[0] = next(x for x in signal_series if not np.isnan(x))
        signal_interpolation = signal_series.interpolate(method=method, order=order)
        return signal_interpolation
    
    def _butter_highpass(self, cutoff, sampling_freq, order=5):
        nyq = 0.5 * sampling_freq
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    
    def _butter_lowpass(self, cutoff, sampling_freq, order=5):
        nyq = 0.5 * sampling_freq
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def _butter_bandpass(self, cutoff, sampling_freq, order=5):
        nyq = 0.5 * sampling_freq
        low = cutoff[0] / nyq
        high = cutoff[1] / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def _exclude_broken_channel(self):
        pass
    
    def _iir_notch_filter(self, signal):
        """
        From: https://www.geeksforgeeks.org/design-an-iir-notch-filter-to-denoise-signal-using-python/
        """
        quality_factor = 20.0  # Quality factor
        sampling_frequency = signal.shape[0] / self.recording_seconds
        
        # Design a notch filter using signal.iirnotch
        b_notch, a_notch = dsp.iirnotch(self.notch_freq, quality_factor, sampling_frequency)
        
        # Apply notch filter to the noisy signal using signal.filtfilt
        outputSignal = dsp.filtfilt(b_notch, a_notch, signal)
        
        return outputSignal

    def _butter_highpass_filter(self, signal, cutoff, order=5):
        sampling_frequency = signal.shape[0] / self.recording_seconds
        b, a = self._butter_highpass(cutoff, sampling_frequency, order=order)
        y = lfilter(b, a, signal)
        return y

    def _butter_lowpass_filter(self, signal, cutoff, order=5):       
        sampling_frequency = signal.shape[0] / self.recording_seconds
        b, a = self._butter_lowpass(cutoff, sampling_frequency, order=order)
        y = lfilter(b, a, signal)
        return y
        
    def _butter_bandpass_filter(self, signal, cutoff, order=5): 
        sampling_frequency = signal.shape[0] / self.recording_seconds
        b, a = self._butter_bandpass(cutoff, sampling_frequency, order=order)
        y = lfilter(b, a, signal)
        return y
    
    def _simpleDespike(self, signal, factor=3, use_hilbert=False, threshold_value=None):
        """
        Simple thresholding method to despike a signal. Works suprisingly well!
        If you have a better/more robust method to detect and delete spikes, 
        PLEASE contribute! :)

        Parameters
        ----------
        signal : 1D FLOAT ARRAY
            A 1D timeseries with spikes.
        factor : INT, optional
            Factor by which the interquartile range (IQR) of the signal gets multiplied.
            The default is 3.
        use_hilbert : bool, optional
            If this is set to True, the analytical signal (envelope) of the signal is computed
            via the Hilber transformation. The analytical signal will be used for thresholding
            instead of the original signal, which leads to a more conservative despiking but
            can improve the signal to noise ratio        
        threshold_value : INT, optional
            Simple threshold. When exceeded this signal part is characterized as "spike".
            When set to None, the interquartile range (IQR) of the signal is computed and 
            multiplicated by 2 as threshold factor.
            The default is None.

        Returns
        -------
        signal: 1D FLOAT ARRAY.
            A despiked 1D timeseries.

        """        
        y = signal
        y_despike = signal.copy()
        
        if threshold_value == None:               
            iqr_val = iqr(y_despike)
            threshold = factor * iqr_val
            print("Using a threshold  value of %.4f. IQR: %.4f, Factor: %d" %(threshold, iqr_val, factor))
        else:
            threshold = threshold_value
            print("Using a threshold  value of %.4f." %(threshold))

        if use_hilbert == True:
            analytic_signal = hilbert(signal)
            signal = np.abs(analytic_signal)
        
        threshold_idx = []
        for n in range(len(y)-1):
            if signal[n] >= threshold:
                threshold_idx.append(n)
            if signal[n] <= -threshold:
                threshold_idx.append(n)
        
        threshold_idx = np.array(threshold_idx)            
        if len(threshold_idx) >= 1:             
            y_despike[threshold_idx] = np.nan
            y_despike = pd.Series.to_numpy(self._interpolation(y_despike))
        
        signal = y_despike
    
        return signal
    
    def filter_signal(self, signal, despike=False, use_hilbert=False, threshold_value=None,
                      low_cutoff=None, high_cutoff=None, order_low=5, order_high=5, notch=False):
        
        if self.downsample is not None:
            print("Downsampling signal to %.2f Hz" %self.downsample)
            signal = self._downsample_signal(signal)
            
        if despike == True:
            print("Despiking signal!")
            if use_hilbert == True:
                print("Applying hilbert transformation to detect outliers!")
            signal = self._simpleDespike(signal, factor=3, use_hilbert=use_hilbert, threshold_value=threshold_value)
            
        
        if low_cutoff != None and high_cutoff == None:
            print("Applying low_pass filter!")
            signal = self._butter_lowpass_filter(signal, cutoff=low_cutoff, order=order_low)
            
        if high_cutoff != None and low_cutoff == None:
            print("Applying high_pass filter!")
            signal = self._butter_highpass_filter(signal, cutoff=high_cutoff, order=order_low)
        
        if low_cutoff != None and high_cutoff != None:
            print("Applying band_pass filter!") #TODO: Revisit before publishing! 
            # signal = self._butter_bandpass_filter(signal, [low_cutoff,high_cutoff], order=order_low)
            signal = self._butter_highpass_filter(signal, cutoff=high_cutoff, order=order_low)
            signal = self._butter_lowpass_filter(signal, cutoff=low_cutoff, order=order_low)

        if notch == True:
            print("Applying notch flter at %.2f Hz" %self.notch_freq)
            signal = self._iir_notch_filter(signal)
            
        return signal
    