
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:37:36 2017

@author: DennisDa
"""

''' Importing Scripts '''
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import ddhelp
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import ddPlotting
import SubthresholdProperties as Sub
from scipy.signal import find_peaks_cwt as findpeaks
from collections import Counter
import pandas as pd


''' START OF THE SCRIPT: '''

''' Different Objects for Analysis ''' 
class FindAPs:
    def __init__ (self,Time,Wave,SampFreq,APThreshold):
        self.Time = Time # Time Only with Stimulus
        self.Wave = Wave # Wave Only with Stimulus
        self.SampFreq = SampFreq
        self.APThreshold = APThreshold
        
        self.WaveIndex = np.where(self.Wave >= self.APThreshold)
        self.WaveProcessed = self.Wave[self.WaveIndex]
        self.APIndices = findpeaks(self.WaveProcessed, np.arange(1,20))
        self.APTimesAlmost = self.Time[self.WaveIndex[0][self.APIndices]]-self.Time[0]
        self.APPeaksAlmost = self.Wave[self.WaveIndex[0][self.APIndices]]-self.Time[0]
        self.APNum = len(self.APIndices)       
        
        i = 0
        self.APDist = 1 # in ms!!!
        self.APIndiciesReal = [None]*self.APNum
        while i < self.APNum:
            self.APIndiciesReal[i] = np.argmax(self.Wave[int((self.APTimesAlmost[i]-self.APDist)*self.SampFreq):int((self.APTimesAlmost[i]+self.APDist)*self.SampFreq)])
            self.APIndiciesReal[i] = self.APIndiciesReal[i]+int((self.APTimesAlmost[i]-self.APDist)*self.SampFreq)
            i +=1

        self.APTimes = self.Time[self.APIndiciesReal]  
        self.APPeaks = self.Wave[self.APIndiciesReal]  
        self.APIntervals = np.diff(self.APTimes)

class FindAPs2:
    def __init__ (self,Time,Wave,SampFreq,Interval,Amplitude,WaveZeroedTo=-30):
        self.Time = Time # Time Only with Stimulus
        self.Wave = Wave # Wave Only with Stimulus
        self.SampFreq = SampFreq
        self.Interval = Interval*self.SampFreq #Interval in ms!
        self.Amplitude = Amplitude # min Amplitude difference from one peak to other
        self.ZeroedTo = WaveZeroedTo # Threshold
        
        # Threshold for Peak:          
         #+ self.WaveAmp
        if self.ZeroedTo < 0:
            self.Wave0 = np.ones(shape=(len(self.Wave),1))           
            i = 0
            while i < len(self.Wave0):
                self.Wave0[i] = self.Wave0[i]*self.ZeroedTo
                i += 1
        else:
            self.Wave0 = np.zeros(shape=(len(self.Wave),1))

        i = 0
        while i < len(self.Wave):
            if self.Wave[i] > self.ZeroedTo:
                self.Wave0[i] = self.Wave[i]
            i +=1

        # Create Search Windows:
        self.MaxPoints = len(self.Wave0)
        self.SubWindowSize = np.round(self.Interval/4)
        self.NumWindows = np.absolute(ddhelp.rounddown(self.MaxPoints/self.SubWindowSize))
        self.Windows = np.zeros(shape=(self.NumWindows,4))
        self.Windows [0,0] = int(0)
        self.Windows [0,1] = int(self.SubWindowSize)
        self.Windows [0,2] = int(2*self.SubWindowSize)
        self.Windows [0,3] = int(3*self.SubWindowSize)
        
        i = 1
        while i < self.NumWindows:
            k = 0
            while k < 4:
                self.Windows[i,k] = int(self.Windows[i-1,k] + self.SubWindowSize)
                k +=1
            i +=1
        
        self.Windows = np.delete(self.Windows,(np.where(self.Windows[:,3]>self.MaxPoints)),axis=0)
        
        self.Maxes = np.zeros(shape=(len(self.Windows),2))
        
        i = 0
        while i < len(self.Windows):
            if np.argmax(self.Wave0[int(self.Windows[i,0]):int(self.Windows[i,3])]) >= self.SubWindowSize and \
            np.argmax(self.Wave0[int(self.Windows[i,0]):int(self.Windows[i,3])]) < (3*self.SubWindowSize): # and \
            #np.absolute(np.max(self.Wave0[int(self.Windows[i,0]):int(self.Windows[i,3])]) - np.min(self.Wave0[int(self.Windows[i,0]):int(self.Windows[i,3])])) >= self.Amplitude :
                
                self.Maxes [i,0] = np.argmax(self.Wave0[int(self.Windows[i,0]):int(self.Windows[i,3])])+int(self.Windows[i,0])
                self.Maxes [i,1] = np.max(self.Wave0[int(self.Windows[i,0]):int(self.Windows[i,3])])
            i +=1
            
        self.Maxes = np.delete(self.Maxes,(np.where(self.Maxes[:,0]==0)),axis=0)
        u,self.UniqueIdx = np.unique(self.Maxes[:,0],return_index=True)
        self.APsMatrix = self.Maxes[self.UniqueIdx,:]
        self.APsMatrix = self.APsMatrix[self.APsMatrix[:,0].argsort()]
        
        i = 1
        #self.a[i]=np.nan(shape=(len(self.APsMatrix),1))
        while i < len(self.APsMatrix):
            if np.absolute (self.APsMatrix[i,0]-self.APsMatrix[i-1,0]) < self.Interval and \
            self.APsMatrix[i,1] > self.APsMatrix[i-1,1]:  
                self.APsMatrix = np.delete(self.APsMatrix,(i-1),axis=0)
            elif np.absolute (self.APsMatrix[i,0]-self.APsMatrix[i-1,0]) < self.Interval and \
            self.APsMatrix[i,1] < self.APsMatrix[i-1,1]:  
                self.APsMatrix = np.delete(self.APsMatrix,(i),axis=0)
            i +=1
        
        # Calculate Back:
        self.APTimes = self.APsMatrix[:,0]/self.SampFreq 
        self.APPeaks = self.APsMatrix[:,1]
#        self.APPeaks = [None]*len(self.APsMatrix[:,0])
#        i = 0
#        while i < len(self.APsMatrix[:,0]):
#            self.APPeaks = self.Wave[int(self.APsMatrix[i,0])]
#            i += 1
        self.APIntervals = np.diff(self.APTimes)
        self.APNum = len(self.APPeaks)

            
class Threshold:
    def __init__ (self,Time,Wave,SampFreq):        
        self.Time = Time # Time Only +/- something the Peak!
        self.Wave = Wave # Wave Only +/- something the Peak! 
        self.SampFreq = SampFreq    
        
        # Calculation Threshold:
        self.APSlope = np.diff(self.Wave)
        self.APSlopeMax = np.max(self.APSlope)
        self.APSlopeMax10 = 0.1*self.APSlopeMax
        self.FindWave = np.absolute(self.APSlope[0:np.argmax(self.APSlope)]-self.APSlopeMax10)
        self.ThresholdTime = self.Time[np.argmin(self.FindWave)+1]
        self.ThresholdVm = self.Wave[np.argmin(self.FindWave)+1]
        
        # Calculation Projected Threshold: 
        self.ThresProjectWave = np.absolute (self.Wave[np.argmax(self.Wave):] - self.ThresholdVm)
        self.ThresProjectTime = self.Time[np.argmin(self.ThresProjectWave)+np.argmax(self.Wave)]
        self.ThresProjectVm = self.ThresholdVm
        
class HalfWidth:
    def __init__ (self,Time,Wave,SampFreq):        
        self.Time = Time # Time Only Threshold to Mirrored Threshold
        self.Wave = Wave # Wave Only Threshold to Mirrored Threshold 
        self.SampFreq = SampFreq     
        
        self.HalfAmp = (np.max(self.Wave)- self.Wave[0])/2
        self.WaveCalc = self.Wave - self.Wave[0]
        
        self.Range = [None]*2
        self.HWTime = [None]*2
        self.HWVm = [None]*2

        self.Range[0] = np.argmin(np.absolute(self.WaveCalc[0:np.argmax(self.Wave)] - self.HalfAmp))
        self.Range[1] = np.argmin(np.absolute (self.WaveCalc[np.argmax(self.Wave):-1] - self.HalfAmp))
        self.HWTime[0]= self.Time[self.Range[0]]
        self.HWVm[0]= self.Wave[self.Range[0]]
        self.HWTime[1] = self.Time[(np.argmax(self.Wave)+self.Range[1])]
        self.HWVm[1] = self.Wave[self.Range[0]]
        #self.HWVm[1]= self.Wave[(np.argmax(Wave)+self.Range[1])]
        self.HW = self.HWTime[1]-self.HWTime[0]
    
    
class Slope:        
    def __init__ (self,Time,Wave,SampFreq):        
        self.Time = Time # Time Only Threshold to Peak or Peak to Thresmirrored
        self.Wave = Wave # Wave Only Threshold to Peak or Peak to Thresmirrored
        self.SampFreq = SampFreq    
        
        # Find 20 and 80%
        self.Range = [None]*2
        self.WaveCalc = self.Wave - self.Wave[0]
        self.Range[0]= np.argmin(np.absolute(self.WaveCalc - ((self.Wave[-1]-self.Wave[0])*0.2)))
        self.Range[1] = np.argmin(np.absolute (self.WaveCalc - ((self.Wave[-1]-self.Wave[0])*0.8)))
        self.Range.sort()

        # Slope as: m = (y2 - y1) / (x2 - x1)
        self.Slope = (self.Wave[self.Range[1]]-self.Wave[self.Range[0]])/(self.Time[self.Range[1]]-self.Time[self.Range[0]])

class AHP:
    def __init__ (self,Time,Wave,SampFreq,BaselineSD):   
        self.Time = Time # Time Only Mirrored Threshold to next AP or 300 ms
        self.Wave = Wave # Like Time + Normalised to VStable
        self.SampFreq = SampFreq     
        self.BaselineSD = BaselineSD
        self.FiltAHP,self.DivAHP = ddhelp.Deri.AHP(self.Time,self.Wave,self.SampFreq)
        
        # Predefine Variables:
        self.fAHPVm = np.nan
        self.fAHPTime = np.nan
        self.fAHPTP = np.nan
        self.ADPVm = np.nan
        self.ADPTime = np.nan
        self.ADPTP = np.nan
        self.mAHPVm = np.nan
        self.mAHPTime = np.nan
        self.mAHPTP = np.nan
            
        # Find Globale Maxima:
        self.GlobMaxVm = np.max(self.FiltAHP)
        self.GlobMaxTime = np.argmax(self.FiltAHP)/self.SampFreq
        #Zero all Global Maxima, that are nearer than 5 ms to boarder 
        #or smaller than 8times BaselineSD: 
        if self.GlobMaxTime <= 5 or self.GlobMaxVm <= 8*self.BaselineSD or self.GlobMaxTime > (len(self.Wave)/6)/self.SampFreq :
            self.GlobMaxVm = 0
            self.GlobMaxTime = 0
        
        # Find Global Minima
        self.GlobMinVm = np.min(self.FiltAHP)
        self.GlobMinTime = np.argmin(self.FiltAHP)/self.SampFreq
        self.ZDistance = int(1.7*self.SampFreq)
        
        # Test for fMAHP:
        if self.GlobMinTime  > 5:
            self.mAHPVm = self.GlobMinVm  
            self.mAHPTime = self.GlobMinTime
            self.mAHPTP = int(self.mAHPTime*self.SampFreq)
            
            # Potential ADP:
            self.LocalMaxVmTrace = self.FiltAHP[self.ZDistance:self.mAHPTP]
            self.LocalMaxTime = np.argmax(self.LocalMaxVmTrace)/self.SampFreq
            self.LocalMaxTime = self.LocalMaxTime+self.ZDistance/self.SampFreq
            self.LocalMaxTP = int(self.LocalMaxTime*self.SampFreq)
            self.LocalMaxVm = self.Wave[self.LocalMaxTP]
            # Zero all Local Maxima, that are nearer than 0.5 ms to boarder
            if self.LocalMaxTime < 0.5 or self.LocalMaxTime > (len(self.LocalMaxVmTrace)/self.SampFreq)-0.5:
                self.LocalMaxTP = 0
                self.LocalMaxVm = 0
                self.LocalMaxTime = 0
                
            # Potential fAHP
            self.LocalMinVmTrace = self.FiltAHP[0:self.LocalMaxTP+self.ZDistance]
            self.LocalMinTime = np.argmin(self.LocalMinVmTrace)/self.SampFreq
            self.LocalMinTime = self.LocalMinTime
            self.LocalMinTP = int(self.LocalMinTime*self.SampFreq)
            self.LocalMinVm = self.Wave[self.LocalMinTP]
            # Zero all Local Maxima, that are nearer than 0.5 ms to boarder 
            if self.LocalMinTime < 0.35 or self.LocalMinTime > (len(self.LocalMinVmTrace)/self.SampFreq)-0.5:
                self.LocalMinTP = 0
                self.LocalMinVm = 0
                self.LocalMinTime = 0
                
        # Test for FmAHP:
        if self.GlobMinTime  < 5:
            self.fAHPVm = self.GlobMinVm  
            self.fAHPTime = self.GlobMinTime
            self.fAHPTP = int(self.fAHPTime*self.SampFreq) 
            
            # Potential ADP:
                # First Searchwindow 10 ms, then Searchwindow 15 ms
            # Search Windows consistant with lens of Trace?:
            self.SearchADPW1 = self.fAHPTP + int(10*SampFreq)
            self.SearchADPW2 = self.fAHPTP + int(15*SampFreq)
            if self.SearchADPW1 > len(self.FiltAHP):
                self.SearchADPW1 = -1
            if self.SearchADPW2 > len(self.FiltAHP):
                self.SearchADPW2 = -1    
                
            self.LocalMaxVmTrace = self.FiltAHP[self.fAHPTP:self.SearchADPW1]
            self.LocalMaxTime = np.argmax(self.LocalMaxVmTrace)/self.SampFreq
            self.LocalMaxTime = self.LocalMaxTime+self.fAHPTime
            self.LocalMaxTP = int(self.LocalMaxTime*self.SampFreq)
            self.LocalMaxVm = self.Wave[self.LocalMaxTP]
            if self.LocalMaxTime < 0.5 or self.LocalMaxTime > (len(self.LocalMaxVmTrace)/self.SampFreq)-1.75:
                self.LocalMaxTP = 0
                self.LocalMaxVm = 0
                self.LocalMaxTime = 0
                
                self.LocalMaxVmTrace = self.FiltAHP[self.fAHPTP:self.SearchADPW2]
                self.LocalMaxTime = np.argmax(self.LocalMaxVmTrace)/self.SampFreq
                self.LocalMaxTime = self.LocalMaxTime+self.fAHPTime
                self.LocalMaxTP = int(self.LocalMaxTime*self.SampFreq)
                self.LocalMaxVm = self.Wave[self.LocalMaxTP]
                
                if self.LocalMaxTime < 0.5 or self.LocalMaxTime > (len(self.LocalMaxVmTrace)/self.SampFreq)-1.75:
                    self.LocalMaxTP = 0
                    self.LocalMaxVm = 0
                    self.LocalMaxTime = 0
                    
            # Potential mAHP:
            self.LocalMinVmTrace = self.FiltAHP[self.LocalMaxTP:-1]
            #self.LocalMinVmTraceVm = self.Wave[self.LocalMaxTP:-1]
            self.LocalMinTime = np.argmin(self.LocalMinVmTrace)/self.SampFreq
            self.LocalMinTime = self.LocalMinTime+self.LocalMaxTime
            self.LocalMinTP = int(self.LocalMinTime*self.SampFreq)
            self.LocalMinVm = self.Wave[self.LocalMinTP]
            
            if self.LocalMinTime < 0.5 or self.LocalMinTime > (len(self.LocalMinVmTrace)/self.SampFreq)-0.5:
                self.LocalMinTP = 0
                self.LocalMinVm = 0
                self.LocalMinTime = 0
       
        # Criteria to form fAHP, ADP, mAHP with PeakReal!, TPreal!, TimeToPeak, 
        # further: Amplitude
        if not np.isnan(self.fAHPVm):
            #print(self.fAHPVm)
            if np.nonzero(self.LocalMaxVm) and np.absolute(self.LocalMaxVm-self.LocalMinVm) > (3/4)*self.BaselineSD:
                self.ADPVm = self.LocalMaxVm
                self.ADPTP = self.LocalMaxTP
                self.ADPTime = self.LocalMaxTime
            elif np.nonzero(self.GlobMaxVm) and np.absolute(self.GlobMaxVm) > 8*self.BaselineSD:
                self.ADPVm = self.GlobMaxVm
                self.ADPTP = self.GlobMaxTP
                self.ADPTime = self.GlobMaxTime
            if np.nonzero(self.LocalMinVm) and np.absolute(self.LocalMaxVm-self.LocalMinVm) > self.BaselineSD and np.absolute(self.fAHPVm) > 0:
                self.mAHPVm = self.LocalMinVm
                self.mAHPTP = self.LocalMinTP
                self.mAHPTime = self.LocalMinTime
            if np.isnan(self.ADPVm) or np.isnan(self.mAHPVm) or  self.fAHPTP > self.ADPTP or self.LocalMinTP < self.fAHPTP or (np.absolute(self.mAHPVm) < self.BaselineSD and np.absolute(self.fAHPVm) < 0):
                self.ADPVm = np.nan
                self.ADPTime = np.nan
                self.ADPTP = np.nan
                self.mAHPVm = np.nan
                self.mAHPTime = np.nan
                self.mAHPTP = np.nan    
            if self.mAHPTime > 200:
                self.mAHPVm = np.nan
                self.mAHPTime = np.nan
                self.mAHPTP = np.nan  
            if self.ADPVm < 8 * self.BaselineSD and (np.isnan(self.mAHPVm) or not np.nonzero(self.mAHPVm)):
                self.ADPVm = np.nan
                self.ADPTime = np.nan
                self.ADPTP = np.nan
                
        elif not np.isnan(self.mAHPVm): 
            if np.nonzero(self.LocalMaxVm) and np.absolute(self.LocalMaxVm - self.LocalMinVm) > self.BaselineSD or self.LocalMaxTP < self.mAHPTP:
                self.ADPVm = self.LocalMaxVm
                self.ADPTime = self.LocalMaxTime
                self.ADPTP = self.LocalMaxTP
            if self.LocalMinVm != 0 and np.absolute(self.LocalMaxVm - self.LocalMinVm) > self.BaselineSD or self.LocalMinTP < self.mAHPTP:
                self.fAHPVm = self.LocalMinVm
                self.fAHPTime = self.LocalMinTime
                self.fAHPTP = self.LocalMinTP
            if self.LocalMinVm == 0 or self.LocalMaxVm == 0 or self.fAHPTP > self.ADPTP:
                self.fAHPVm = np.nan
                self.fAHPTime = np.nan
                self.fAHPTP = np.nan   
                self.ADPVm = np.nan
                self.ADPTime = np.nan
                self.ADPTP = np.nan
                
        # Naming:
        if not np.isnan(self.fAHPVm) and np.isnan(self.ADPVm) and np.isnan(self.mAHPVm):
            self.AHPType = 'fAHP'
        elif not np.isnan(self.mAHPVm) and np.isnan(self.ADPVm) and np.isnan(self.fAHPVm):
            self.AHPType = 'mAHP'
        elif not np.isnan(self.ADPVm) and np.isnan(self.mAHPVm) and np.isnan(self.fAHPVm): 
            self.AHPType = 'ADP'
        elif not np.isnan(self.ADPVm) and not np.isnan(self.mAHPVm) and not np.isnan(self.fAHPVm) and self.fAHPVm < self.mAHPVm:
            self.AHPType = 'FmAHP'
        elif not np.isnan(self.ADPVm) and not np.isnan(self.mAHPVm) and not np.isnan(self.fAHPVm) and self.mAHPVm < self.fAHPVm:
            self.AHPType = 'fMAHP'
#        else:
#            self.AHPType = '?'

        # Repolarisation Index:
        self.RepoIndexMin = (np.min(self.Wave)*0.1)
        self.RepoWaveIdx = 0
        self.RepoWaveIdx = np.where((self.Wave-self.RepoIndexMin) < 0)
        self.RepoMin = np.min(self.Wave)
        if not self.RepoWaveIdx == 0 and len(self.RepoWaveIdx[0]) > 2:
            self.RepoTime = self.Time[self.RepoWaveIdx[0][0]:self.RepoWaveIdx[0][-1]]
            self.RepoWave = self.Wave[self.RepoWaveIdx[0][0]:self.RepoWaveIdx[0][-1]]
            self.RepoDuration = self.RepoTime[-1]-self.RepoTime[0]
            self.RepoPotential = np.mean(self.RepoWave)
            self.RepoIndex = self.RepoPotential/self.RepoDuration
            
        else:
            self.RepoIndex = np.nan

class AHPEnd:
    def __init__ (self,Time,Wave,AHPPoint,SampFreq,BaselineSD):   
        self.Time = Time # Time Only Mirrored Threshold StimOff
        self.Wave = Wave # Like Time + Normalised to VStable (= mean last 10ms)
        self.AHPPoint = AHPPoint     
        self.SampFreq = SampFreq
        self.BaselineSD = BaselineSD  
        self.AHPTime = (len(self.Time)/self.SampFreq)/50
        
        
            # Get VStable; Trace from AHPPoint to end: 
        self.AHPIdx = np.argmin(np.absolute(self.Time-self.AHPPoint))
        self.TimeVStable = self.Time[self.AHPIdx:-1]
        self.WaveVStable = self.Wave[self.AHPIdx:-1]
            
            # Define VStable by minimal Increment
        if len(self.Time) > 100*self.SampFreq:
            self.WaveVStableFilt = ddhelp.Filters.MovingAverage(self.WaveVStable, window_len=int(50 * self.SampFreq), window='hanning')
            self.WaveVStableDiff = np.diff(self.WaveVStableFilt)
            Threshold = (np.std(self.WaveVStableDiff))/18
            self.WaveStableZeros, = np.where(np.absolute(self.WaveVStableDiff)<=Threshold);
            self.LengthWaveStableZeros = len(self.WaveStableZeros)
            if not self.LengthWaveStableZeros:
                self.LengthWaveStableZeros = 1
        
        # Calculate VStable as mean from points with zero increment and AHP Time: 
            self.VStableCalc = np.mean(self.WaveVStable[int(-self.AHPTime*self.LengthWaveStableZeros):-1])
        else:
            self.VStableCalc = np.mean(self.WaveVStable[-int(len(self.WaveVStable)/6):-1]) 
            self.VStablePre = [52]
        
        self.VStableCurve = np.full((len(self.TimeVStable)), self.VStableCalc)
        # Get Time To VStable as Schnittmenge with VStable:
        self.VStablePre, = np.where(np.absolute(self.WaveVStable-self.VStableCalc)<=10**-1);
        if len(self.VStablePre) <= 0:
            self.VStablePre = [len(self.WaveVStable)]

        if self.VStablePre[0]<50:
            self.WhereBigger, = np.where(self.VStablePre > 50)
            if len(self.WhereBigger) > 0:
                self.VStablePre[0] = self.VStablePre[self.WhereBigger[0]]
            else:
                self.VStablePre[0]=self.VStablePre[-1]
                

        self.VStableTime = (self.VStablePre[0]+self.AHPIdx)/self.SampFreq

        # Repolarisation Area: 
        # Start: When Diff is minimum
        self.WaveStartDiff = np.diff(self.Wave[0:self.AHPIdx])
        self.WaveStartSD = np.std(self.WaveStartDiff[-100:-1])
        self.WaveStartPre, = np.where(np.absolute(self.WaveStartDiff)<=self.WaveStartSD)
        if len(self.WaveStartPre) >=1:
            self.WaveStart = self.WaveStartPre[0]
        else:
            self.WaveStart = 0
        # End: Point to VStable: 
        self.VStablePoint = self.VStablePre[0]+self.AHPIdx
        self.AreaTime = self.Time[self.WaveStart:self.VStablePoint]
        self.AreaWave = self.Wave[self.WaveStart:self.VStablePoint]-self.VStableCalc
        self.AreaVec = self.VStableCurve = np.full((len(self.AreaTime)), self.VStableCalc)
        self.Area = sp.integrate.simps(self.AreaWave,self.AreaTime)
        
        # if Latency > 400 ms no time for AHP development:
        if len(self.Time) < 100*self.SampFreq:
            self.Cave = 1
        else:
            self.Cave = 0
            
#        print('area:',self.Area)
        # Output:
#        print('area:',self.Area,'mV^2/ms')
#        print('VStableTime:',self.VStableTime,'ms')
#        print('AreaVec:',self.AreaVec)
#        plt.figure()
#        plt.ion()
#        plt.plot(self.Time,self.Wave)
#        plt.plot(self.AreaTime,(self.AreaWave+self.VStableCalc))
##        plt.plot(self.AreaTime,self.AreaVec)
#        plt.fill_between(self.AreaTime,self.AreaVec,self.AreaWave+self.VStableCalc)
#        plt.plot(self.TimeVStable[self.VStablePre[0]],self.WaveVStable[self.VStablePre[0]],'o')
##        plt.plot(self.TimeVStable,self.VStableCurve)
#        plt.plot(self.Time[self.WaveStart],self.Wave[self.WaveStart],'o')
#        a
        
class FiringAdaption:
    def __init__(self,TimePoints,Wave,SampFreq,UseLinearisedFit):
        self.Time = TimePoints
        self.Wave = Wave
        self.SampFreq = SampFreq
        self.UseLinFit = UseLinearisedFit # 0 or 1 
        self.BaselinePositions = [np.argmin(self.Wave)]
        
        # Potential Slow Adaption; Monoexponential Fit:    
        self.Scale = np.max(self.Wave)-self.Wave[self.BaselinePositions[0]]
        self.FitStart = [self.Scale,-0.001,0]
        
        if self.UseLinFit == 1:
            self.APBaseAdaptionFitValues = ddhelp.FitMonoExLinearised(self.Time,self.Wave)
            self.APBaseAdaptionFitValues.FitParams[1] = self.APBaseAdaptionFitValues.FitParams[0]
        else:
            self.APBaseAdaptionFitValues = ddhelp.FitMonoEx(self.Time,self.Wave,self.BaselinePositions,self.FitStart)
           
            if self.APBaseAdaptionFitValues.FitParams[1] == -0.001 or self.APBaseAdaptionFitValues.r < 0.999 or self.APBaseAdaptionFitValues.FitParams[1] < -1:
                self.FitStart = [self.Scale,-0.025,0]  
                self.APBaseAdaptionFitValues = ddhelp.FitMonoEx(self.Time,self.Wave,self.BaselinePositions,self.FitStart)
    #            print('1')
            if self.APBaseAdaptionFitValues.FitParams[1] == -0.025  or self.APBaseAdaptionFitValues.r < 0.99 or self.APBaseAdaptionFitValues.FitParams[1] < -1:
                self.FitStart = [self.Scale,-0.05,0]
                self.APBaseAdaptionFitValues = ddhelp.FitMonoEx(self.Time,self.Wave,self.BaselinePositions,self.FitStart)
    #            print('2')
            if self.APBaseAdaptionFitValues.FitParams[1] == -0.05 or self.APBaseAdaptionFitValues.r < 0.99 or self.APBaseAdaptionFitValues.FitParams[1] < -1:
                self.FitStart = [self.Scale,-0.1,0]   
                self.APBaseAdaptionFitValues = ddhelp.FitMonoEx(self.Time,self.Wave,self.BaselinePositions,self.FitStart)
    #            print('3')
            if self.APBaseAdaptionFitValues.FitParams[1] == -0.1 or self.APBaseAdaptionFitValues.r < 0.99 or self.APBaseAdaptionFitValues.FitParams[1] < -1:
                self.APBaseAdaptionFitValues = ddhelp.FitMonoExLinearised(self.Time,self.Wave)
                self.APBaseAdaptionFitValues.FitParams[1] = self.APBaseAdaptionFitValues.FitParams[0]
    #            print('4')
    #        if self.APBaseAdaptionFitValues.r < 0.99 or self.APBaseAdaptionFitValues.FitParams[1] < -1:
    #            self.APBaseAdaptionFitValues.FitParams[1] = np.nan
    #            self.APBaseAdaptionFitValues.r = np.nan
    #            self.APBaseAdaptionFitValues.FitCurve = np.zeros([len(self.Time),1])

        
        self.SlowAdaptionFit = self.APBaseAdaptionFitValues.FitParams[1]
        self.SlowAdaptionFitr = self.APBaseAdaptionFitValues.r
        self.SlowAdaptionFitCurveT = self.Time
        self.SlowAdaptionFitCurve = self.APBaseAdaptionFitValues.FitCurve


        # Potential Fast Adaption as 3*SD from Fit Curve for Amplitudes:
        self.FitCurveMean = np.mean(self.SlowAdaptionFitCurve)   
        self.DistFitCurve = self.Wave - self.SlowAdaptionFitCurve
        self.DistFitCurveSD = np.std(self.DistFitCurve)
        self.PotDist1 = np.where((self.Wave-self.SlowAdaptionFitCurve) < -3 * self.DistFitCurveSD)
        if hasattr(self, 'PotDist1') and len(self.PotDist1[0]) > 0:
            self.PotDist = self.PotDist1[0]
            if 0 < self.PotDist[0] < 3 and self.Wave[self.PotDist[0]]-self.SlowAdaptionFitCurve[self.PotDist[0]] < -4:

                self.FastAdaptionPoint = np.empty([1, 2])
                self.FastAdaptionPoint[0,0] = self.Time[self.PotDist[0]]
                self.FastAdaptionPoint[0,1] = self.Wave[self.PotDist[0]]
                self.FastAdaptionIndex = self.Wave[self.PotDist[0]]/self.Wave[0]
            else:
                self.FastAdaptionPoint = np.empty([1, 2])
                self.FastAdaptionPoint[0,0] = np.nan
                self.FastAdaptionPoint[0,1] = np.nan
                self.FastAdaptionIndex = 1
        else:
            self.FastAdaptionPoint = np.empty([1, 2])
            self.FastAdaptionPoint[0,0] = np.nan
            self.FastAdaptionPoint[0,1] = np.nan
            self.FastAdaptionIndex = 1
        
        # FiringChangeIndex:
        self.ChangeIndex = (self.Wave[-1]/self.Wave[0])*100

class FiringAdaption2:
    def __init__(self,TimePoints,Wave,SampFreq):
        self.Time = TimePoints
        self.Wave = Wave
        self.SampFreq = SampFreq
        
        # General Firing-Change-Index:
        self.ChangeIndex = (self.Wave[-1]/self.Wave[0])*100        
        
        # Look for Positive Value in Deviation and Divide Trace in Fast and Slow Part: 
        self.WaveDiff = np.diff(self.Wave)
#        self.otSeparation = [None]
        self.PotSeparation = [np.argmax(self.WaveDiff[0:int(len(np.round((self.Wave)/2)))])]
#        plt.ion()
#        A = plt.figure()
#        plt.subplot(1,2,1)
#        plt.plot(self.Wave)
#        plt.subplot(1,2,2)
#        plt.plot(self.WaveDiff)
#        plt.show(A)
        # ALTERNATIVE:
#        self.AlternativeSeparationMatrix, = np.where(self.WaveDiff > 0)
#        if len(self.AlternativeSeparationMatrix)>1:
#            self.AlternativeSeparationMatrix = self.AlternativeSeparationMatrix[self.AlternativeSeparationMatrix <= 5]
#        if len(self.AlternativeSeparationMatrix) > 2:
#            self.AlternativeSeparation = [np.max(self.AlternativeSeparationMatrix)]
#        else:
#            self.AlternativeSeparation = [0]
#        # Alternative, when PotSeparation wrong and other is ok:
#        if self.PotSeparation[0] <= 1 or  self.PotSeparation[0]>len(self.Wave)-3 or \
#        self.PotSeparation[0] > len(self.Wave)/3:
#            print('PotSep wrong')
#            if self.AlternativeSeparation[0] >= 1 and  self.AlternativeSeparation[0]<len(self.Wave)-3 and \
#            self.AlternativeSeparation[0] <= len(self.Wave)/3:
#                self.PotSeparation = [self.AlternativeSeparation[0]]           
#        print(self.PotSeparation)
        
        if self.PotSeparation[0] < 1 or  self.PotSeparation[0]>len(self.Wave)-3 or \
        self.PotSeparation[0] >= int(np.round(len(self.Wave)/3)+1): # or self.PotSeparation[0] is not None: # or np.max(self.Wave) < 0:
            self.PotSeparation = [int(np.round(len(self.Wave)/3))]    
#            print('CANGED')
#            print(self.PotSeparation[0],int(np.round(len(self.Wave)/3)+1))
#            print(self.PotSeparation[0] >= int(np.round(len(self.Wave)/3)+1))
        
        if self.PotSeparation[0] >= 1  and \
        len(self.Wave) > 3:# and self.PotSeparation[0] <= len(self.Wave)/3: #and  self.PotSeparation[0]<len(self.Wave)-3
                        
            self.SecWave = self.Wave[self.PotSeparation[0]+1:]
            self.SecTime = self.Time[self.PotSeparation[0]+1:]
            if self.SecWave[0] < np.mean(self.SecWave):
                self.NewPotSep = np.argmax(self.SecWave) + self.PotSeparation[0]+1
                if self.NewPotSep < len(self.Wave)-3:
                    self.SecWave = self.Wave[self.NewPotSep:]
                    self.SecTime = self.Time[self.NewPotSep:]
            self.FirstWave = self.Wave[0:self.PotSeparation[0]+1]
            self.FirstTime = self.Time[0:self.PotSeparation[0]+1] 
            
            # set Time To Zero:
            self.FirstTimeDelta = self.FirstTime[0]
            self.FirstTime = self.FirstTime-self.FirstTimeDelta
            self.SecTimeDelta = self.SecTime[0]
            self.SecTime = self.SecTime-self.SecTimeDelta
            
            ##Fitting first(fast)Part:
            self.FastFitValues = ddhelp.FitMonoExLinearised(self.FirstTime,self.FirstWave)#,self.BaselinePositions,self.FitStart)
            self.FastAdaptionFitr = self.FastFitValues.r
            self.FastAdaptionFitCurveT = self.FirstTime
            self.FastAdaptionFitCurve = self.FastFitValues.FitCurve
            self.FastAdaptionFit = self.FastFitValues.FitParams[0]
            
#            if self.FastAdaptionFitr < 0.9: 
#                self.BaselinePositions = [np.argmin(self.FirstWave)]
#                self.Scale = np.max(self.FirstWave)-self.FirstWave[self.BaselinePositions[0]]
#                self.FitStart = [self.Scale,-0.001,0]
#                self.FastFitValues = ddhelp.FitMonoEx(self.FirstTime,self.FirstWave,self.BaselinePositions,self.FitStart)
#                if self.FastFitValues.r > 0.9:
#                    self.FastAdaptionFitr = self.FastFitValues.r
#                    self.FastAdaptionFitCurveT = self.FirstTime
#                    self.FastAdaptionFitCurve = self.FastFitValues.FitCurve
#                    self.FastAdaptionFit = self.FastFitValues.FitParams[1]

            # Fitting last(slow)Part:
            self.SlowFitValues = ddhelp.FitMonoExLinearised(self.SecTime,self.SecWave)#,self.BaselinePositions,self.FitStart)
            self.SlowAdaptionFitr = self.SlowFitValues.r
            self.SlowAdaptionFitCurveT = self.SecTime
            self.SlowAdaptionFitCurve = self.SlowFitValues.FitCurve
            self.SlowAdaptionFit = self.SlowFitValues.FitParams[0]
#            if self.SlowAdaptionFitr < 0.9: 
#                self.BaselinePositions = [np.argmin(self.SecWave)]
#                self.Scale = np.max(self.SecWave)- self.SecWave[self.BaselinePositions[0]]
#                self.FitStart = [self.Scale,-0.001,0]
#                self.SlowFitValues = ddhelp.FitMonoEx(self.SecTime,self.SecWave,self.BaselinePositions,self.FitStart)
#            
#                if self.SlowFitValues.r > 0.9:
#                    self.SlowAdaptionFitr = self.SlowFitValues.r
#                    self.SlowAdaptionFitCurveT = self.SecTime
#                    self.SlowAdaptionFitCurve = self.SlowFitValues.FitCurve
#                    self.SlowAdaptionFit = self.SlowFitValues.FitParams[1]
                    
            # Getting Fits-Times back:    
            self.FastAdaptionFitCurveT = self.FastAdaptionFitCurveT + self.FirstTimeDelta
            self.SlowAdaptionFitCurveT = self.SlowAdaptionFitCurveT + self.SecTimeDelta
            
        # If No Slow Part just fit Wave as Slow Wave:
        else:
            self.SeparatedWaves = 0 
            # First: Linearised:
            self.SlowFitValues = ddhelp.FitMonoExLinearised(self.Time,self.Wave)#,self.BaselinePositions,self.FitStart)
            self.SlowAdaptionFitr = self.SlowFitValues.r
            self.SlowAdaptionFitCurveT = self.Time
            self.SlowAdaptionFitCurve = self.SlowFitValues.FitCurve
            self.SlowAdaptionFit = self.SlowFitValues.FitParams[0]
            if self.SlowAdaptionFitr < 0.9: 
                self.BaselinePositions = [np.argmin(self.Wave)]
                self.Scale = np.max(self.Wave)-self.Wave[self.BaselinePositions[0]]
                self.FitStart = [self.Scale,-0.001,0]
                self.SlowFitValues = ddhelp.FitMonoEx(self.Time,self.Wave,self.BaselinePositions,self.FitStart)
                if self.SlowFitValues.r > 0.9:
                    self.SlowAdaptionFitr = self.SlowFitValues.r
                    self.SlowAdaptionFitCurveT = self.Time
                    self.SlowAdaptionFitCurve = self.SlowFitValues.FitCurve
                    self.SlowAdaptionFit = self.SlowFitValues.FitParams[1]

            self.FastAdaptionFitr = np.nan
            self.FastAdaptionFitCurveT = np.nan
            self.FastAdaptionFitCurve = np.nan
            self.FastAdaptionFit = np.nan


''' Main Script Action Potentials Properties: '''
class MainAP:    
    def __init__ (self,Names,Time,Wave,Stimulus,SampFreq,PrintShow = 0):
        self.WaveNames = Names
        self.Time = Time
        self.Wave = Wave
        self.Stimulus = Stimulus
        self.SampFreq = SampFreq
        self.PrintShow = PrintShow
        
        # Define StimOn- StimOffset
        self.StimDiffWave = np.diff(self.Stimulus)
        self.StimDiffPoints = np.where(self.StimDiffWave != 0)
        self.StimOnset = np.asarray(self.StimDiffPoints[0][0])
        self.StimOffset = np.asarray(self.StimDiffPoints[0][1])
        self.StimOnsetTime = self.StimOnset/SampFreq
        self.StimOffTime = self.StimOffset/SampFreq
        
        #Calculation Basline
        self.Baseline = np.mean(self.Wave[0:(self.StimOnset)])
        self.BaselineSD = np.std(self.Wave[0:(self.StimOnset)])
        self.BaselineP = np.zeros((2,self.StimOnset))
        self.BaselineP[0] = self.Time[0:self.StimOnset]
        self.BaselineP[1]= self.Baseline
        
        #Calculation Stable Response
        self.VStable = np.mean(self.Wave[(self.StimOffset-int(100*SampFreq)):(self.StimOffset)])
        self.VStableP = np.zeros((2,int(100*self.SampFreq)))
        self.VStableP[0] = self.Time[(self.StimOffset-int(100*self.SampFreq)):(self.StimOffset)]
        self.VStableP[1]= self.VStable
        self.VStableAmplitude = self.VStable - self.Baseline
        
        # Find Action Potentials Peaks
        self.APPeakThreshold = -10
        self.FindAPsTime = self.Time[self.StimOnset:self.StimOffset]
        self.FindAPsWave = self.Wave[self.StimOnset:self.StimOffset]
        self.APs = FindAPs2(self.FindAPsTime,self.FindAPsWave,self.SampFreq,3,1,self.APPeakThreshold) 
        self.APTimes = self.APs.APTimes+self.StimOnsetTime
        self.APpeaks = self.APs.APPeaks
        self.InstFiringFreq = (1/self.APs.APIntervals)*1000

        if np.size(self.APpeaks) == 0:
            self.APPeakThreshold = -30
            self.FindAPsTime = self.Time[self.StimOnset:self.StimOffset]
            self.FindAPsWave = self.Wave[self.StimOnset:self.StimOffset]
            self.APs = FindAPs2(self.FindAPsTime,self.FindAPsWave,self.SampFreq,2,1,self.APPeakThreshold) 
            self.APTimes = self.APs.APTimes+self.StimOnsetTime
            self.APpeaks = self.APs.APPeaks
            self.InstFiringFreq = (1/self.APs.APIntervals)*1000
            
        if any(self.InstFiringFreq >= 500):
            print('APChanged')
            self.APPeakThreshold = -10
            self.FindAPsTime = self.Time[self.StimOnset:self.StimOffset]
            self.FindAPsWave = self.Wave[self.StimOnset:self.StimOffset]
            self.APs = FindAPs2(self.FindAPsTime,self.FindAPsWave,self.SampFreq,10,30,self.APPeakThreshold) 
            self.APTimes = self.APs.APTimes+self.StimOnsetTime
            self.APpeaks = self.APs.APPeaks
            self.InstFiringFreq = (1/self.APs.APIntervals)*1000
            self.APNum = self.APs.APNum
            
        self.APTime = self.APTimes[0]
        self.APPeak = self.APpeaks[0]
        self.BurstAPs = 0
                    
        # Test Figure:
#        plt.figure()
#        plt.plot(self.Time,self.Wave)
#        plt.plot(self.APTimes,self.APpeaks,'o')
#        plt.show()
#        a
        
        # Burst Calculations:
        self.BurstDefinition = 40
        if any(self.InstFiringFreq >= 40):
            self.Burst = 1
            self.APType = 'Burst'
            self.BurstAPList = np.where(self.InstFiringFreq >= self.BurstDefinition)
            self.ToMuchBurst, = np.where(np.diff(self.InstFiringFreq)>=self.BurstDefinition)
            self.BurstAPs = len(self.BurstAPList[0])-len(self.ToMuchBurst)
            self.NumAPs = self.BurstAPs+1
            self.APTime = self.APTimes[0:(self.BurstAPs+1)]
            self.APPeak = self.APpeaks[0:(self.BurstAPs+1)]

            # Continoue Burst Analysis:
            i = 0
            self.ThresTime1 = [None]*(self.BurstAPs+1)
            self.ThresholdTimes = [None]*(self.BurstAPs+1)
            self.ThresholdWaves = [None]*(self.BurstAPs+1)
            self.ThresholdValues = [None]*(self.BurstAPs+1)
            self.ThresholdTime1 = [None]*(self.BurstAPs+1)
            self.ThresholdVm1 = [None]*(self.BurstAPs+1)
            self.ThresholdProjectTime1 = [None]*(self.BurstAPs+1)
            self.ThresholdProjectVm1 = [None]*(self.BurstAPs+1)
            
            self.RiseTime = [None]*(self.BurstAPs+1)
            self.RiseWave = [None]*(self.BurstAPs+1)
            self.DecayTime = [None]*(self.BurstAPs+1)
            self.DecayWave = [None]*(self.BurstAPs+1)
            self.RiseValues = [None]*(self.BurstAPs+1)
            self.SlopeRise1 = [None]*(self.BurstAPs+1)
            self.DecayValues = [None]*(self.BurstAPs+1)
            self.SlopeDecay1 = [None]*(self.BurstAPs+1)
            
            self.HWTime = [None]*(self.BurstAPs+1)
            self.HWWave = [None]*(self.BurstAPs+1)
            self.HWValues = [None]*(self.BurstAPs+1)
            self.HalfWidth1 = [None]*(self.BurstAPs+1)
            
            self.APAmpBaseline1 = [None]*(self.BurstAPs+1)
            self.APAmpThres1 = [None]*(self.BurstAPs+1)
            self.TimeToPeak1 = [None]*(self.BurstAPs+1)
            self.APLatency1 = [None]*(self.BurstAPs+1)
            self.HWPlotTime = [None]*(self.BurstAPs+1)
            self.HWPlotVm = [None]*(self.BurstAPs+1)
            
            while i <= self.BurstAPs:
                # Threshold:
                self.ThresTimeWindowStart = 1.5
                self.ThresTimeWindowEnd = 2
                self.ThresTime1[i] = int(self.APTime[i]*self.SampFreq)
                self.ThresTime = [None]*2
                self.ThresTime[0] = int(self.ThresTime1[i] - (self.ThresTimeWindowStart*self.SampFreq))
                self.ThresTime[1] = int(self.ThresTime1[i] + (self.ThresTimeWindowEnd*self.SampFreq))
                self.ThresholdTimes[i] = self.Time[self.ThresTime[0]:self.ThresTime[1]]
                self.ThresholdWaves[i] = self.Wave[self.ThresTime[0]:self.ThresTime[1]] 
                self.ThresholdValues[i] = Threshold(self.ThresholdTimes[i],self.ThresholdWaves[i],self.SampFreq)
                
                self.ThresholdTime1[i] = self.ThresholdValues[i].ThresholdTime
                self.ThresholdVm1[i] = self.ThresholdValues[i].ThresholdVm
                self.ThresholdProjectTime1[i] = self.ThresholdValues[i].ThresProjectTime
                self.ThresholdProjectVm1[i] = self.ThresholdValues[i].ThresProjectVm
                
                # Slopes:
                self.RiseT = [None]*2
                self.RiseT[0] = int(self.ThresholdTime1[i]*self.SampFreq)
                self.RiseT[1] = int(self.APTimes[i]*self.SampFreq)
                self.RiseTime[i] = self.Time[self.RiseT[0]:self.RiseT[1]]
                self.RiseWave[i] = self.Wave[self.RiseT[0]:self.RiseT[1]]
                self.DecayT = [None]*2
                self.DecayT[0] = int(self.APTimes[i]*self.SampFreq)
                self.DecayT[1] = int(self.ThresholdProjectTime1[i]*self.SampFreq)
                self.DecayTime[i] = self.Time[self.DecayT[0]:self.DecayT[1]]
                self.DecayWave[i] = self.Wave[self.DecayT[0]:self.DecayT[1]]
                 
                self.RiseValues[i] = Slope(self.RiseTime[i],self.RiseWave[i],self.SampFreq)
                self.SlopeRise1[i] = self.RiseValues[i].Slope
                self.DecayValues[i] = Slope(self.DecayTime[i],self.DecayWave[i],self.SampFreq)
                self.SlopeDecay1[i] = self.DecayValues[i].Slope
                    
                # Half Width:
                self.HWTimes = [None]*2
                self.HWTimes[0] = int(self.ThresholdTime1[i]*self.SampFreq)
                self.HWTimes[1] = int(self.ThresholdProjectTime1[i]*self.SampFreq)
                self.HWTime[i] = self.Time[self.HWTimes[0]:self.HWTimes[1]]
                self.HWWave[i] = self.Wave[self.HWTimes[0]:self.HWTimes[1]]
                self.HWValues[i] = HalfWidth(self.HWTime[i],self.HWWave[i],self.SampFreq)
                self.HalfWidth1[i] = self.HWValues[i].HW 
                self.HWPlotTime[i] = self.HWValues[i].HWTime 
                self.HWPlotVm[i] = self.HWValues[i].HWVm 
                
                # Single Calculations:
                self.APAmpBaseline1[i] = self.APpeaks[i]-self.Baseline
                self.APAmpThres1[i] = self.APpeaks[i]-self.ThresholdVm1[i]
                self.TimeToPeak1[i] = self.APTimes[i]-self.ThresholdTime1[i]
                self.APLatency1[i] = self.ThresholdTime1[i] - self.StimOnsetTime    
                
                i += 1
                
            # Burst AHPs:    
            i = 0
            self.BurstAHPPosi = [None]*(self.BurstAPs)
            self.BurstAHPTime = [None]*(self.BurstAPs)
            self.BurstAHPVm1 = [None]*(self.BurstAPs)
            while i < self.BurstAPs:
                self.BurstAHPPosi[i] = np.argmin(self.Wave[int(self.APTimes[i]*self.SampFreq):int(self.APTimes[(i+1)]*self.SampFreq)])
                self.BurstAHPTime[i] = self.Time[(self.BurstAHPPosi[i]+int(self.APTimes[i]*self.SampFreq))]
                self.BurstAHPVm1 [i] = self.Wave[(self.BurstAHPPosi[i]+int(self.APTimes[i]*self.SampFreq))]
                i +=1
               
            # Burst Calculations: Burst Duration and Burst Area
            self.FirstThresMirrored = np.argmin(np.absolute(self.Wave[int(self.APTimes[-1]*self.SampFreq):-1]-self.ThresholdVm1[0]))
            self.FirstThresMirrored = self.FirstThresMirrored+int(self.APTimes[-1]*self.SampFreq)
            self.BurstCalcTime = self.Time[int(self.ThresholdTime1[0]*self.SampFreq):self.FirstThresMirrored]
            self.WaveForBurstFilt = ddhelp.Filters.MovingAverage(self.Wave,window_len=int(10*SampFreq),window='hanning') 
            self.BurstCalcWaveFilt = self.WaveForBurstFilt[int(self.ThresholdTime1[0]*self.SampFreq):self.FirstThresMirrored]
            self.BurstAreaWave = self.BurstCalcWaveFilt-self.BurstCalcWaveFilt[0] 
            self.BurstDuration = len(self.BurstCalcTime)/self.SampFreq
            self.BurstDurationVecTime = self.BurstCalcTime
            self.BurstDurationVecVm = np.full((len(self.BurstDurationVecTime)), self.ThresholdVm1[0])
            self.BurstArea = sp.integrate.simps(self.BurstAreaWave,self.BurstCalcTime)
            
            # Final Burst Calculations:    
            self.APAmpBaseline = self.APAmpBaseline1[0]
            self.APAmpBaselineChange = (np.absolute(self.APAmpBaseline1[-1])-np.absolute(self.APAmpBaseline1[0]))/np.absolute(self.APAmpBaseline1[0])
            self.APAmpThres = self.APAmpThres1[0]
            self.APAmpThresChange = (np.absolute(self.APAmpThres1[-1])-np.absolute(self.APAmpThres1[0]))/np.absolute(self.APAmpThres1[0])
            self.TimeToPeak = self.TimeToPeak1[0]
            self.TimeToPeakChange = (np.absolute(self.TimeToPeak1[-1])-np.absolute(self.TimeToPeak1[0]))/np.absolute(self.TimeToPeak1[0])
            self.APLatency = self.APLatency1[0]
            
            self.ThresholdVm = self.ThresholdVm1[0]
            self.ThresholdVmChange = (np.absolute(self.ThresholdVm1[-1])-np.absolute(self.ThresholdVm1[0]))/np.absolute(self.ThresholdVm1[0])
            self.ThresholdProjectTime = self.ThresholdProjectTime1[-1]
            self.ThresholdProjectVm = self.Wave[int(self.ThresholdProjectTime1[-1]*self.SampFreq)]
            self.SlopeRise = self.SlopeRise1[0]
            self.SlopeRiseChange = (np.absolute(self.SlopeRise1[-1])-np.absolute(self.SlopeRise1[0]))/np.absolute(self.SlopeRise1[0])
            self.SlopeDecay = self.SlopeDecay1[0]
            self.SlopDecayChange = (np.absolute(self.SlopeDecay1[-1])-np.absolute(self.SlopeDecay1[0]))/np.absolute(self.SlopeDecay1[0])
            self.HalfWidth = self.HalfWidth1[0]
            self.HalfWidthChange = (np.absolute(self.HalfWidth1[-1])-np.absolute(self.HalfWidth1[0]))/np.absolute(self.HalfWidth1[0])
            self.APLatency = self.APLatency1[0]
            self.BurstAHPVm = self.BurstAHPVm1[0]
            self.BurstAHPVmChange = (np.absolute(self.BurstAHPVm1[-1])-np.absolute(self.BurstAHPVm1[0]))/np.absolute(self.BurstAHPVm1[0])
            self.BurstAHPT = self.BurstAHPTime[0]    
               
        # Single AP Analysis        
        else:
            self.APType = 'Single Spike'
            self.Burst = 0
            self.NumAPs = 1
            self.APTime = self.APTimes[0]
            self.APPeak = self.APpeaks[0]
            
            # Thresholds and Projected Threshold
            self.ThresTimeWindow = 2
            self.ThresTime1 = int(self.APTimes[0]*self.SampFreq)
            self.ThresTime = [None]*2
            self.ThresTime[0] = int(self.ThresTime1 - (self.ThresTimeWindow*self.SampFreq))
            self.ThresTime[1] = int(self.ThresTime1 + (self.ThresTimeWindow*self.SampFreq))
            self.ThresholdTime = self.Time[self.ThresTime[0]:self.ThresTime[1]]
            self.ThresholdWave = self.Wave[self.ThresTime[0]:self.ThresTime[1]]
            
            self.ThresholdValues = Threshold(self.ThresholdTime,self.ThresholdWave,self.SampFreq)
            self.ThresholdTime1 = self.ThresholdValues.ThresholdTime
            self.ThresholdVm1 = self.ThresholdValues.ThresholdVm
            self.ThresholdProjectTime1 = self.ThresholdValues.ThresProjectTime
            self.ThresholdProjectVm1 = self.ThresholdValues.ThresholdVm
            
            # Slopes
            self.RiseT = [None]*2
            self.RiseT[0] = int(self.ThresholdTime1*self.SampFreq)
            self.RiseT[1] = int(self.APTimes[0]*self.SampFreq)
            self.RiseTime = self.Time[self.RiseT[0]:self.RiseT[1]]
            self.RiseWave = self.Wave[self.RiseT[0]:self.RiseT[1]]
            self.DecayT = [None]*2
            self.DecayT[0] = int(self.APTimes[0]*self.SampFreq)
            self.DecayT[1] = int(self.ThresholdProjectTime1*self.SampFreq)
            self.DecayTime = self.Time[self.DecayT[0]:self.DecayT[1]]
            self.DecayWave = self.Wave[self.DecayT[0]:self.DecayT[1]]
            
            self.RiseValues = Slope(self.RiseTime,self.RiseWave,self.SampFreq)
            self.SlopeRise1 = self.RiseValues.Slope
            self.DecayValues = Slope(self.DecayTime,self.DecayWave,self.SampFreq)
            self.SlopeDecay1 = self.DecayValues.Slope
            
            #Half Width: 
            self.HWTimes = [None]*2
            self.HWTimes[0] = int(self.ThresholdTime1*self.SampFreq)
            self.HWTimes[1] = int(self.ThresholdProjectTime1*self.SampFreq)
            self.HWTime = self.Time[self.HWTimes[0]:self.HWTimes[1]]
            self.HWWave = self.Wave[self.HWTimes[0]:self.HWTimes[1]]
            self.HWValues = HalfWidth(self.HWTime,self.HWWave,self.SampFreq)
            self.HWPlotTime = [None]
            self.HWPlotVm = [None]
            self.HWPlotTime[0] = self.HWValues.HWTime 
            self.HWPlotVm[0] = self.HWValues.HWVm 
            self.HalfWidth1 = self.HWValues.HW
            
            # Single Calculations: 
            self.APAmpBaseline1 = self.APPeak-self.Baseline
            self.APAmpBaseline = self.APAmpBaseline1
            self.APAmpBaselineChange = 0
            self.APAmpThres1 = self.APPeak-self.ThresholdVm1
            self.APAmpThres = self.APAmpThres1
            self.APAmpThresChange = 0
            self.TimeToPeak1 = self.APTime-self.ThresholdTime1
            self.TimeToPeak = self.TimeToPeak1
            self.TimeToPeakChange = 0
            self.APLatency1 = self.ThresholdTime1 - self.StimOnsetTime
            self.APLatency = self.APLatency1
        
            self.ThresholdVm = self.ThresholdVm1
            self.ThresholdVmChange = 0
            self.ThresholdProjectTime = self.ThresholdProjectTime1
            self.ThresholdProjectVm = self.ThresholdProjectVm1
            self.SlopeRise = self.SlopeRise1
            self.SlopeRiseChange = 0
            self.SlopeDecay = self.SlopeDecay1
            self.SlopDecayChange = 0
            self.HalfWidth = self.HalfWidth1
            self.HalfWidthChange = 0    
            
            self.BurstAHPTime = np.nan
            self.BurstAHPVm1 = np.nan
            self.BurstAHPVm = np.nan
            self.BurstAHPT = np.nan
            self.BurstAHPVmChange = 0
            self.BurstDuration = np.nan
            self.BurstDurationVecTime = np.nan
            self.BurstDurationVecVm = np.nan
            self.BurstArea = np.nan
            
    # AHP Properties:  
    # AHP Wave:
        self.AHPEnd = self.ThresholdProjectTime + 200
        if len(self.APpeaks) > 1: 
            if self.Burst == 1:
                if len(self.APTimes) > self.BurstAPs+1:
                    if self.AHPEnd > self.APTimes[int(self.BurstAPs+1)]:
                        #print('1')
                        self.AHPEnd = self.APTimes[self.BurstAPs+1]
            if self.Burst == 0 and self.AHPEnd > self.APTimes[1]:
                #print('2')
                self.AHPEnd = self.APTimes[1]-2
        if self.AHPEnd > self.StimOffTime:
            #print('2')
            self.AHPEnd = self.StimOffTime-1    
            
        self.AHPStartTP = int(self.ThresholdProjectTime*self.SampFreq)
        self.AHPEndTP = int(self.AHPEnd*self.SampFreq)
        
        self.AHPTime = self.Time[self.AHPStartTP:self.AHPEndTP]
        self.AHPWave = self.Wave[self.AHPStartTP:self.AHPEndTP]-self.VStable
        self.AHPValues = AHP(self.AHPTime,self.AHPWave,self.SampFreq,self.BaselineSD)
        self.AHPType = self.AHPValues.AHPType
        self.fAHPVm = self.AHPValues.fAHPVm+self.VStable
        self.fAHPTtP = self.AHPValues.fAHPTime
        self.fAHPTP = (self.AHPValues.fAHPTP+self.AHPStartTP)/self.SampFreq
        self.ADPVm = self.AHPValues.ADPVm+self.VStable
        self.ADPTtP = self.AHPValues.ADPTime
        self.ADPTP = (self.AHPValues.ADPTP+self.AHPStartTP)/self.SampFreq        
        self.mAHPVm = self.AHPValues.mAHPVm+self.VStable
        self.mAHPTtP = self.AHPValues.mAHPTime
        self.mAHPTP = (self.AHPValues.mAHPTP+self.AHPStartTP)/self.SampFreq
        self.RepoIndex = self.AHPValues.RepoIndex
        
        # Area and Time to VStable
        '''
        '''
        self.AHPTimeEnd = self.Time[self.AHPStartTP:int(self.StimOffTime*self.SampFreq)-1]
        self.AHPWaveEnd = self.Wave[self.AHPStartTP:int(self.StimOffTime*self.SampFreq)-1]
        if self.mAHPTP is not np.nan:
            self.AHPPoint = self.mAHPTP
        elif self.ADPTP is not np.nan: 
            self.AHPPoint = self.ADPTP
        elif self.fAHPTP is not np.nan: 
            self.AHPPoint = self.fAHPTP
        self.AHPEndCalc = AHPEnd(self.AHPTimeEnd,self.AHPWaveEnd,self.AHPPoint,self.SampFreq,self.BaselineSD) 
        self.AHPArea = self.AHPEndCalc.Area
        self.AHPTtVStable = self.AHPEndCalc.VStableTime+(self.AHPStartTP/self.SampFreq)
        self.AHPTimePointVStable = int(self.AHPTtVStable*self.SampFreq)
        self.AHPAreaTimeVec = self.AHPEndCalc.AreaTime
        self.AHPAreaStableVec = self.AHPEndCalc.AreaVec
        self.AHPAreaWaveVec = self.AHPEndCalc.AreaWave+self.AHPAreaStableVec 
        if self.AHPEndCalc.Cave == 1:   
            self.AHPArea = np.nan
            self.AHPTtVStable = np.nan
            self.AHPTimePointVStable = self.AHPEndTP
            self.AHPAreaTimeVec = [np.nan]
            self.AHPAreaStableVec = [np.nan]
            self.AHPAreaWaveVec = [np.nan]   
#        
        # sAHP
        self.sAHPTime = self.Time[self.StimOffset:-1]
        self.sAHPWave = self.Wave[self.StimOffset:-1]*-1
        
        self.sAHPValues = Sub.ReboundResponse(self.sAHPTime,self.sAHPWave,self.SampFreq)
        self.sAHPT = self.sAHPValues.MaxResponseT+self.StimOffTime
        self.sAHPVm = self.sAHPValues.MaxResponseVm*-1
        self.sAHPAmplitude = (np.absolute(self.sAHPValues.MaxResponseVm)-np.absolute(self.Baseline))
             
    # Plotting Figure:
        # Prevent from popping up:
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
            self.Figure = plt.figure()
#            self.Figure.set_dpi(300)
#            self.Figure.set_size_inches(11.69, 8.27, forward=True)
            
        # Subplotting:
        self.gs = gridspec.GridSpec(2, 2)
        self.gs.update(left=0.1, bottom= 0.1, top = 0.9, right=0.9, wspace=0.25)
        
        self.ax = plt.subplot(self.gs[:,0])
        self.axAP = plt.subplot(self.gs[0,1])
        self.axAHP = plt.subplot(self.gs[1,1])

        # Calculations for Plotting:
        self.LatencyPlot = np.zeros(shape=(2,2))
        self.LatencyPlot[0,0] = self.StimOnsetTime
        self.LatencyPlot[0,1] = self.StimOnsetTime+self.APLatency
        self.LatencyPlot[1,0] = self.ThresholdProjectVm+3
        self.LatencyPlot[1,1] = self.ThresholdProjectVm+3
        self.PlotHW = np.zeros(shape=(self.BurstAPs,4))
        
        # Each Line in list: PlotWave:
        self.PlotWave = [None] * (12+self.NumAPs)
#        print(len(self.PlotWave))
        self.XPlot = [None] * self.NumAPs
        if self.Burst == 1:
            self.PlotWave = [None] * (12+self.NumAPs)
            self.XPlot = [None] * (self.NumAPs)
            
        self.PlotWave[0], = self.ax.plot(self.Time,self.Wave,'k')
        self.PlotWave[1], = self.ax.plot(self.AHPAreaTimeVec,self.AHPAreaStableVec,'--', color = [0.729, 0.729, 0.729])
        self.PlotWave[2], = self.ax.plot(self.APTime,self.APPeak,'o',color = [0.6, 0.043, 0.215])
        self.PlotWave[3], = self.ax.plot(self.ThresholdTime1,self.ThresholdVm1,'o',color = [0.023, 0.674, 0.258])
        self.PlotWave[4], = self.ax.plot(self.ThresholdProjectTime,self.ThresholdProjectVm,'o',color = [0.023, 0.674, 0.258])    
        self.PlotWave[5], = self.ax.plot(self.LatencyPlot[0,:],self.LatencyPlot[1,:],'--',color = [0.309, 0.321, 0.058])
        self.PlotWave[6], = self.ax.plot(self.fAHPTP,self.fAHPVm,'o',color = [0.509, 0.027, 0.913])   
        self.PlotWave[7], = self.ax.plot(self.ADPTP,self.ADPVm,'o', color =[0.913, 0.509, 0.027])
        self.PlotWave[8], = self.ax.plot(self.mAHPTP,self.mAHPVm,'o', color =[0.027, 0.501, 0.913])
        self.PlotWave[9], = self.ax.plot(self.sAHPT,self.sAHPVm,'o', color = [0.772, 0.913, 0.027])
        self.PlotWave[10], = self.ax.plot(self.BurstAHPTime,self.BurstAHPVm1,'o', color = [0.509, 0.027, 0.913])   
        self.PlotWave[11], = self.ax.plot(self.BaselineP[0],self.BaselineP[1],color = [0.149, 0.670, 0.239])
        
        # Single Lines per Calculation, with burst!!! 
        i = 0
        while i < self.NumAPs:
            #print(i)
            self.XPlot[i] = self.ax.plot(self.HWPlotTime[i],self.HWPlotVm[i],'-', color = [0.976, 0.737, 0.301])
            self.PlotWave[12+i] = self.XPlot[i][0]
            i += 1
            
        if self.Burst == 1:    
            i = 0
            while i < self.NumAPs:
                #print(i)
                self.XPlot[i] = self.ax.plot(self.HWPlotTime[i],self.HWPlotVm[i],'-', color = [0.976, 0.737, 0.301])
                self.PlotWave[12+i] = self.XPlot[i][0]
                i += 1
        
        # Annotation of Whole Wave:
        if self.Burst == 0:     
            self.AnnotWave = [None] * 6
            self.AnnotWave[0] = self.ax.annotate("Base: %.1f mV" % self.BaselineP[1][0],xy=(self.BaselineP[0][0],self.BaselineP[1][0]),xytext=(1,5),color = [0.149, 0.670, 0.239],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[1] = self.ax.annotate("Peak: %.1f mV" % self.APPeak,xy=(self.APTime,self.APPeak),xytext=(5,-2),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[2] = self.ax.annotate("Peak-Baseline Amp: %.1f mV" % self.APAmpBaseline,xy=(self.APTime,self.APPeak),xytext=(5,-10),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[3] = self.ax.annotate("AHP Area: %.0f mV$^{2}$/ms" % self.AHPArea,xy=(self.AHPAreaTimeVec[0],self.AHPAreaStableVec[0]),xytext=(5,5),color = [0.729, 0.729, 0.729],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[4] = self.ax.annotate("Latency: %.1f ms" % self.APLatency,xy=(self.LatencyPlot[0,0],self.LatencyPlot[1,0]),xytext=(1,5),color = [0.309, 0.321, 0.058],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[5] = self.ax.annotate("sAHP Amp: %.1f mV" % self.sAHPAmplitude,xy=(self.sAHPT,self.sAHPVm),xytext=(-25,8),color = [0.772, 0.913, 0.027],xycoords='data',textcoords='offset points', fontsize = 6)
        elif self.Burst == 1:     
            self.AnnotWave = [None] * 7
            self.AnnotWave[0] = self.ax.annotate("Base: %.1f mV" % self.BaselineP[1][0],xy=(self.BaselineP[0][0],self.BaselineP[1][0]),xytext=(1,5),color = [0.149, 0.670, 0.239],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[1] = self.ax.annotate("Peak: %.1f mV" % self.APPeak[0],xy=(self.APTime[0],self.APPeak[0]),xytext=(5,-2),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[2] = self.ax.annotate("Peak-Baseline Amp: %.1f mV" % self.APAmpBaseline,xy=(self.APTime[0],self.APPeak[0]),xytext=(5,-10),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[3] = self.ax.annotate("Peak-Baseline Amp Change: %.1f" % self.APAmpBaselineChange,xy=(self.APTime[0],self.APPeak[0]),xytext=(5,-18),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[4] = self.ax.annotate("AHP Area: %.0f mV$^{2}$/ms" % self.AHPArea,xy=(self.AHPAreaTimeVec[0],self.AHPAreaStableVec[0]),xytext=(5,5),color = [0.729, 0.729, 0.729],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[5] = self.ax.annotate("Latency: %.1f ms" % self.APLatency,xy=(self.LatencyPlot[0,0],self.LatencyPlot[1,0]),xytext=(1,5),color = [0.309, 0.321, 0.058],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[6] = self.ax.annotate("sAHP Amp: %.1f mV" % self.sAHPAmplitude,xy=(self.sAHPT,self.sAHPVm),xytext=(-25,8),color = [0.772, 0.913, 0.027],xycoords='data',textcoords='offset points', fontsize = 6)

        #Set Axes for Wave!
        self.ax.set_xlabel('ms')
        self.ax.set_ylabel('mV')
        self.ax.set_xlim(self.Time[0],self.Time[-1])
        
        # Each Line in list: PlotAP:
        self.PlotAP = [None] * (6+self.NumAPs)
        self.PlotAPX = [None] * self.NumAPs
        if self.Burst == 1:
            self.PlotAP = [None] * (6+self.NumAPs)
            self.PlotAPX = [None] * self.NumAPs
        self.PlotAP[0], = self.axAP.plot(self.Time[int((self.LatencyPlot[0,1]-2)*self.SampFreq):int((self.ThresholdProjectTime+50)*SampFreq)],self.Wave[int((self.LatencyPlot[0,1]-2)*self.SampFreq):int((self.ThresholdProjectTime+50)*self.SampFreq)],'k')
        self.PlotAP[1], = self.axAP.plot(self.APTime,self.APPeak,'o',color = [0.6, 0.043, 0.215])
        self.PlotAP[2], = self.axAP.plot(self.ThresholdTime1,self.ThresholdVm1,'o',color = [0.023, 0.674, 0.258])
        self.PlotAP[3], = self.axAP.plot(self.ThresholdProjectTime,self.ThresholdProjectVm,'o',color = [0.023, 0.674, 0.258])
        self.PlotAP[4], = self.axAP.plot(self.BurstAHPTime,self.BurstAHPVm1,'o', color = [0.509, 0.027, 0.913])   
        self.PlotAP[5], = self.axAP.plot(self.BurstDurationVecTime,self.BurstDurationVecVm,'--', color = [0.976, 0.623, 0.203])   
        
        i = 0
        while i < self.NumAPs:
            self.PlotAPX[i] = self.axAP.plot(self.HWPlotTime[i],self.HWPlotVm[i],'-',color = [0.976, 0.737, 0.301])
            self.PlotAP[6+i] = self.PlotAPX[i][0]
            i += 1    
        
        if self.Burst == 1:
            while i < self.NumAPs:
                self.PlotAPX[i] = self.axAP.plot(self.HWPlotTime[i],self.HWPlotVm[i],'-',color = [0.976, 0.737, 0.301])
                self.PlotAP[6+i] = self.PlotAPX[i][0]
                i += 1
                
        # Annotation for PlotAP:
        if self.Burst == 0:  
            self.AnnotAP = [None] * 4
            self.AnnotAP[0] = self.axAP.annotate("Peak: %.1f mV" % self.APPeak,xy=(self.APTime,self.APPeak),xytext=(5,-2),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAP[1] = self.axAP.annotate("Peak-Thres Amp: %.1f mV" % self.APAmpThres,xy=(self.APTime,self.APPeak),xytext=(5,-10),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAP[2] = self.axAP.annotate("Half Width: %.1f ms" % self.HalfWidth,xy=(self.HWPlotTime[0][1],self.HWPlotVm[0][1]),xytext=(5,-2),color = [0.976, 0.737, 0.301],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAP[3] = self.axAP.annotate("Threshold: %.1f mV" % self.ThresholdVm,xy=(self.ThresholdProjectTime,self.ThresholdProjectVm),xytext=(5,-2),color = [0.023, 0.674, 0.258],xycoords='data',textcoords='offset points', fontsize = 6)
        elif self.Burst ==1:
            self.AnnotAP = [None] * 10
            self.AnnotAP[0] = self.axAP.annotate("Peak: %.1f mV" % self.APPeak[0],xy=(self.APTime[0],self.APPeak[0]),xytext=(5,-2),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAP[1] = self.axAP.annotate("Peak-Thres Amp: %.1f mV" % self.APAmpThres,xy=(self.APTime[0],self.APPeak[0]),xytext=(5,-10),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAP[2] = self.axAP.annotate("Peak-Thres Change: %.1f" % self.APAmpThresChange,xy=(self.APTime[0],self.APPeak[0]),xytext=(5,-18),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAP[3] = self.axAP.annotate("Half Width: %.1f ms" % self.HalfWidth,xy=(self.HWPlotTime[0][1],self.HWPlotVm[0][1]),xytext=(5,-2),color = [0.976, 0.737, 0.301],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAP[4] = self.axAP.annotate("Half Width Change: %.1f " % self.HalfWidthChange,xy=(self.HWPlotTime[0][1],self.HWPlotVm[0][1]),xytext=(5,-10),color = [0.976, 0.737, 0.301],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAP[5] = self.axAP.annotate("Threshold: %.1f mV" % self.ThresholdVm,xy=(self.ThresholdProjectTime,self.ThresholdVm),xytext=(5,-2),color = [0.023, 0.674, 0.258],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAP[6] = self.axAP.annotate("Threshold Change: %.1f" % self.ThresholdVmChange,xy=(self.ThresholdProjectTime,self.ThresholdVm),xytext=(5,-10),color = [0.023, 0.674, 0.258],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAP[7] = self.axAP.annotate("BurstAHP: %.1f" % self.BurstAHPVm,xy=(self.BurstAHPT,self.BurstAHPVm),xytext=(-2,-2),color = [0.509, 0.027, 0.913], xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAP[8] = self.axAP.annotate("BurstAHP Change: %.1f" % self.BurstAHPVmChange,xy=(self.BurstAHPT,self.BurstAHPVm),xytext=(-2,-10),color = [0.509, 0.027, 0.913], xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAP[9] = self.axAP.annotate("Burst Area: %.0f mV$^{2}$/ms" % self.BurstArea,xy=(self.BurstDurationVecTime[0],self.BurstDurationVecVm[0]),xytext=(5,2),color = [0.976, 0.623, 0.203],xycoords='data',textcoords='offset points', fontsize = 6)
            
        # Set Axes for APWave:    
        self.axAP.set_xlim(self.Time[int((self.LatencyPlot[0,1]-2)*self.SampFreq)],self.Time[int((self.ThresholdProjectTime+50)*SampFreq)])
        self.axAP.set_ylabel('mV')   
        
        # Calculations for AHPs:
        self.AHPLimt = np.zeros(shape=(4,1))
        self.AHPLimt[0] = self.fAHPVm
        self.AHPLimt[1] = self.ADPVm
        self.AHPLimt[2] = self.mAHPVm
        self.AHPLimt[3] = self.ThresholdProjectVm
        
        self.ADPPlotMin = np.nanmin(self.AHPLimt)
        self.ADPPlotMax = np.nanmax(self.AHPLimt)
        
        self.fAHPPlot = np.zeros(shape=(2,2))
        self.fAHPPlot[0,0] = self.ThresholdProjectTime
        self.fAHPPlot[0,1] = self.ThresholdProjectTime+self.fAHPTtP
        
        self.ADPPlot = np.zeros(shape=(2,2))
        self.ADPPlot[0,0] = self.ThresholdProjectTime
        self.ADPPlot[0,1] = self.ThresholdProjectTime+self.ADPTtP
        
        self.mAHPPlot = np.zeros(shape=(2,2))
        self.mAHPPlot[0,0] = self.ThresholdProjectTime
        self.mAHPPlot[0,1] = self.ThresholdProjectTime+self.mAHPTtP
        
        self.fAHPPlot[1,0] = self.ADPPlotMin-2
        self.fAHPPlot[1,1] = self.ADPPlotMin-2
        self.ADPPlot[1,0] = self.ADPPlotMin-7
        self.ADPPlot[1,1] = self.ADPPlotMin-7
        self.mAHPPlot[1,0] = self.ADPPlotMin-12
        self.mAHPPlot[1,1] = self.ADPPlotMin-12
        
        # Each Line in list: PlotAHP:
        self.PlotAHP = [None] * 10
        self.PlotAHP[0], = self.axAHP.plot(self.Time[self.AHPStartTP:self.AHPTimePointVStable],self.Wave[self.AHPStartTP:self.AHPTimePointVStable],'k')
        self.PlotAHP[1], = self.axAHP.plot(self.ThresholdProjectTime,self.ThresholdProjectVm,'o',color = [0.023, 0.674, 0.258])
        self.PlotAHP[2], = self.axAHP.plot(self.fAHPTP,self.fAHPVm,'o',color = [0.509, 0.027, 0.913])   
        self.PlotAHP[3], = self.axAHP.plot(self.ADPTP,self.ADPVm,'o', color =[0.913, 0.509, 0.027])
        self.PlotAHP[4], = self.axAHP.plot(self.mAHPTP,self.mAHPVm,'o', color =[0.027, 0.501, 0.913])
        self.PlotAHP[5], = self.axAHP.plot(self.fAHPPlot[0,:],self.fAHPPlot[1,:],color = [0.509, 0.027, 0.913])
        self.PlotAHP[6], = self.axAHP.plot(self.ADPPlot[0,:],self.ADPPlot[1,:],color =[0.913, 0.509, 0.027])
        self.PlotAHP[7], = self.axAHP.plot(self.mAHPPlot[0,:],self.mAHPPlot[1,:],color =[0.027, 0.501, 0.913])
        
        self.PlotAHP[8], = self.axAHP.plot(self.Time[self.AHPStartTP],self.ADPPlotMin-5,'.w')
        self.PlotAHP[9], = self.axAHP.plot(self.Time[self.AHPStartTP],self.ADPPlotMax,'.w')
        
        # Annotation for AHP: 
        if self.AHPType == 'fMAHP' or self.AHPType == 'FmAHP':
            self.AnnotAHP = [None] * 6
            self.AnnotAHP[0] = self.axAHP.annotate("fAHP: %.1f mV" % self.fAHPVm,xy=(self.fAHPTP,self.fAHPVm),xytext=(5,-2),color = [0.509, 0.027, 0.913],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAHP[1] = self.axAHP.annotate("ADP: %.1f mV" % self.ADPVm,xy=(self.ADPTP,self.ADPVm),xytext=(5,-2),color = [0.913, 0.509, 0.027],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAHP[2] = self.axAHP.annotate("mAHP: %.1f mV" % self.mAHPVm,xy=(self.mAHPTP,self.mAHPVm),xytext=(5,-2),color = [0.027, 0.501, 0.913],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAHP[3] = self.axAHP.annotate("fAHPTtP: %.1f ms" % self.fAHPTtP,xy=(self.fAHPPlot[0,0],self.fAHPPlot[1,0]),xytext=(5,-5),color = [0.509, 0.027, 0.913],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAHP[4] = self.axAHP.annotate("ADPTtP: %.1f ms" % self.ADPTtP,xy=(self.ADPPlot[0,0],self.ADPPlot[1,0]),xytext=(5,-3),color = [0.913, 0.509, 0.027],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAHP[5] = self.axAHP.annotate("mAHPTtP: %.1f ms" % self.mAHPTtP,xy=(self.mAHPPlot[0,0],self.mAHPPlot[1,0]),xytext=(5,+3),color = [0.027, 0.501, 0.913],xycoords='data',textcoords='offset points', fontsize = 6)
        elif self.AHPType == 'fAHP':
            self.AnnotAHP = [None] * 2
            self.AnnotAHP[0] = self.axAHP.annotate("fAHP: %.1f mV" % self.fAHPVm,xy=(self.fAHPTP,self.fAHPVm),xytext=(5,-2),color = [0.509, 0.027, 0.913],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAHP[1] = self.axAHP.annotate("fAHPTtP: %.1f ms" % self.fAHPTtP,xy=(self.fAHPPlot[0,0],self.fAHPPlot[1,0]),xytext=(5,-5),color = [0.509, 0.027, 0.913],xycoords='data',textcoords='offset points', fontsize = 6)
        elif self.AHPType == 'ADP':
            self.AnnotAHP = [None] * 2
            self.AnnotAHP[0] = self.axAHP.annotate("ADP: %.1f mV" % self.ADPVm,xy=(self.ADPTP,self.ADPVm),xytext=(5,-10),color = [0.913, 0.509, 0.027],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAHP[1] = self.axAHP.annotate("ADPTtP: %.1f ms" % self.ADPTtP,xy=(self.ADPPlot[0,0],self.ADPPlot[1,0]),xytext=(5,-5),color = [0.913, 0.509, 0.027],xycoords='data',textcoords='offset points', fontsize = 6)
        elif self.AHPType == 'mAHP':
            self.AnnotAHP = [None] * 2
            self.AnnotAHP[0] = self.axAHP.annotate("mAHP: %.1f mV" % self.mAHPVm,xy=(self.mAHPTP,self.mAHPVm),xytext=(5,-2),color = [0.027, 0.501, 0.913],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAHP[1] = self.axAHP.annotate("mAHPTtP: %.1f ms" % self.mAHPTtP,xy=(self.mAHPPlot[0,0],self.mAHPPlot[1,0]),xytext=(5,+3),color = [0.027, 0.501, 0.913],xycoords='data',textcoords='offset points', fontsize = 6)
        
        # Set Axes for APWave:    
        self.axAHP.set_xlim([self.AHPTime[0],self.AHPTime[-1]])
        self.axAHP.set_xlabel('ms')
        self.axAHP.set_ylabel('mV')

''' Main Script Firing Properties: '''
class MainFiring:    
    def __init__ (self,Names,Time,Wave,Stimulus,SampFreq,PrintShow=0):
        self.WaveNames = Names
        self.Time = Time
        self.Wave = Wave
        self.Stimulus = Stimulus
        self.SampFreq = SampFreq
        self.PrintShow = PrintShow
        
        # Define StimOn- StimOffset
        self.StimDiffWave = np.diff(self.Stimulus)
        self.StimDiffPoints = np.where(self.StimDiffWave != 0)
        self.StimOnset = np.asarray(self.StimDiffPoints[0][0])
        self.StimOffset = np.asarray(self.StimDiffPoints[0][1])
        self.StimOffset = self.StimOffset+int(1*self.SampFreq)
        self.StimOnsetTime = self.StimOnset/SampFreq
        self.StimOffTime = self.StimOffset/SampFreq
        
        #Calculation Basline
        self.Baseline = np.mean(self.Wave[0:(self.StimOnset)])
        self.BaselineSD = np.std(self.Wave[0:(self.StimOnset)])
        self.BaselineP = np.zeros((2,self.StimOnset))
        self.BaselineP[0] = self.Time[0:self.StimOnset]
        self.BaselineP[1]= self.Baseline
        
        # Find Peaks
        self.APPeakThreshold = -20
        self.FindAPsTime = self.Time[self.StimOnset:self.StimOffset]
        self.FindAPsWave = self.Wave[self.StimOnset:self.StimOffset]
        self.APs = FindAPs2(self.FindAPsTime,self.FindAPsWave,self.SampFreq,2,1,self.APPeakThreshold) 
        self.APTimes = self.APs.APTimes+self.StimOnsetTime
        self.APpeaks = self.APs.APPeaks
        self.InstFiringFreq = (1/self.APs.APIntervals)*1000
        self.APNum = self.APs.APNum
             
        if np.size(self.APpeaks) < 3:
            self.APPeakThreshold = -30
            self.FindAPsTime = self.Time[self.StimOnset:self.StimOffset]
            self.FindAPsWave = self.Wave[self.StimOnset:self.StimOffset]
            self.APs = FindAPs2(self.FindAPsTime,self.FindAPsWave,self.SampFreq,1,0.75,self.APPeakThreshold) 
            self.APTimes = self.APs.APTimes+self.StimOnsetTime
            self.APpeaks = self.APs.APPeaks
            self.InstFiringFreq = (1/self.APs.APIntervals)*1000
            self.APNum = self.APs.APNum
            print('FiringThreshold to -30mV')
   
        self.HighFreq = 0
        if any(self.InstFiringFreq >= 500):
            self.APPeakThreshold = -10
            self.FindAPsTime = self.Time[self.StimOnset:self.StimOffset]
            self.FindAPsWave = self.Wave[self.StimOnset:self.StimOffset]
            self.APs = FindAPs2(self.FindAPsTime,self.FindAPsWave,self.SampFreq,7.5,30,self.APPeakThreshold) 
            self.APTimes = self.APs.APTimes+self.StimOnsetTime
            self.APpeaks = self.APs.APPeaks
            self.InstFiringFreq = (1/self.APs.APIntervals)*1000
            self.APNum = self.APs.APNum
            
        self.HighFreq = 0
        if any(self.InstFiringFreq >= 500):
            print('Driss')
            self.APPeakThreshold = 10
            self.APs = FindAPs2(self.FindAPsTime,self.FindAPsWave,self.SampFreq,3,4,self.APPeakThreshold) 
            self.APTimes = self.APs.APTimes+self.StimOnsetTime
            self.APpeaks = self.APs.APPeaks
            self.InstFiringFreq = (1/self.APs.APIntervals)*1000
            self.APNum = self.APs.APNum
            self.HighFreq = 1
        
        # Thresholds
        self.ThresStart = [None]*len(self.APpeaks)
        self.ThresStop = [None]*len(self.APpeaks)
        self.ThresAPTP = [None]*len(self.APpeaks)
        self.ThresholdTimes = [None]*len(self.APpeaks)
        self.ThresholdWaves = [None]*len(self.APpeaks)
        self.ThresholdValues = [None]*len(self.APpeaks)
        self.ThresholdTime1 = [None]*len(self.APpeaks)
        self.ThresholdVm1 = [None]*len(self.APpeaks)
        self.ThresholdProjectTime1 = [None]*len(self.APpeaks)
        self.ThresholdProjectVm1 = [None]*len(self.APpeaks)
        
        # New: Minima Between Peaks for Windows! 
        self.ThresTimeWindowStart = 2
        self.ThresTimeWindowEnd = 3
        if self.HighFreq == 1:
            self.ThresTimeWindowStart = 1.
            self.ThresTimeWindowEnd = 1.5 
        # First AP: 
        self.ThresStart[0] = int(self.APTimes[0]*self.SampFreq-(self.ThresTimeWindowStart*self.SampFreq))
        if self.ThresStart[0] < self.StimOnset+int(0.5*self.SampFreq):
            self.ThresStart[0] = self.StimOnset + int(0.5*self.SampFreq)
        # Middle APs: 
        i = 1
        self.Minima = [None]*(len(self.APTimes)-1)
        while i < (len(self.APTimes)):
            self.Minima[i-1] = int(self.APTimes[i-1]*self.SampFreq)+np.argmin(self.Wave[int(self.APTimes[i-1]*self.SampFreq):int(self.APTimes[i]*self.SampFreq)])
            i += 1
        i = 1
        while i < (len(self.APTimes)): 
            self.ThresStart[i] = self.Minima[i-1]
            i += 1
        i = 0
        while i < (len(self.APTimes)-1): 
            self.ThresStop[i] = self.Minima[i]
            i += 1
        # Last AP Stop:
        self.ThresStop[-1] = int(self.APTimes[-1]*self.SampFreq+(self.ThresTimeWindowEnd*self.SampFreq))
        
        i = 0
        while i < len(self.APTimes):  
            self.ThresAPTP[i] = int(self.APTimes[i]*self.SampFreq)
            if self.ThresAPTP[i]-self.ThresStart[i] > 3*self.SampFreq:
                self.ThresStart[i] = self.ThresAPTP[i]-int(3*self.SampFreq)
                if self.ThresStart[i] < self.ThresAPTP[i-1]:
                    self.ThresStart[i] = self.ThresAPTP[i-1]+int(0.5*self.SampFreq)
                
            self.ThresholdTimes[i] = self.Time[self.ThresStart[i]:self.ThresAPTP[i]]
            self.ThresholdWaves[i] = self.Wave[self.ThresStart[i]:self.ThresAPTP[i]]
            self.ThresholdValues[i] = Threshold(self.ThresholdTimes[i],self.ThresholdWaves[i],self.SampFreq)            
            self.ThresholdTime1[i] = self.ThresholdValues[i].ThresholdTime
            self.ThresholdVm1[i] = self.ThresholdValues[i].ThresholdVm
            self.ThresholdProjectTime1[i] = self.ThresholdValues[i].ThresProjectTime
            self.ThresholdProjectVm1[i] = self.ThresholdValues[i].ThresholdVm
            i += 1
               
        self.ThresholdTimes = np.asarray(self.ThresholdTime1[:])
        self.ThresholdsVm = np.asarray(self.ThresholdVm1[:])
        
        if len(self.APpeaks) >= 3:

            # Baseline Peak Adaption:
            self.AmplitudesPosition = self.APTimes #np.arange(0,len(self.APpeaks),1)
            self.APBase = self.APpeaks - self.Baseline
            self.APBaseAmplitude = self.APBase[0]
            self.APBaseAdaptionValues = FiringAdaption2(self.AmplitudesPosition,self.APBase,self.SampFreq)
            self.APBaseSlowAdaptionFit = self.APBaseAdaptionValues.SlowAdaptionFit
            self.APBaseSlowAdaptionFitr = self.APBaseAdaptionValues.SlowAdaptionFitr
            self.APBaseSlowAdaptionFitCurve = self.APBaseAdaptionValues.SlowAdaptionFitCurve
            self.APBaseSlowAdaptionFitCurveT = self.APBaseAdaptionValues.SlowAdaptionFitCurveT
            self.APBaseFastAdaptionFit = self.APBaseAdaptionValues.FastAdaptionFit
            self.APBaseFastAdaptionFitr = self.APBaseAdaptionValues.FastAdaptionFitr
            self.APBaseFastAdaptionCurve = self.APBaseAdaptionValues.FastAdaptionFitCurve
            self.APBaseFastAdaptionCurveT = self.APBaseAdaptionValues.FastAdaptionFitCurveT
            self.APBaseAdaptionIndex = self.APBaseAdaptionValues.ChangeIndex 
                      
            
            # Threshold Peak Adaption: 
            self.APThres = self.APpeaks - self.ThresholdsVm
            self.APThresAmplitude = self.APThres[0]
            self.APThresAdaptionValues = FiringAdaption2(self.AmplitudesPosition,self.APThres,self.SampFreq)
            self.APThresSlowAdaptionFit = self.APThresAdaptionValues.SlowAdaptionFit
            self.APThresSlowAdaptionFitr = self.APThresAdaptionValues.SlowAdaptionFitr
            self.APThresSlowAdaptionFitCurve = self.APThresAdaptionValues.SlowAdaptionFitCurve
            self.APThresSlowAdaptionFitCurveT = self.APThresAdaptionValues.SlowAdaptionFitCurveT
            self.APThresFastAdaptionFit = self.APThresAdaptionValues.FastAdaptionFit
            self.APThresFastAdaptionFitr = self.APThresAdaptionValues.FastAdaptionFitr
            self.APThresFastAdaptionCurve = self.APThresAdaptionValues.FastAdaptionFitCurve
            self.APThresFastAdaptionCurveT = self.APThresAdaptionValues.FastAdaptionFitCurveT
            self.APThresAdaptionIndex = self.APThresAdaptionValues.ChangeIndex 
            
            # FiringFrequency Accomodation: 
            self.FiringInstTime = np.diff(self.APTimes)
            self.FiringFrequencies = 1000./self.FiringInstTime
            self.FiringFreqPosi = np.arange(0,len(self.FiringFrequencies),1)
            i = 0
            while i < len(self.FiringFrequencies):
                self.FiringFreqPosi[i] = self.APTimes[i] + self.FiringInstTime[i]/2   
                i += 1 
                
                
            # Firing Values:
            self.FiringDuration = self.APTimes[-1]-self.APTimes[0]
            self.FiringFrequencyMean  = np.mean(self.FiringFrequencies)
            self.FirstFreqency = self.FiringFrequencies[0]
            self.FiringFrequencySD  = np.std(self.FiringFrequencies)
            self.FirstSpikeLatency = self.APTimes[0]-self.StimOnsetTime
            # For Accomodation:
            self.FiringFreqPosiFit =  self.FiringFreqPosi   
            self.FiringFrequenciesFit = self.FiringFrequencies
            
            # Accomodation:
            self.FFValues = FiringAdaption2(self.FiringFreqPosiFit,self.FiringFrequenciesFit,self.SampFreq)
            self.FFSlowAdaptionFit = self.FFValues.SlowAdaptionFit
            self.FFSlowAdaptionFitr = self.FFValues.SlowAdaptionFitr
            self.FFSlowAdaptionFitCurve = self.FFValues.SlowAdaptionFitCurve
            self.FFSlowAdaptionFitCurveT = self.FFValues.SlowAdaptionFitCurveT
            self.FFFastAdaptionFit = self.FFValues.FastAdaptionFit
            self.FFFastAdaptionFitr = self.FFValues.FastAdaptionFitr
            self.FFFastAdaptionCurve = self.FFValues.FastAdaptionFitCurve
            self.FFFastAdaptionCurveT = self.FFValues.FastAdaptionFitCurveT
                           
            self.FFAdaptionIndex = self.FFValues.ChangeIndex 
            
            # FiringType:
            if self.APTimes[-1] < (2/3*(self.StimOffTime-self.StimOnsetTime)+self.StimOnsetTime):
                self.FType1 = 'Fast '
            elif self.APTimes[-1] > (2/3*(self.StimOffTime-self.StimOnsetTime)+self.StimOnsetTime):
                self.FType1 = 'Continuous '   
            elif self.FirstSpikeLatency > 100:
                self.FType1 = 'Delayed '
            else:
                self.FType1 = ''  
                
            if self.FirstFreqency > 80 and self.FFFastAdaptionFit < -0.05: #and self.APBaseFastAdaptionFit < -0.05:
                 self.FType1 = 'BurstOnset '
            if self.FirstFreqency > 80 and self.APBaseFastAdaptionFit < -0.1:
                 self.FType1 = 'BurstOnset '
            if self.FFFastAdaptionFit < -0.05 and self.APBaseFastAdaptionFit < -0.1:
                 self.FType1 = 'BurstOnset '
                 
                 
            self.FType2 = '?'
            if (self.APBaseSlowAdaptionFit < -0.003 and self.FFSlowAdaptionFit < 0.002):#\
#            or -0.05 > self.APBaseFastAdaptionFit < -0.01:
                self.FType2 = 'Adapting'
            elif self.APBaseSlowAdaptionFit < np.absolute(0.003) and self.FFSlowAdaptionFit < 0.002:
                self.FType2 = 'Non-Adapting'
            if self.FFSlowAdaptionFit > 0.002 or self.FFFastAdaptionFit >0.002: 
                self.FType2 = 'Accelerating'
            if self.FiringFrequencyMean > 75 and self.APNum > 30 and self.FirstFreqency > 100:
                self.FType2 = 'Fast Spiking'
            if any(self.InstFiringFreq) > (5*np.std(self.InstFiringFreq)+np.mean(self.InstFiringFreq)):
                self.FType2 = 'Irregular Spiking'
#            if any(self.APs.APIntervals) > ((self.StimOffTime-self.StimOnsetTime)/10):
#                self.FType2 = 'Stuttering'   
            
            # Previous:
#            if self.APBaseSlowAdaptionFit > -0.004 and self.APBaseSlowAdaptionFit > -0.03 \
#            and self.FFSlowAdaptionFit >-0.01 and self.FFFastAdaptionFit >-0.03 :
#                self.FType2 = 'Non-Adapting'
#            else:
#                self.FType2 = 'Adapting'
#            if self.FFSlowAdaptionFit > 0.001 or self.FFFastAdaptionFit >0.0001 :
#                self.FType2 = 'Accelerating'
#            if self.FiringFrequencyMean > 50 and self.APNum > 30 and self.FirstFreqency > 100:
#                self.FType2 = 'Fast Spiking'
                
            self.FiringType = self.FType1 + self.FType2
            # Plotting: 
            # Prevent from popping up:
            if self.PrintShow <= 1:
                plt.ioff()
            else:
                plt.ion()
            if self.PrintShow >=1:
                self.Figure = plt.figure()
#                self.Figure.set_dpi(300)
#                self.Figure.set_size_inches(11.69, 8.27, forward=True)
            
            # Subplotting:
            self.gs = gridspec.GridSpec(2, 2)
            self.gs.update(left=0.1, bottom= 0.1, top = 0.9, right=0.9, wspace=0.25, hspace = 0.35)
            
            self.ax = plt.subplot(self.gs[1,:])
            self.axFreq = plt.subplot(self.gs[0,0])
            self.axAmp = plt.subplot(self.gs[0,1])
            
            # PlotCalculations: 
            self.LatencyPlot = np.zeros(shape=(2,2))
            self.LatencyPlot[0,0] = self.StimOnsetTime
            self.LatencyPlot[0,1] = self.StimOnsetTime+self.FirstSpikeLatency
            self.LatencyPlot[1,0] = self.ThresholdsVm[0]+3
            self.LatencyPlot[1,1] = self.ThresholdsVm[0]+3
            
            # Plotting Wave
            self.PlotWave = [None] * (4)
            self.PlotWave[0], = self.ax.plot(self.Time,self.Wave,'k')
            self.PlotWave[1], = self.ax.plot(self.APTimes,self.APpeaks,'o',color = [0.6, 0.043, 0.215])
            self.PlotWave[2], = self.ax.plot(self.ThresholdTimes,self.ThresholdsVm,'o',color = [0.023, 0.674, 0.258])    
            self.PlotWave[3], = self.ax.plot(self.LatencyPlot[0,:],self.LatencyPlot[1,:],'--',color = [0.309, 0.321, 0.058])
            # Annotating Wave:
            self.AnnotWave = [None] * (4)
            self.AnnotWave[0] = self.ax.annotate("AP-Base Amp: %.1f mV" % self.APBaseAmplitude,xy=(self.APTimes[0],self.APpeaks[0]),xytext=(5,-10),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[1] = self.ax.annotate("AP-Thres Amp: %.1f mV" % self.APThresAmplitude,xy=(self.APTimes[0],self.ThresholdsVm[0]),xytext=(5,30),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[2] = self.ax.annotate("First Spike Latency: %.1f ms" % self.FirstSpikeLatency,xy=(self.LatencyPlot[0,0],self.LatencyPlot[1,0]),xytext=(1,5),color = [0.309, 0.321, 0.058],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[3] = self.ax.annotate("Firing Duration: %.1f ms" % self.FiringDuration,xy=(self.StimOnsetTime,self.Baseline),xytext=(30,5),color = [0.309, 0.321, 0.058],xycoords='data',textcoords='offset points', fontsize = 6)
            
            # Plotting Frequency
            self.PlotFreq = [None] * (3)
            self.PlotFreq[0] = self.axFreq.plot(self.FiringFreqPosi,self.FiringFrequencies,'o', color = [0.176, 0.580, 0.843])
            self.PlotFreq[1] = self.axAmp.plot(self.FFFastAdaptionCurveT,self.FFFastAdaptionCurve,color = [0.6, 0.043, 0.215])
            self.PlotFreq[2] = self.axFreq.plot(self.FFSlowAdaptionFitCurveT,self.FFSlowAdaptionFitCurve, color = [0.968, 0.658, 0.149])
            
            # Annotating Frequency
            self.AnnotFreq = [None] * (3)
            self.AnnotFreq[0] = self.axFreq.annotate("Fast Adaption: %.3f" % self.FFFastAdaptionFit,xy=(self.FiringFreqPosi[0],np.max(self.FiringFrequencies)),xytext=(20,-30),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotFreq[1] = self.axFreq.annotate("Slow Adaption: %.3f" % self.FFSlowAdaptionFit,xy=(self.FiringFreqPosi[0],np.max(self.FiringFrequencies)),xytext=(20,-20),color = [0.968, 0.658, 0.149],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotFreq[2] = self.axFreq.annotate("Adaption Index: %.3f" % self.FFAdaptionIndex,xy=(self.FiringFreqPosi[0],np.max(self.FiringFrequencies)),xytext=(20,-40),color = [0, 0, 0],xycoords='data',textcoords='offset points', fontsize = 6)
            
            # Plotting Amplitude
            self.PlotAmp = [None] * (6)
            self.PlotAmp[0] = self.axAmp.plot(self.AmplitudesPosition,self.APBase,'o',color = [0.176, 0.580, 0.843])
            self.PlotAmp[1] = self.axAmp.plot(self.APBaseFastAdaptionCurveT,self.APBaseFastAdaptionCurve,color = [0.6, 0.043, 0.215])
            self.PlotAmp[2] = self.axAmp.plot(self.APBaseSlowAdaptionFitCurveT,self.APBaseSlowAdaptionFitCurve,color = [0.968, 0.658, 0.149])
            self.PlotAmp[3] = self.axAmp.plot(self.AmplitudesPosition,self.APThres,'o',color = [0.176, 0.843, 0.435])
            self.PlotAmp[4] = self.axAmp.plot(self.APThresFastAdaptionCurveT,self.APThresFastAdaptionCurve,color = [0.6, 0.043, 0.215])
            self.PlotAmp[5] = self.axAmp.plot(self.APThresSlowAdaptionFitCurveT,self.APThresSlowAdaptionFitCurve,color = [0.968, 0.658, 0.149])
            # Annotating Amplitude
            self.AnnotAmp = [None] * (6)
            self.AnnotAmp[0] = self.axAmp.annotate("AP-Base Fast Adaption: %.3f" % self.APBaseFastAdaptionFit,xy=(self.APTimes[0],40),xytext=(5,5),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAmp[1] = self.axAmp.annotate("AP-Base Slow Adaption: %.3f" % self.APBaseSlowAdaptionFit,xy=(self.APTimes[0],50),xytext=(5,5),color = [0.176, 0.580, 0.843],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAmp[2] = self.axAmp.annotate("AP-Base Adaption Index: %.3f" % self.APBaseAdaptionIndex,xy=(self.APTimes[0],30),xytext=(5,5),color = [0, 0, 0],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAmp[3] = self.axAmp.annotate("AP-Thres Fast Adaption: %.3f" % self.APThresFastAdaptionFit,xy=(self.APTimes[0],10),xytext=(5,5),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAmp[4] = self.axAmp.annotate("AP-Thres Slow Adaption: %.3f" % self.APThresSlowAdaptionFit,xy=(self.APTimes[0],20),xytext=(5,5),color = [0.176, 0.843, 0.435],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotAmp[5] = self.axAmp.annotate("AP-Thres Adaption Index: %.3f" % self.APThresAdaptionIndex,xy=(self.APTimes[0],0),xytext=(5,5),color = [0, 0, 0],xycoords='data',textcoords='offset points', fontsize = 6)

            # Set Apperances:
            self.ax.set_xlabel('ms')
            self.ax.set_ylabel('mV')
            self.ax.set_xlim(self.Time[0],self.Time[-1])
    
            self.axFreq.set_ylabel('Hz')
            self.axFreq.set_xlabel('AP Number')
            self.axFreq.set_xlim(self.FiringFreqPosi[0],self.FiringFreqPosi[-1])
            self.axFreq.set_ylim(0,ddhelp.roundup(np.max(self.FiringFrequencies)))
            self.axFreq.set_xticks(np.arange(np.min(self.FiringFreqPosi), np.max(self.FiringFreqPosi)+1, 1.0))
        
            self.axAmp.set_ylabel('mV')
            self.axAmp.set_xlabel('ms')
            self.axAmp.set_xlim(self.AmplitudesPosition[0],self.AmplitudesPosition[-1])
            self.axAmp.set_ylim(0,ddhelp.roundup(np.max(self.APBase)+5))
        else:
            self.OnsetCell = 1
            self.FiringType = 'Onset Cell'
            
            self.APBaseAmplitude = self.APpeaks[0] - self.Baseline
            self.APBaseSlowAdaptionFit = np.nan
            self.APBaseSlowAdaptionFitr = np.nan
            self.APBaseSlowAdaptionFitCurve = np.nan
            self.APBaseFastAdaptionFit = np.nan
            self.APBaseFastAdaptionFitr = np.nan
            self.APBaseFastAdaptionFitCurve = np.nan
            self.APBaseAdaptionIndex = np.nan
            self.APThresAmplitude = self.APpeaks[0] - self.ThresholdsVm[0]
            self.APThresAdaptionValues = np.nan
            self.APThresSlowAdaptionFit = np.nan
            self.APThresSlowAdaptionFitr = np.nan
            self.APThresSlowAdaptionFitCurve = np.nan
            self.APThresFastAdaptionFit = np.nan
            self.APThresFastAdaptionFitr = np.nan
            self.APThresFastAdaptionFitCurve = np.nan
            self.APThresAdaptionIndex = np.nan
            self.FiringDuration = np.nan
            self.FiringFrequencyMean  = np.nan
            self.FirstFreqency = np.nan
            self.FirstSpikeLatency = self.APTimes[0]-self.StimOnsetTime
            self.FFSlowAdaptionFit = np.nan
            self.FFSlowAdaptionFitr = np.nan
            self.FFSlowAdaptionFitCurve = np.nan
            self.FFFastAdaptionFitCurve = np.nan
            self.FFFastAdaptionFit = np.nan
            self.FFFastAdaptionFitr = np.nan
            self.FFAdaptionIndex = np.nan
            
            #PlotWave:
            self.gs = gridspec.GridSpec(2, 2)
            self.gs.update(left=0.1, bottom= 0.1, top = 0.9, right=0.9, wspace=0.25, hspace = 0.35)
            
            self.ax = plt.subplot(self.gs[1,:])
            self.axFreq = plt.subplot(self.gs[0,0])
            self.axAmp = plt.subplot(self.gs[0,1])
            
            self.LatencyPlot = np.zeros(shape=(2,2))
            self.LatencyPlot[0,0] = self.StimOnsetTime
            self.LatencyPlot[0,1] = self.StimOnsetTime+self.FirstSpikeLatency
            self.LatencyPlot[1,0] = self.ThresholdsVm[0]+3
            self.LatencyPlot[1,1] = self.ThresholdsVm[0]+3
            
            self.PlotWave = [None] * (4)
            self.PlotWave[0], = self.ax.plot(self.Time,self.Wave,'k')
            self.PlotWave[1], = self.ax.plot(self.APTimes,self.APpeaks,'o',color = [0.6, 0.043, 0.215])
            self.PlotWave[2], = self.ax.plot(self.ThresholdTimes,self.ThresholdsVm,'o',color = [0.023, 0.674, 0.258])    
            self.PlotWave[3], = self.ax.plot(self.LatencyPlot[0,:],self.LatencyPlot[1,:],'--',color = [0.309, 0.321, 0.058])
            
            # Annotating Wave:
            self.AnnotWave = [None] * (2)
            self.AnnotWave[0] = self.ax.annotate("AP-Base Amp: %.1f mV" % self.APBaseAmplitude,xy=(self.APTimes[0],self.APpeaks[0]),xytext=(5,-10),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            self.AnnotWave[1] = self.ax.annotate("AP-Thres Amp: %.1f mV" % self.APThresAmplitude,xy=(self.APTimes[0],self.ThresholdsVm[0]),xytext=(5,30),color = [0.6, 0.043, 0.215],xycoords='data',textcoords='offset points', fontsize = 6)
            
            # Get Other Stuff
#            self.PlotFreq = [None] 
##            self.PlotFreq[0] = self.axFreq.plot([self.StimOnsetTime,self.StimOnsetTime+1],[self.APpeaks[0],self.APpeaks[0]+1])
#            self.AnnotFreq = [None] 
##            self.AnnotFreq[0] = self.axFreq.annotate('', xy=(self.StimOnsetTime,self.APpeaks[0]),xytext=(20,-20),color = [0.968, 0.658, 0.149],xycoords='data',textcoords='offset points', fontsize = 6)
#            
#            self.PlotAmp = [None] 
##            self.PlotAmp[0] = self.axAmp.plot(self.StimOnsetTime,self.APpeaks[0])
#            self.AnnotAmp = [None] 
##            self.AnnotAmp[0] = self.axAmp.annotate(" ", xy=(self.StimOnsetTime,self.APpeaks[0]),xytext=(5,5),color = [0.176, 0.580, 0.843],xycoords='data',textcoords='offset points', fontsize = 6)
            

    
''' Extraced AP Values '''
class APValues:
    def __init__ (self,MainValues,Names,NumWaves):
        self.MainValues = MainValues
        self.WaveNames = Names
        self.NumWaves = NumWaves
        
        # Names Of AP and AHP:
        i = 0
        self.APAll = [None] * self.NumWaves
        self.AHPAll = [None] * self.NumWaves
        if self.NumWaves > 1:
            while i < self.NumWaves:
                self.APAll[i] = self.MainValues[i].APType
                self.AHPAll[i] = self.MainValues[i].AHPType
                i += 1
                try:
                    self.APType = max(k for k,v in Counter(self.APAll).most_common(1) if v>1)
                    self.AHPType = max(k for k,v in Counter(self.AHPAll).most_common(1) if v>1)
                except:
                     self.APType = self.MainValues[0].APType
                     self.AHPType = self.MainValues[0].AHPType
                     continue

        else:
            self.APAll = [self.MainValues[0].APType]
            self.AHPAll = [self.MainValues[0].AHPType]
            self.APType = self.MainValues[0].APType
            self.AHPType = self.MainValues[0].AHPType    

        # AP Values:
        self.NumAPs =  ddhelp.Extract(self.MainValues,'NumAPs')
        self.APAmplitudeBaseline = ddhelp.Extract(self.MainValues,'APAmpBaseline')
        self.APAmplitudeThreshold = ddhelp.Extract(self.MainValues,'APAmpThres')
        self.Threshold = ddhelp.Extract(self.MainValues,'ThresholdVm')
        self.APTtP = ddhelp.Extract(self.MainValues,'TimeToPeak')
        self.Latency = ddhelp.Extract(self.MainValues,'APLatency')
        self.SlopeRise = ddhelp.Extract(self.MainValues,'SlopeRise')
        self.SlopeDecay = ddhelp.Extract(self.MainValues,'SlopeDecay')
        self.HalfWidth = ddhelp.Extract(self.MainValues,'HalfWidth')
        
        # AHP Values:
        self.fAHPVm = ddhelp.Extract(self.MainValues,'fAHPVm')
        self.fAHPTtP = ddhelp.Extract(self.MainValues,'fAHPTtP')
        self.ADPVm = ddhelp.Extract(self.MainValues,'ADPVm')
        self.ADPTtP = ddhelp.Extract(self.MainValues,'ADPTtP')
        self.mAHPVm = ddhelp.Extract(self.MainValues,'mAHPVm')
        self.mAHPTtP = ddhelp.Extract(self.MainValues,'mAHPTtP')
        self.sAHPAmp = ddhelp.Extract(self.MainValues,'sAHPAmplitude')
        self.AHPArea = ddhelp.Extract(self.MainValues,'AHPArea')
        
        # Burst and Change Values:
        self.BurstDuration = ddhelp.Extract(self.MainValues,'BurstDuration')
        self.BurstArea = ddhelp.Extract(self.MainValues,'BurstArea')
        self.BurstAHPVm = ddhelp.Extract(self.MainValues,'BurstAHPVm')
        self.APAmpBaseChange = ddhelp.Extract(self.MainValues,'APAmpBaselineChange')
        self.APAmpThesChange = ddhelp.Extract(self.MainValues,'APAmpThresChange')
        self.ThresChange = ddhelp.Extract(self.MainValues,'ThresholdVmChange')
        self.APTtPChange = ddhelp.Extract(self.MainValues,'TimeToPeakChange')
        self.SlopeRiseChange = ddhelp.Extract(self.MainValues,'SlopeRiseChange')
        self.SlopeDecayChange = ddhelp.Extract(self.MainValues,'SlopDecayChange')   
        self.HalfWidthChange = ddhelp.Extract(self.MainValues,'HalfWidthChange')
        self.BurstAHPVmChange = ddhelp.Extract(self.MainValues,'BurstAHPVmChange')
        
        self.APToTake = [None]
        
        # Table: 
        self.APType1 = [None]*self.NumWaves
        self.AHPType1 = [None]*self.NumWaves
        self.Names =[None]*self.NumWaves
        
        if self.NumWaves <2:
            self.Names[0] = self.WaveNames
            self.APType1[0] = self.APType
            self.AHPType1[0] = self.AHPType
        else:
            i = 0
            while i < NumWaves:
                self.Names[i] = self.WaveNames[i]
                self.APType1[i] = self.MainValues[i].APType
                self.AHPType1[i] = self.MainValues[i].AHPType
                i +=1
            
        self.BaselineMean = ddhelp.Extract(self.MainValues,'Baseline') 
        self.BaselineSD = ddhelp.Extract(self.MainValues,'BaselineSD') 
#        self.APPeak =   ddhelp.Extract(self.MainValues,'APPeak')
        
        self.Header = ['WaveName','Baseline[mV]','BaselineSD','APType','NumAPs',\
                       'APBaseAmp[mV]','Threshold[mV]','APThresAmp[mV]','TimeToPeak[ms]',\
                       'HalfWidth[ms]','SlopeRise[mV/ms]','SlopeDecay[mV/ms]','Latency[ms]',\
                       'AHPType','fAHP[Vm]','fAHPTtP[ms]','ADP[Vm]','ADPTtP[ms]','mAHP[Vm]',\
                       'mAHPTtP[ms]','AHPArea','sAHPAmp[mV]','BurstAHP[mV]','APBaseAmpChange','ThresholdChange',\
                       'APThresAmpChange','TimeToPeakChange','HalfWidthChange','SlopeRiseChange',\
                       'SlopeDecayChange','BurstAHPChange']#,'1','x','y','z']
        
        self.Table = pd.DataFrame(self.Names)
        self.Table = pd.concat([self.Table,\
                                pd.DataFrame(self.BaselineMean.All),\
                                pd.DataFrame(self.BaselineSD.All),\
                                pd.DataFrame(self.APType1),\
                                pd.DataFrame(self.NumAPs.All),\
                                pd.DataFrame(self.APAmplitudeBaseline.All),\
                                pd.DataFrame(self.Threshold.All),\
                                pd.DataFrame(self.APAmplitudeThreshold.All),\
                                pd.DataFrame(self.APTtP.All),\
                                pd.DataFrame(self.HalfWidth.All),\
                                pd.DataFrame(self.SlopeRise.All),\
                                pd.DataFrame(self.SlopeDecay.All),\
                                pd.DataFrame(self.Latency.All),\
                                pd.DataFrame(self.AHPType1),\
                                pd.DataFrame(self.fAHPVm.All),\
                                pd.DataFrame(self.fAHPTtP.All),\
                                pd.DataFrame(self.ADPVm.All),\
                                pd.DataFrame(self.ADPTtP.All),\
                                pd.DataFrame(self.mAHPVm.All),\
                                pd.DataFrame(self.mAHPTtP.All),\
                                pd.DataFrame(self.AHPArea.All),\
                                pd.DataFrame(self.sAHPAmp.All),\
                                pd.DataFrame(self.BurstAHPVm.All),\
                                pd.DataFrame(self.APAmpBaseChange.All),\
                                pd.DataFrame(self.ThresChange.All),\
                                pd.DataFrame(self.APAmpThesChange.All),\
                                pd.DataFrame(self.APTtPChange.All),\
                                pd.DataFrame(self.HalfWidthChange.All),\
                                pd.DataFrame(self.SlopeRiseChange.All),\
                                pd.DataFrame(self.SlopeDecayChange.All),\
                                pd.DataFrame(self.BurstAHPVmChange.All),\
                                ],axis=1) #join_axes=[self.Table.index]
        
        self.Table.columns = self.Header
                                

        
''' Plot AP Values: '''
class PlotAPs:    
    def __init__ (self,Waves,Values,PrintShow = 0,CellName = 'NA'): 
        self.SingleWaves = Waves
        self.Waves = [None] * len(self.SingleWaves)
        self.WaveNames = [None] * len(self.SingleWaves)
        self.Values = Values  
        self.PrintShow = PrintShow
        self.CellName = CellName
        
        # Extract Plotting Waves:
        i = 0
        self.Waves = [None] * len(self.SingleWaves)
        self.NumWholeWave = [None] * len(self.SingleWaves)
        self.NumWholeWaveAnnot = [None] * len(self.SingleWaves)
        self.NumAPWave = [None] * len(self.SingleWaves)
        self.NumAPWaveAnnot = [None] * len(self.SingleWaves)
        self.NumAHPWave = [None] * len(self.SingleWaves)
        self.NumAHPWaveAnnot = [None] * len(self.SingleWaves)
        
        while i < len(Waves):
            self.Waves[i] = ddPlotting.ExtractPlotting(self.SingleWaves[i])
            self.WaveNames[i] = self.SingleWaves[i].WaveNames
            self.NumWholeWave[i] = len(self.Waves[i].Waves)
            self.NumWholeWaveAnnot[i] = len(self.Waves[i].Annot)
            self.NumAPWave[i] = len(self.Waves[i].WavesAP)
            self.NumAPWaveAnnot[i] = len(self.Waves[i].AnnotAP)
            self.NumAHPWave[i] = len(self.Waves[i].WavesAHP)
            self.NumAHPWaveAnnot[i] = len(self.Waves[i].AnnotAHP)
            i += 1
        
        self.NumCells = len(self.Waves)
        if self.NumCells > 10:
            self.NumCells = 10
        
        # Figure: 
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
            self.Figure = plt.figure()
            self.Figure.set_size_inches(11.69, 8.27, forward=True)
        if self.PrintShow == 1:
            self.Figure.set_dpi(300)

        self.AxWhole = [None] * (self.NumCells)
        self.AxAP = [None] * (self.NumCells)
        self.AxAHP = [None] * (self.NumCells)
        self.gs = gridspec.GridSpec(2, 5)
        self.gs.update(left=0.05, bottom= 0.05, top = 0.7, right=0.95, wspace=0.2)
        self.Markersize = 5

        # Set Grid for Plotting: Row/Colum
        if self.NumCells >= 1:
            self.inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec = self.gs[0], wspace=0.2, hspace=0.2)
            self.AxWhole[0] = plt.subplot(self.inner_grid[1,:])
            self.AxAP[0] = plt.subplot(self.inner_grid[0,0])
            self.AxAHP[0] = plt.subplot(self.inner_grid[0,1])
        if self.NumCells >= 2:
            self.inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec = self.gs[1], wspace=0.2, hspace=0.2)
            self.AxWhole[1] = plt.subplot(self.inner_grid[1,:])
            self.AxAP[1] = plt.subplot(self.inner_grid[0,0])
            self.AxAHP[1] = plt.subplot(self.inner_grid[0,1])
        if self.NumCells >= 3:

            self.inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec = self.gs[2], wspace=0.2, hspace=0.2)
            self.AxWhole[2] = plt.subplot(self.inner_grid[1,:])
            self.AxAP[2] = plt.subplot(self.inner_grid[0,0])
            self.AxAHP[2] = plt.subplot(self.inner_grid[0,1])
        if self.NumCells >= 4:

            self.inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec = self.gs[3], wspace=0.2, hspace=0.2)
            self.AxWhole[3] = plt.subplot(self.inner_grid[1,:])
            self.AxAP[3] = plt.subplot(self.inner_grid[0,0])
            self.AxAHP[3] = plt.subplot(self.inner_grid[0,1])
        if self.NumCells >= 5:
            self.inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec = self.gs[4], wspace=0.2, hspace=0.2)
            self.AxWhole[4] = plt.subplot(self.inner_grid[1,:])
            self.AxAP[4] = plt.subplot(self.inner_grid[0,0])
            self.AxAHP[4] = plt.subplot(self.inner_grid[0,1])

        if self.NumCells >= 6:
            self.inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec = self.gs[5], wspace=0.2, hspace=0.2)
            self.AxWhole[5] = plt.subplot(self.inner_grid[1,:])
            self.AxAP[5] = plt.subplot(self.inner_grid[0,0])
            self.AxAHP[5] = plt.subplot(self.inner_grid[0,1])

        if self.NumCells >= 7:
            self.inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec = self.gs[6], wspace=0.2, hspace=0.2)
            self.AxWhole[6] = plt.subplot(self.inner_grid[1,:])
            self.AxAP[6] = plt.subplot(self.inner_grid[0,0])
            self.AxAHP[6] = plt.subplot(self.inner_grid[0,1])

        if self.NumCells >= 8:
            self.inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec = self.gs[7], wspace=0.2, hspace=0.2)
            self.AxWhole[7] = plt.subplot(self.inner_grid[1,:])
            self.AxAP[7] = plt.subplot(self.inner_grid[0,0])
            self.AxAHP[7] = plt.subplot(self.inner_grid[0,1])

        if self.NumCells >= 9:
            self.inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec = self.gs[8], wspace=0.2, hspace=0.2)
            self.AxWhole[8] = plt.subplot(self.inner_grid[1,:])
            self.AxAP[8] = plt.subplot(self.inner_grid[0,0])
            self.AxAHP[8] = plt.subplot(self.inner_grid[0,1])

        if self.NumCells >= 10:
            self.inner_grid = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec = self.gs[9], wspace=0.2, hspace=0.2)
            self.AxWhole[9] = plt.subplot(self.inner_grid[1,:])
            self.AxAP[9] = plt.subplot(self.inner_grid[0,0])
            self.AxAHP[9] = plt.subplot(self.inner_grid[0,1])

        
        # Plot Waves:
        i = 0 
        while i < self.NumCells: 
            j = 0
            while j < self.NumWholeWave[i]:
                # Set Markersize
                if self.Waves[i].Waves[j].get_marker() == '.' or self.Waves[i].Waves[j].get_marker() == 'o':
                    msset = self.Markersize
                else:
                    msset = 1.0 
                # Plotting    
                self.AxWhole[i].plot(self.Waves[i].Waves[j].get_data()[0], self.Waves[i].Waves[j].get_data()[1],self.Waves[i].Waves[j].get_marker(), linestyle = self.Waves[i].Waves[j].get_linestyle(), color = self.Waves[i].Waves[j].get_color(), ms = msset)
                # Axes:
                self.AxWhole[i].set_xlim(0,800)
                self.AxWhole[i].xaxis.set_ticks(np.arange(0,800,200))
                plt.setp(self.AxWhole[i].get_xticklabels(), fontsize = 4)
                self.AxWhole[i].set_xlabel('ms',fontsize = 6, labelpad=0.5)
                self.AxWhole[i].set_ylabel('mV',fontsize = 6, rotation = 90, labelpad = 0.5)
                plt.setp(self.AxWhole[i].get_yticklabels(),rotation = 'vertical', fontsize = 6)
                self.AxWhole[i].spines["top"].set_visible(False)
                self.AxWhole[i].spines["right"].set_visible(False)
                j +=1
            k = 0
            while k < self.NumWholeWaveAnnot[i]:
                # Annotating:
                self.AxWhole[i].annotate(self.Waves[i].Annot[k]._text, self.Waves[i].Annot[k].xy,self.Waves[i].Annot[k].xyann, self.Waves[i].Annot[k].xycoords, self.Waves[i].Annot[k]._textcoords, fontsize = 4,color = self.Waves[i].Annot[k].get_color())
                k +=1
        
            i +=1

        # Plot APWaves:
        i = 0 
        while i < self.NumCells: 
            j = 0
            while j < self.NumAPWave[i]:
                # Set Markersize
                if self.Waves[i].WavesAP[j].get_marker() == '.' or self.Waves[i].WavesAP[j].get_marker() == 'o':
                    msset = self.Markersize
                else:
                    msset = 1.0 
                # Plotting    
                self.AxAP[i].plot(self.Waves[i].WavesAP[j].get_data()[0], self.Waves[i].WavesAP[j].get_data()[1],self.Waves[i].WavesAP[j].get_marker(), linestyle = self.Waves[i].WavesAP[j].get_linestyle(), color = self.Waves[i].WavesAP[j].get_color(), ms = msset)
                # Axes:
                plt.setp(self.AxAP[i].get_xticklabels(), fontsize = 4)
                self.AxAP[i].set_ylabel('mV',fontsize = 6, rotation = 90, labelpad=0.5)
                plt.setp(self.AxAP[i].get_yticklabels(),rotation = 'vertical', fontsize = 6)
                self.AxAP[i].spines["top"].set_visible(False)
                self.AxAP[i].spines["right"].set_visible(False)
                self.AxAP[i].set_title(self.WaveNames[i]+": %i " % i, fontsize = 6, y=0.9)
                j +=1
                
            k = 0
            while k < self.NumAPWaveAnnot[i]:
                # Annotating:
                self.AxAP[i].annotate(self.Waves[i].AnnotAP[k]._text, self.Waves[i].AnnotAP[k].xy,self.Waves[i].AnnotAP[k].xyann, self.Waves[i].AnnotAP[k].xycoords, self.Waves[i].AnnotAP[k]._textcoords, fontsize = 4,color = self.Waves[i].AnnotAP[k].get_color())
                k +=1
        
            i +=1
            
        # Plot AHPWaves:
        i = 0 
        while i < self.NumCells: 
            j = 0
            while j < self.NumAHPWave[i]:
                # Set Markersize
                if self.Waves[i].WavesAHP[j].get_marker() == '.' or self.Waves[i].WavesAHP[j].get_marker() == 'o':
                    msset = self.Markersize
                else:
                    msset = 1.0 
                # Plotting    
                self.AxAHP[i].plot(self.Waves[i].WavesAHP[j].get_data()[0], self.Waves[i].WavesAHP[j].get_data()[1],self.Waves[i].WavesAHP[j].get_marker(), linestyle = self.Waves[i].WavesAHP[j].get_linestyle(), color = self.Waves[i].WavesAHP[j].get_color(), ms = msset)
                # Axes:
                plt.setp(self.AxAHP[i].get_xticklabels(), fontsize = 4)
                self.AxAHP[i].set_ylabel('mV',fontsize = 6, rotation = 90, labelpad=0.5)
                plt.setp(self.AxAHP[i].get_yticklabels(),rotation = 'vertical', fontsize = 6)
                self.AxAHP[i].spines["top"].set_visible(False)
                self.AxAHP[i].spines["right"].set_visible(False)
                j +=1
                
            k = 0
            while k < self.NumAHPWaveAnnot[i]:
                # Annotating:
                self.AxAHP[i].annotate(self.Waves[i].AnnotAHP[k]._text, self.Waves[i].AnnotAHP[k].xy,self.Waves[i].AnnotAHP[k].xyann, self.Waves[i].AnnotAHP[k].xycoords, self.Waves[i].AnnotAHP[k]._textcoords, fontsize = 4,color = self.Waves[i].AnnotAHP[k].get_color())
                k +=1
        
            i +=1
            
        # Plot APValues: 
        # GridSpace:
        self.gsValues = gridspec.GridSpec(1,9)
        self.gsValues.update(left=0.05, bottom= 0.75, top = 0.95, right=0.65, wspace=0.75) 
        
        # Lists for Loop:
        self.List = [None] * 9
        self.ListName = [None] * 9
        self.axValues = [None] * 9
        self.YLimMin =  [None] * 9
        self.YLimMax =  [None] * 9
        self.List[0] = self.Values.NumAPs
        self.ListName[0] = 'Number Aps'
        self.List[1] = self.Values.APAmplitudeBaseline
        self.ListName[1] = 'Ap-Base [mV]'
        self.List[2] = self.Values.APAmplitudeThreshold
        self.ListName[2] = 'Ap-Thres [mV]'
        self.List[3] = self.Values.Threshold
        self.ListName[3] = 'Threshold [mV]'
        self.List[4] = self.Values.Latency
        self.ListName[4] = 'Latency [ms]'
        self.List[5] = self.Values.APTtP
        self.ListName[5] = 'Time To Peak'
        self.List[6] = self.Values.HalfWidth
        self.ListName[6] = 'Half Width [ms]'
        self.List[7] = self.Values.SlopeRise
        self.ListName[7] = 'Slope Rise [mV/s]'
        self.List[8] = self.Values.SlopeDecay
        self.ListName[8] = 'Slope Decay [mV/s]'
        
        i = 0
        while i < 9 :
            # Plotting:
            self.axValues[i] = plt.subplot(self.gsValues[0,i])
            self.axValues[i].plot(np.ones(len(self.List[i].All)),self.List[i].All,'ok') 
            self.axValues[i].errorbar(1,self.List[i].Mean,self.List[i].SD,linestyle='None', marker='o')
            # PointAnnotations:
            j = 0
            while j < len(self.List[i].All):
                self.axValues[i].annotate('%.0f ' % j,xy = (1,self.List[i].All[j]),xytext=(-8,0),xycoords='data',textcoords='offset points', fontsize = 4)
                j +=1
            # Mean SD:
            self.axValues[i].annotate('%.1f' % self.List[i].Mean,xy = (1,self.List[i].Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 4)
            self.axValues[i].annotate('+/- %.1f' % self.List[i].SD,xy = (1,self.List[i].Mean),xytext=(5,-4.5),xycoords='data',textcoords='offset points', fontsize = 4)
            
            #AxesLimits:
            if min(self.List[i].All) < 0:
                self.YLimMin[i] = ddhelp.roundup(min(self.List[i].All)-10)
            else:
                self.YLimMin[i] = ddhelp.roundup(min(self.List[i].All)-10)
            
            if max(self.List[i].All) < 0:   
                self.YLimMax[i] = ddhelp.roundup(max(self.List[i].All))
            else:
                self.YLimMax[i] = ddhelp.roundup(max(self.List[i].All))
                
            if min(self.List[i].All) > 0 and max(self.List[i].All) < 10:
                self.YLimMin[i] = np.round(min(self.List[i].All)-0.1,decimals = 1)    
                self.YLimMax[i] = np.round(max(self.List[i].All)+0.1,decimals = 1)
            
            if self.List[i].SD <= 0.0001:
                self.YLimMin[i] = self.List[i].Mean-1
                self.YLimMax[i] = self.List[i].Mean+1
                
            self.axValues[i].set_ylim ([self.YLimMin[i],self.YLimMax[i]])
            
            #Axes Apparence:
            self.axValues[i].set_ylabel(self.ListName[i],fontsize = 6, rotation = 90, labelpad=0.5)
            plt.setp(self.axValues[i].get_yticklabels(),rotation = 'vertical', fontsize = 6)
            self.axValues[i].set_xlim ([0,3])
            self.axValues[i].set_xticklabels([])
            self.axValues[i].tick_params(axis='x', which='both',bottom='off')      
            self.axValues[i].spines["top"].set_visible(False)
            self.axValues[i].spines["right"].set_visible(False)
            i +=1
        # self.OVTitle = self.CellName2 + '\nof ' + self.CellName +':'
        self.OVTitle = 'AP Properties of ' + self.CellName +':'
        self.axValues[0].set_title(self.OVTitle,fontsize = 14, y = 1.05, fontweight='bold',loc='left')
        
            
        # Plot AHPValues: 
        self.AHPValues = [None]*4
        # GridSpace:
        self.gsAHPValues = gridspec.GridSpec(1,4,width_ratios=[1,2,2,1])
        self.gsAHPValues.update(left=0.675, bottom= 0.75, top = 0.95, right=0.95, wspace=0.95) 
        self.AHPValues[0] = plt.subplot(self.gsAHPValues[0,0])
        self.AHPValues[1] = plt.subplot(self.gsAHPValues[0,1])
        self.AHPValues[2] = plt.subplot(self.gsAHPValues[0,2])
        self.AHPValues[3] = plt.subplot(self.gsAHPValues[0,3])
        
        # AHP Area:
        self.AHPValues[0].plot(np.ones(len(self.Values.AHPArea.All)),self.Values.AHPArea.All,'ok') 
        self.AHPValues[0].errorbar(1,self.Values.AHPArea.Mean,self.Values.AHPArea.SD,linestyle='None', marker='o')
        j = 0
        while j < len(self.Values.AHPArea.All):
            self.AHPValues[0].annotate('%.0f ' % j,xy = (1,self.Values.AHPArea.All[j]),xytext=(-8,0),xycoords='data',textcoords='offset points', fontsize = 4)
            j +=1
        self.AHPValues[0].annotate('%.3f' % self.Values.AHPArea.Mean,xy = (1,self.Values.AHPArea.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 4)
        self.AHPValues[0].annotate('+/- %.3f' % self.Values.AHPArea.SD,xy = (1,self.Values.AHPArea.Mean),xytext=(5,-4.5),xycoords='data',textcoords='offset points', fontsize = 4)
        
        #Axes Apparence:
        self.AHPValues[0].set_ylabel('AHPArea',fontsize = 6, rotation = 90, labelpad=0.5)
        plt.setp(self.AHPValues[0].get_yticklabels(),rotation = 'vertical', fontsize = 6)
        self.AHPValues[0].set_xlim ([0,3])
        #self.AHPValues[0].set_ylim ([np.round(np.nanmin(self.Values.AHPArea.All)-10,decimals = 0),np.round(np.nanmax(self.Values.AHPArea.All)+10,decimals = 0)])
        self.AHPValues[0].set_xticklabels([])
        self.AHPValues[0].tick_params(axis='x', which='both',bottom='off')      
        self.AHPValues[0].spines["top"].set_visible(False)
        self.AHPValues[0].spines["right"].set_visible(False)

        #AHP Plot:
        self.AHPLimitsVm = np.empty((3,3,))
        self.AHPLimitsVm[:] = np.nan
        self.AHPLimitsTtP = np.empty((3,2,))
        self.AHPLimitsTtP[:] = np.nan
        if not np.isnan(self.Values.fAHPVm.Mean):
            fX = [1]*len(self.Values.fAHPVm.All)
            self.AHPValues[1].plot(fX ,self.Values.fAHPVm.All,'ok') 
            self.AHPValues[1].errorbar(1,self.Values.fAHPVm.Mean,self.Values.fAHPVm.SD,linestyle='None', marker='o')
            self.AHPValues[2].plot(fX ,self.Values.fAHPTtP.All,'ok') 
            self.AHPValues[2].errorbar(1,self.Values.fAHPTtP.Mean,self.Values.fAHPTtP.SD,linestyle='None', marker='o')
            self.AHPLimitsVm[0,0] = np.nanmin(self.Values.fAHPVm.All)
            self.AHPLimitsVm[0,1] = np.nanmax(self.Values.fAHPVm.All)
            self.AHPLimitsTtP[0,0] = np.nanmin(self.Values.fAHPTtP.All)
            self.AHPLimitsTtP[0,1] = np.nanmax(self.Values.fAHPTtP.All)
        j = 0
        while j < len(self.Values.fAHPVm.All):
            self.AHPValues[1].annotate('%.0f ' % j,xy = (1,self.Values.fAHPVm.All[j]),xytext=(-8,0),xycoords='data',textcoords='offset points', fontsize = 4)
            self.AHPValues[2].annotate('%.0f ' % j,xy = (1,self.Values.fAHPTtP.All[j]),xytext=(-8,0),xycoords='data',textcoords='offset points', fontsize = 4)
            j +=1
        self.AHPValues[1].annotate('%.1f mV' % self.Values.fAHPVm.Mean,xy = (1,self.Values.fAHPVm.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 4)
        self.AHPValues[1].annotate('+/- %.1f' % self.Values.fAHPVm.SD,xy = (1,self.Values.fAHPVm.Mean),xytext=(5,-4.5),xycoords='data',textcoords='offset points', fontsize = 4)
        self.AHPValues[2].annotate('%.1f ms' % self.Values.fAHPTtP.Mean,xy = (1,self.Values.fAHPTtP.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 4)
        self.AHPValues[2].annotate('+/- %.1f' % self.Values.fAHPTtP.SD,xy = (1,self.Values.fAHPTtP.Mean),xytext=(5,-4.5),xycoords='data',textcoords='offset points', fontsize = 4)
        
        
        if not np.isnan(self.Values.ADPVm.Mean):
            mX = [2]*len(self.Values.ADPVm.All)
            self.AHPValues[1].plot(mX,self.Values.ADPVm.All,'ok') 
            self.AHPValues[1].errorbar(2,self.Values.ADPVm.Mean,self.Values.ADPVm.SD,linestyle='None', marker='o')
            self.AHPValues[2].plot(mX,self.Values.ADPTtP.All,'ok') 
            self.AHPValues[2].errorbar(2,self.Values.ADPTtP.Mean,self.Values.ADPTtP.SD,linestyle='None', marker='o')
            self.AHPLimitsVm[1,0] = np.nanmin(self.Values.ADPVm.All)
            self.AHPLimitsVm[1,1] = np.nanmax(self.Values.ADPVm.All)
            self.AHPLimitsTtP[1,0] = np.nanmin(self.Values.ADPTtP.All)
            self.AHPLimitsTtP[1,1] = np.nanmax(self.Values.ADPTtP.All)
        j = 0
        while j < len(self.Values.ADPVm.All):
            self.AHPValues[1].annotate('%.0f ' % j,xy = (2,self.Values.ADPVm.All[j]),xytext=(-6,0),xycoords='data',textcoords='offset points', fontsize = 4)
            self.AHPValues[2].annotate('%.0f ' % j,xy = (2,self.Values.ADPTtP.All[j]),xytext=(-6,0),xycoords='data',textcoords='offset points', fontsize = 4)
            j +=1
        self.AHPValues[1].annotate('%.1f mV' % self.Values.ADPVm.Mean,xy = (2,self.Values.ADPVm.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 4)
        self.AHPValues[1].annotate('+/- %.1f' % self.Values.ADPVm.SD,xy = (2,self.Values.ADPVm.Mean),xytext=(5,-4.5),xycoords='data',textcoords='offset points', fontsize = 4)
        self.AHPValues[2].annotate('%.1f ms' % self.Values.ADPTtP.Mean,xy = (2,self.Values.ADPTtP.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 4)
        self.AHPValues[2].annotate('+/- %.1f' % self.Values.ADPTtP.SD,xy = (2,self.Values.ADPTtP.Mean),xytext=(5,-4.5),xycoords='data',textcoords='offset points', fontsize = 4)
        
        if not np.isnan(self.Values.mAHPVm.Mean):
            mX = [3]*len(self.Values.mAHPVm.All)
            self.AHPValues[1].plot(mX,self.Values.mAHPVm.All,'ok') 
            self.AHPValues[1].errorbar(3,self.Values.mAHPVm.Mean,self.Values.mAHPVm.SD,linestyle='None', marker='o')
            self.AHPValues[2].plot(mX,self.Values.mAHPTtP.All,'ok') 
            self.AHPValues[2].errorbar(3,self.Values.mAHPTtP.Mean,self.Values.mAHPTtP.SD,linestyle='None', marker='o')
            self.AHPLimitsVm[2,0] = np.nanmin(self.Values.mAHPVm.All)
            self.AHPLimitsVm[2,1] = np.nanmax(self.Values.mAHPVm.All)
            self.AHPLimitsTtP[2,0] = np.nanmin(self.Values.mAHPTtP.All)
            self.AHPLimitsTtP[2,1] = np.nanmax(self.Values.mAHPTtP.All)
        j = 0
        while j < len(self.Values.mAHPVm.All):
            self.AHPValues[1].annotate('%.0f ' % j,xy = (3,self.Values.mAHPVm.All[j]),xytext=(-8,0),xycoords='data',textcoords='offset points', fontsize = 4)
            self.AHPValues[2].annotate('%.0f ' % j,xy = (3,self.Values.mAHPTtP.All[j]),xytext=(-8,0),xycoords='data',textcoords='offset points', fontsize = 4)
            j +=1
            
        self.AHPValues[1].annotate('%.1f mV' % self.Values.mAHPVm.Mean,xy = (3,self.Values.mAHPVm.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 4)
        self.AHPValues[1].annotate('+/- %.1f' % self.Values.mAHPVm.SD,xy = (3,self.Values.mAHPVm.Mean),xytext=(5,-4.5),xycoords='data',textcoords='offset points', fontsize = 4)
        self.AHPValues[2].annotate('%.1f ms' % self.Values.mAHPTtP.Mean,xy = (3,self.Values.mAHPTtP.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 4)
        self.AHPValues[2].annotate('+/- %.1f' % self.Values.mAHPTtP.SD,xy = (3,self.Values.mAHPTtP.Mean),xytext=(5,-4.5),xycoords='data',textcoords='offset points', fontsize = 4)
        
        #Limits:
        self.AHPVmMin = ddhelp.roundup(np.nanmin(self.AHPLimitsVm[:,0])-10)
        self.AHPVmMax = ddhelp.roundup(np.nanmax(self.AHPLimitsVm[:,1]))
        self.AHPTtPMin = np.round(np.nanmin(self.AHPLimitsTtP[:,0]),decimals = 1)-1
        self.AHPTtPMax = np.round(np.nanmax(self.AHPLimitsTtP[:,1]),decimals = 1)+1
        #Axes Apparence AHPVm:
        self.AHPValues[1].set_ylabel('AHP [mv]',fontsize = 6, rotation = 90, labelpad=0.5)
        plt.setp(self.AHPValues[1].get_yticklabels(),rotation = 'vertical', fontsize = 6)
        self.AHPValues[1].set_xlim([0,4])
        self.AHPValues[1].set_ylim([self.AHPVmMin,self.AHPVmMax])
        self.AHPValues[1].set_xticks([1,2,3])
        self.AHPValues[1].set_xticklabels(['fAHP','ADP','mAHP'],rotation = 90,  fontsize=4)
        self.AHPValues[1].tick_params(axis='x', which='both',bottom='off')      
        self.AHPValues[1].spines["top"].set_visible(False)
        self.AHPValues[1].spines["right"].set_visible(False)
        #Axes Apparence AHPTtP:
        self.AHPValues[2].set_ylabel('AHP [ms]',fontsize = 6, rotation = 90, labelpad=0.5)
        plt.setp(self.AHPValues[2].get_yticklabels(),rotation = 'vertical', fontsize = 6)
        self.AHPValues[2].set_xlim([0,4])
        self.AHPValues[2].set_ylim([self.AHPTtPMin,self.AHPTtPMax])
        self.AHPValues[2].set_xticks([1,2,3])
        self.AHPValues[2].set_xticklabels(['fAHP','ADP','mAHP'],rotation = 90, fontsize=4)
        self.AHPValues[2].tick_params(axis='x', which='both',bottom='off')      
        self.AHPValues[2].spines["top"].set_visible(False)
        self.AHPValues[2].spines["right"].set_visible(False)

        # sAHP Amplitude: 
        self.AHPValues[3].plot(np.ones(len(self.Values.sAHPAmp.All)),self.Values.sAHPAmp.All,'ok') 
        self.AHPValues[3].errorbar(1,self.Values.sAHPAmp.Mean,self.Values.sAHPAmp.SD,linestyle='None', marker='o')
        while j < len(self.Values.sAHPAmp.All):
            self.AHPValues[3].annotate('%.0f ' % j,xy = (1,self.Values.sAHPAmp.All[j]),xytext=(-8,0),xycoords='data',textcoords='offset points', fontsize = 4)
            j +=1
        self.AHPValues[3].annotate('%.1f mV' % self.Values.sAHPAmp.Mean,xy = (1,self.Values.sAHPAmp.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 4)
        self.AHPValues[3].annotate('+/- %.1f' % self.Values.sAHPAmp.SD,xy = (1,self.Values.sAHPAmp.Mean),xytext=(5,-4.5),xycoords='data',textcoords='offset points', fontsize = 4)
        #Axes Apparence:
        self.AHPValues[3].set_ylabel('sAHP Amp [mv]',fontsize = 6, rotation = 90, labelpad=0.5)
        plt.setp(self.AHPValues[3].get_yticklabels(),rotation = 'vertical', fontsize = 6)
        self.AHPValues[3].set_xlim ([0,3])
        #self.AHPValues[3].set_ylim ([np.round(np.min(self.Values.sAHPAmp.All)-1,decimals = 0),np.round(np.max(self.Values.sAHPAmp.All)+1,decimals = 0)])
        self.AHPValues[3].set_xticklabels([])
        self.AHPValues[3].tick_params(axis='x', which='both',bottom='off')      
        self.AHPValues[3].spines["top"].set_visible(False)
        self.AHPValues[3].spines["right"].set_visible(False)
        
        #Saving:
        if self.PrintShow == 1:
            self.SavingName = self.CellName+'_APProperties'
            ddPlotting.save(self.SavingName, ext="png", close=True, verbose=True)
            #ddPlotting.save(self.SavingName, ext="svg", close=True, verbose=True)
            plt.close('All')
            plt.ion()
        
''' Extraced Firing Values '''
class FiringValues:
    def __init__ (self,MainValues,Names,NumWaves):
        self.MainValues = MainValues
        self.WaveNames = Names
        self.NumWaves = NumWaves
  
        # Names:
        i = 0
        self.FiringTypeAll = [None] * self.NumWaves
        if self.NumWaves > 1:
            while i < len(self.MainValues):
                self.FiringTypeAll[i] = self.MainValues[i].FiringType
                i += 1        
#            self.FiringType = max(k for k,v in Counter(self.FiringTypeAll).items() if v>1)
            self.FiringTypeII = Counter(self.FiringTypeAll).most_common(1)[0]
            self.FiringTypeI = str(self.FiringTypeII)
            whitelist = set('abcdefghijklmnopqrstuvwxy ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            self.FiringType = ''.join(filter(whitelist.__contains__, self.FiringTypeI))
        else:
            self.FiringType = self.MainValues[0].FiringType
        # Extracted Values: 
        self.FiringDuration = ddhelp.Extract(self.MainValues,'FiringDuration')
        self.FirstSpikeLatency = ddhelp.Extract(self.MainValues,'FirstSpikeLatency')
        self.FirstFiringFreq = ddhelp.Extract(self.MainValues,'FirstFreqency')
        self.FiringFrequency = ddhelp.Extract(self.MainValues,'FiringFrequencyMean')
        self.FiringFreqSlowAcc = ddhelp.Extract(self.MainValues,'FFSlowAdaptionFit')
        self.FiringFreqFastAcc = ddhelp.Extract(self.MainValues,'FFFastAdaptionFit')
        self.FiringFreqIndex = ddhelp.Extract(self.MainValues,'FFAdaptionIndex')
        self.FirstSpikeAmpBase =  ddhelp.Extract(self.MainValues,'APBaseAmplitude')
        self.APBaseSlowAdaption = ddhelp.Extract(self.MainValues,'APBaseSlowAdaptionFit')
        self.APBaseFastAdaption = ddhelp.Extract(self.MainValues,'APBaseFastAdaptionFit')
        self.APBaseAdaptionIndex = ddhelp.Extract(self.MainValues,'APBaseAdaptionIndex')
        self.FirstSpikeAmpThres = ddhelp.Extract(self.MainValues,'APThresAmplitude')
        self.APThresSlowAdaption = ddhelp.Extract(self.MainValues,'APThresSlowAdaptionFit')
        self.APThresFastAdaption = ddhelp.Extract(self.MainValues,'APThresFastAdaptionFit')
        self.APThresAdaptionIndex = ddhelp.Extract(self.MainValues,'APThresAdaptionIndex')

        self.BaselineMean = ddhelp.Extract(self.MainValues,'Baseline') 
        self.BaselineSD = ddhelp.Extract(self.MainValues,'BaselineSD') 
        
        self.NumAPs = ddhelp.Extract(self.MainValues,'APNum')
        self.FiringFreqSlowAccR = ddhelp.Extract(self.MainValues,'FFSlowAdaptionFitr')
        self.FiringFreqFastAccR = ddhelp.Extract(self.MainValues,'FFFastAdaptionFitr')
        self.APBaseSlowAdaptionR = ddhelp.Extract(self.MainValues,'APBaseSlowAdaptionFitr')
        self.APBaseFastAdaptionR = ddhelp.Extract(self.MainValues,'APBaseFastAdaptionFitr')
        self.APThresSlowAdaptionR = ddhelp.Extract(self.MainValues,'APThresSlowAdaptionFitr')
        self.APThresFastAdaptionR = ddhelp.Extract(self.MainValues,'APThresFastAdaptionFitr')
        
        
        # Table: 
        self.FiringType1 = [None] * self.NumWaves
        self.Names  = [None] * self.NumWaves
        
        if self.NumWaves <2:
            self.Names[0] = self.WaveNames
            self.FiringType1[0] = self.FiringType
        else:
            i = 0
            while i < NumWaves:
                self.Names[i] = self.WaveNames[i]
                self.FiringType1[i] = self.MainValues[i].FiringType
                i +=1        
        self.Header = ['WaveName','Baseline[mV]','BaselineSD','FiringType','NumAPs',\
                       'FiringDuration[ms]','FirstSpikeLatency[ms]','FirstFiringFreq[Hz]','FiringFrequency[Hz]',\
                       'FiringFreqFastAccomodation','FiringFreqFastAccomodationR','FiringFreqSlowAccomodation','FiringFreqSlowAccomodationR','FiringFreqIndex','FirstSpikeAmpBase[mV]',\
                       'APBaseFastAdaption','APBaseFastAdaptionR','APBaseSlowAdaption','APBaseSlowAdaptionR','APBaseAdaptionIndex','FirstSpikeAmpThres[mV]',\
                       'APThresFastAdaption','APThresFastAdaptionR','APThresSlowAdaption','APThresSlowAdaptionR','APThresAdaptionIndex']
        
        self.Table = pd.DataFrame(self.Names)
        self.Table = pd.concat([self.Table,\
                                pd.DataFrame(self.BaselineMean.All),\
                                pd.DataFrame(self.BaselineSD.All),\
                                pd.DataFrame(self.FiringType1),\
                                pd.DataFrame(self.NumAPs.All),\
                                pd.DataFrame(self.FiringDuration.All),\
                                pd.DataFrame(self.FirstSpikeLatency.All),\
                                pd.DataFrame(self.FirstFiringFreq.All),\
                                pd.DataFrame(self.FiringFrequency.All),\
                                pd.DataFrame(self.FiringFreqFastAcc.All),\
                                pd.DataFrame(self.FiringFreqFastAccR.All),\
                                pd.DataFrame(self.FiringFreqSlowAcc.All),\
                                pd.DataFrame(self.FiringFreqSlowAccR.All),\
                                pd.DataFrame(self.FiringFreqIndex.All),\
                                pd.DataFrame(self.FirstSpikeAmpBase.All),\
                                pd.DataFrame(self.APBaseFastAdaption.All),\
                                pd.DataFrame(self.APBaseFastAdaptionR.All),\
                                pd.DataFrame(self.APBaseSlowAdaption.All),\
                                pd.DataFrame(self.APBaseSlowAdaptionR.All),\
                                pd.DataFrame(self.APBaseAdaptionIndex.All),\
                                pd.DataFrame(self.FirstSpikeAmpBase.All),\
                                pd.DataFrame(self.APThresFastAdaption.All),\
                                pd.DataFrame(self.APThresFastAdaptionR.All),\
                                pd.DataFrame(self.APThresSlowAdaption.All),\
                                pd.DataFrame(self.APThresSlowAdaptionR.All),\
                                
                                pd.DataFrame(self.APThresAdaptionIndex.All),\
                                ],axis=1) # join_axes=[self.Table.index]
        
        self.Table.columns = self.Header
        
''' Plot Firing Values: '''
class PlotFiring:    
    def __init__ (self,Waves,Values,PrintShow = 0,CellName = 'NA'): 
        self.SingleWaves = Waves
        self.Waves = [None] * len(self.SingleWaves)
        self.WaveNames = [None] * len(self.SingleWaves)
        self.Values = Values  
        self.PrintShow = PrintShow
        self.CellName = CellName
        
        # Extract Plotting Waves:
        i = 0
        self.Waves = [None] * len(self.SingleWaves)
        self.NumWholeWave = [None] * len(self.SingleWaves)
        self.NumWholeWaveAnnot = [None] * len(self.SingleWaves)
        self.NumFreqWave = [None] * len(self.SingleWaves)
        self.NumAmpWave = [None] * len(self.SingleWaves)

        
        while i < len(Waves):
            self.Waves[i] = ddPlotting.ExtractPlotting(self.SingleWaves[i])
            self.WaveNames[i] = self.SingleWaves[i].WaveNames
            self.NumWholeWave[i] = len(self.Waves[i].Waves)
            self.NumWholeWaveAnnot[i] = len(self.Waves[i].Annot)
            if hasattr(self.Waves[i], 'WavesFreq'):
                self.NumFreqWave[i] = len(self.Waves[i].WavesFreq)
            else: 
                self.NumFreqWave[i] = 0
#                print(self.NumFreqWave[i])
                
            if hasattr(self.Waves[i], 'WavesAmp'):
                self.NumAmpWave[i] = len(self.Waves[i].WavesAmp)
            else:
                self.NumAmpWave[i] = 0
#                print(i)
#                print(self.NumAmpWave[i])

            i += 1
        
        self.NumCells = len(self.Waves)
        if self.NumCells > 10:
            self.NumCells = 10
        
        # Figure: 
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
            self.Figure = plt.figure()
            self.Figure.set_size_inches(11.69, 8.27, forward=True)
        if self.PrintShow == 1:
            self.Figure.set_dpi(300)
        
        self.AX = [None] * (self.NumCells)
        self.gs = gridspec.GridSpec(2, 5)
        self.gs.update(left=0.05, bottom= 0.05, top = 0.4, right=0.95, wspace=0.2)
        self.Markersize = 5

        # Set Grid for Plotting: Row/Colum       
        if self.NumCells >= 1:
            self.AX[0] = plt.subplot(self.gs[0,0])
        if self.NumCells >= 2:
            self.AX[1] = plt.subplot(self.gs[0,1])
        if self.NumCells >= 3:
            self.AX[2] = plt.subplot(self.gs[0,2])
        if self.NumCells >= 4:
            self.AX[3] = plt.subplot(self.gs[0,3])
        if self.NumCells >= 5:
            self.AX[4] = plt.subplot(self.gs[0,4])
        if self.NumCells >= 6:
            self.AX[5] = plt.subplot(self.gs[1,0])
        if self.NumCells >= 7:
            self.AX[6] = plt.subplot(self.gs[1,1])
        if self.NumCells >= 8:
            self.AX[7] = plt.subplot(self.gs[1,2])
        if self.NumCells >= 9:
            self.AX[8] = plt.subplot(self.gs[1,3])
        if self.NumCells >= 10:
            self.AX[9] = plt.subplot(self.gs[1,4])
       
        i = 0
        self.PlottedAnnot = []
        while i < self.NumCells:
            j = 0
            while j < self.NumWholeWave[i]:
                # Set Markersize
                if self.Waves[i].Waves[j].get_marker() == '.' or self.Waves[i].Waves[j].get_marker() == 'o':
                    msset = self.Markersize
                else:
                    msset = 1.0  
                # Printing:
                self.AX[i].plot(self.Waves[i].Waves[j].get_data()[0], self.Waves[i].Waves[j].get_data()[1],self.Waves[i].Waves[j].get_marker(), linestyle = self.Waves[i].Waves[j].get_linestyle(), color = self.Waves[i].Waves[j].get_color(), ms = msset) 
                j +=1   
                
            self.PlottedAnnot.append([i])
            k = 0
            while k < self.NumWholeWaveAnnot[i]:
                self.PlottedAnnot[i].append(k)
                self.PlottedAnnot[i][k] = self.AX[i].annotate(self.Waves[i].Annot[k]._text, self.Waves[i].Annot[k].xy,self.Waves[i].Annot[k].xyann, self.Waves[i].Annot[k].xycoords, self.Waves[i].Annot[k]._textcoords, fontsize = 4,color = self.Waves[i].Annot[k].get_color())
                a = self.PlottedAnnot[i][k].get_position()
                if k == 0:
                    self.PlottedAnnot[i][k].set_position((a[0]+5,a[1]))
                if k == 1:
                    self.PlottedAnnot[i][k].set_position((a[0]+5,a[1]))
                if k == 2:
                    self.PlottedAnnot[i][k].set_position((a[0]-10,a[1]))
                if k == 3:
                    self.PlottedAnnot[i][k].set_position((a[0]/2,a[1]/4))
                k +=1     
            self.AX[i].set_title(self.WaveNames[i]+": %i" % i, fontsize = 6, y=0.9)
            # Setting Axes,...
            self.AX[i].set_xlim(0,800)
            self.AX[i].xaxis.set_ticks(np.arange(0,800,100))
            self.AX[i].set_xlabel('ms',fontsize = 6)
            plt.setp(self.AX[i].get_xticklabels(), fontsize = 6)            
            self.AX[i].set_ylabel('mV',fontsize = 6, rotation = 90)
            plt.setp(self.AX[i].get_yticklabels(),rotation = 'vertical', fontsize = 6)            
            self.AX[i].spines["top"].set_visible(False)
            self.AX[i].spines["right"].set_visible(False)
            i +=1
        
        # Values Plotting:
        self.AXPValuePlots = [None] * 2
        self.gsValuesPlots = gridspec.GridSpec(1,2) 
        self.gsValuesPlots.update(left=0.05, bottom= 0.45, top = 0.65, right=0.95, wspace=0.2)
        self.AXPValuePlots[0] = plt.subplot(self.gsValuesPlots[0,0])
        # Frequency
        i = 0
        while i < self.NumCells:
            j = 0
            while j < self.NumFreqWave[i]:
                # Set Markersize
                if self.Waves[i].WavesFreq[j][0].get_marker() == '.' or self.Waves[i].WavesFreq[j][0].get_marker() == 'o':
                    msset = self.Markersize
                else:
                    msset = 1.0  
                self.AXPValuePlots[0].plot(self.Waves[i].WavesFreq[j][0].get_data()[0], self.Waves[i].WavesFreq[j][0].get_data()[1],self.Waves[i].WavesFreq[j][0].get_marker(), linestyle = self.Waves[i].WavesFreq[j][0].get_linestyle(), color = self.Waves[i].WavesFreq[j][0].get_color(), ms = msset)
                j +=1 
            if hasattr(self.Waves[i], 'WavesFreq'):
                self.AXPValuePlots[0].annotate('%.0f ' % i,xy = (self.Waves[i].WavesFreq[2][0].get_data()[0][-1],self.Waves[i].WavesFreq[2][0].get_data()[1][-1]),xytext=(+5,+5),xycoords='data',textcoords='offset points', fontsize = 6)
            i +=1
        #self.AXPValuePlots[0].set_xlim(0,800)
        #self.AXPValuePlots[0].xaxis.set_ticks(np.arange(0,800,100))
        self.AXPValuePlots[0].set_xlabel('NumAPs',fontsize = 6)
        plt.setp(self.AXPValuePlots[0].get_xticklabels(), fontsize = 6)            
        self.AXPValuePlots[0].set_ylabel('Hz',fontsize = 6, rotation = 90)
        plt.setp(self.AXPValuePlots[0].get_yticklabels(),rotation = 'vertical', fontsize = 6)            
        self.AXPValuePlots[0].spines["top"].set_visible(False)
        self.AXPValuePlots[0].spines["right"].set_visible(False)
        self.AXPValuePlots[0].set_title("Firing Frequency Accomodation", fontsize = 8, y=0.9)
        
        # Amplitudes:
        self.AXPValuePlots[1] = plt.subplot(self.gsValuesPlots[0,1])
        i = 0
        while i < self.NumCells:
            j = 0
            while j < self.NumAmpWave[i]:
                # Set Markersize
                if self.Waves[i].WavesAmp[j][0].get_marker() == '.' or self.Waves[i].WavesAmp[j][0].get_marker() == 'o':
                    msset = self.Markersize
                else:
                    msset = 1.0  
                k = 0
                while k < self.NumCells:
                    if hasattr(self.Waves[k], 'WavesAmp'):
                        self.AXPValuePlots[1].plot(self.Waves[k].WavesAmp[j][0].get_data()[0], self.Waves[k].WavesAmp[j][0].get_data()[1],self.Waves[k].WavesAmp[j][0].get_marker(), linestyle = self.Waves[k].WavesAmp[j][0].get_linestyle(), color = self.Waves[k].WavesAmp[j][0].get_color(), ms = msset)
                    k +=1 
                j +=1 
            if hasattr(self.Waves[i], 'WavesAmp'):
                self.AXPValuePlots[1].annotate('%.0f ' % i,xy = (self.Waves[i].WavesAmp[2][0].get_data()[0][-1],self.Waves[i].WavesAmp[2][0].get_data()[1][-1]),xytext=(+5,+5),xycoords='data',textcoords='offset points', fontsize = 6)
                self.AXPValuePlots[1].annotate('%.0f ' % i,xy = (self.Waves[i].WavesAmp[5][0].get_data()[0][-1],self.Waves[i].WavesAmp[5][0].get_data()[1][-1]),xytext=(+5,+5),xycoords='data',textcoords='offset points', fontsize = 6)
            i +=1
        #self.AXPValuePlots[1].set_xlim(100,600)
        self.AXPValuePlots[1].set_ylim(bottom=0)
#        self.AXPValuePlots[1].xaxis.set_ticks(np.arange(0,800,100))
        self.AXPValuePlots[1].set_xlabel('ms',fontsize = 6)
        plt.setp(self.AXPValuePlots[1].get_xticklabels(), fontsize = 6)            
        self.AXPValuePlots[1].set_ylabel('mV',fontsize = 6, rotation = 90)
        plt.setp(self.AXPValuePlots[1].get_yticklabels(),rotation = 'vertical', fontsize = 6)            
        self.AXPValuePlots[1].spines["top"].set_visible(False)
        self.AXPValuePlots[1].spines["right"].set_visible(False)
        self.AXPValuePlots[1].set_title("Firing Amplitude Adaption", fontsize = 8, y=0.9)
        
        
        # Plot Values:
        #self.AXValues = [None] * 14
        self.gsValues = gridspec.GridSpec(1,15) 
        self.gsValues.update(left=0.05, bottom= 0.7, top = 0.95, right=0.95, wspace=0.8)    
        
        self.List = [None] * 15
        self.ListName = [None] * 15
        self.axValues = [None] * 15
        self.YLimMin =  [None] * 15
        self.YLimMax =  [None] * 15
        
        self.List[0] = self.Values.FiringDuration
        self.ListName[0] = 'FiringDuration [ms]'
        self.List[1] = self.Values.FirstSpikeLatency
        self.ListName[1] = 'FirstSpikeLatency [ms]'
        
        self.List[2] = self.Values.FiringFrequency
        self.ListName[2] = 'Mean FiringFrequency [Hz]'        
        self.List[3] = self.Values.FirstFiringFreq
        self.ListName[3] = 'First FiringFrequency [Hz]'
        self.List[4] = self.Values.FiringFreqFastAcc
        self.ListName[4] = 'Fast Accomodation'
        self.List[5] = self.Values.FiringFreqSlowAcc
        self.ListName[5] = 'SlowAccomodation'
        self.List[6] = self.Values.FiringFreqIndex
        self.ListName[6] = 'FiringFreqIndex'
        
        self.List[7] = self.Values.FirstSpikeAmpBase
        self.ListName[7] = 'First AP-Base Amplitude'
        self.List[8] = self.Values.APBaseFastAdaption
        self.ListName[8] = 'AP-Base Fast Adaption'
        self.List[9] = self.Values.APBaseSlowAdaption
        self.ListName[9] = 'AP-Base Slow Adaption'        
        self.List[10] = self.Values.APBaseAdaptionIndex
        self.ListName[10] = 'AP-Base Adaption Index'
        
        self.List[11] = self.Values.FirstSpikeAmpThres
        self.ListName[11] = 'First AP-Thres Amplitude'
        self.List[12] = self.Values.APThresFastAdaption
        self.ListName[12] = 'AP-Thres Fast Adaption'
        self.List[13] = self.Values.APThresSlowAdaption
        self.ListName[13] = 'AP-Thres Slow Adaption'
        self.List[14] = self.Values.APThresAdaptionIndex
        self.ListName[14] = 'AP-Thres Adaption Index'
        
        i = 0
        while i < 15 :
            if not all(np.isnan(self.List[i].All)):
                # Plotting:
                self.axValues[i] = plt.subplot(self.gsValues[0,i])
                self.axValues[i].plot(np.ones(len(self.List[i].All)),self.List[i].All,'ok') 
                self.axValues[i].errorbar(1,self.List[i].Mean,self.List[i].SD,linestyle='None', marker='o')
                # PointAnnotations:
                j = 0
                while j < len(self.List[i].All):
                    self.axValues[i].annotate('%.0f ' % j,xy = (1,self.List[i].All[j]),xytext=(-8,0),xycoords='data',textcoords='offset points', fontsize = 4)
                    j +=1
                # Mean SD:
                if self.List[i].Mean > -0.1 and np.nanmax(self.List[i].All) < 0.1:
                    self.axValues[i].annotate('%.2f' % self.List[i].Mean,xy = (1,self.List[i].Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 4)
                    self.axValues[i].annotate('+/- %.2f' % self.List[i].SD,xy = (1,self.List[i].Mean),xytext=(5,-4.5),xycoords='data',textcoords='offset points', fontsize = 4)
                else:
                    self.axValues[i].annotate('%.1f' % self.List[i].Mean,xy = (1,self.List[i].Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 4)
                    self.axValues[i].annotate('+/- %.1f' % self.List[i].SD,xy = (1,self.List[i].Mean),xytext=(5,-4.5),xycoords='data',textcoords='offset points', fontsize = 4)
                
                #AxesLimits:
                if min(self.List[i].All) < 0:
                    self.YLimMin[i] = ddhelp.roundup(np.nanmin(self.List[i].All)-10)
                else:
                    self.YLimMin[i] = ddhelp.roundup(np.nanmin(self.List[i].All)-10)
                
                if max(self.List[i].All) < 0:   
                    self.YLimMax[i] = ddhelp.roundup(np.nanmax(self.List[i].All))
                else:
                    self.YLimMax[i] = ddhelp.roundup(np.nanmax(self.List[i].All))
                    
                if np.nanmin(self.List[i].All) > 0 and np.nanmax(self.List[i].All) < 10:
                    self.YLimMin[i] = np.round(np.nanmin(self.List[i].All)-0.1,decimals = 1)    
                    self.YLimMax[i] = np.round(np.nanmax(self.List[i].All)+0.1,decimals = 1)
                    
                if np.nanmin(self.List[i].All) > -10 and np.nanmax(self.List[i].All) < 5:
                    self.YLimMin[i] = np.round(np.nanmin(self.List[i].All)-0.1,decimals = 1)    
                    self.YLimMax[i] = np.round(np.nanmax(self.List[i].All)+0.1,decimals = 1)
                
                if np.nanmin(self.List[i].All) > -0.1 and np.nanmax(self.List[i].All) < 0.1:
                    self.YLimMin[i] = np.round(np.nanmin(self.List[i].All)-0.01,decimals = 2)    
                    self.YLimMax[i] = np.round(np.nanmax(self.List[i].All)+0.01,decimals = 2)
                
                
#                if self.List[i].SD <= 0.0001:
#                    self.YLimMin[i] = self.List[i].Mean-1
#                    self.YLimMax[i] = self.List[i].Mean+1
                    
         
                self.axValues[i].set_ylim ([self.YLimMin[i],self.YLimMax[i]])
                
                #Axes Apparence:
                self.axValues[i].set_ylabel(self.ListName[i],fontsize = 6, rotation = 90, labelpad=0.5)
                plt.setp(self.axValues[i].get_yticklabels(),rotation = 'vertical', fontsize = 6)
                self.axValues[i].set_xlim ([0,3])
                self.axValues[i].set_xticklabels([])
                self.axValues[i].tick_params(axis='x', which='both',bottom='off')      
                self.axValues[i].spines["top"].set_visible(False)
                self.axValues[i].spines["right"].set_visible(False)
            i +=1
            
        # Title:
        self.OVTitle = 'Firing Properties of ' + self.CellName +':'
        self.axValues[0].set_title(self.OVTitle,fontsize = 14, y = 1.05, fontweight='bold',loc='left')
        
        #Saving:
        if self.PrintShow == 1:
            self.CellName3 = '_FiringProperties'  
            self.SavingName = self.CellName+self.CellName3
            ddPlotting.save(self.SavingName, ext="png", close=True, verbose=True)
#            ddPlotting.save(self.SavingName, ext="svg", close=True, verbose=True)
            plt.close('All')
            plt.ion()
        
        
       
        
        