#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Subthreshold Properties """

''''
Created on Wed May  3 15:27:59 2017     @author: DennisDa
'''

''' Importing Scripts '''
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import ddhelp
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import ddPlotting
import pandas as pd
import SuprathresholdProperties


''' Importing Stuff for Testing '''
#import os
#import DDImport 
#cwd = os.getcwd()
#filesdir = os.path.abspath('..')
#os.chdir(filesdir)
#Waves,Times,SampFreq1,RecTime = DDImport.Folder.ImpC(filesdir)
#
## Test Maximum Response
#WToTake=5
#SampFreq = SampFreq1[WToTake]
#
#Wave = Waves[WToTake]#[int(100*SampFreq):int(600*SampFreq)]*-1 # 8
#Time = Times[WToTake]#[int(100*SampFreq):int(600*SampFreq)] # 8
#
#Stimulus = np.zeros(len(Wave))
#Stimulus[int(100*SampFreq):int(600*SampFreq)] = 1
#
#WaveEnd = Waves[WToTake][int(600*SampFreq)-1:-1]
#TimeEnd = Times[WToTake][int(600*SampFreq)-1:-1]



''' START OF THE SCRIPT: '''

''' Different Objects for Analysis ''' 
class PeakDerivationAssist:
    def __init__(self,Time,Wave, SampFreq, CalcWindow,ZeroBorders):
        self.Time = Time
        self.Wave = Wave
        self.CalcWindow = CalcWindow
        self.SampFreq = SampFreq
        self.WaveFilt = np.zeros(len(Wave))
        self.WaveDiff = np.zeros(len(Wave)-1)
        self.ZeroBoarders = ZeroBorders
        
        self.WaveFilt, self.WaveDiff = ddhelp.Deri.SlowPeak(self.Time, self.Wave, self.SampFreq)
        self.WaveDiffSearch = self.WaveDiff[CalcWindow[0]:CalcWindow[1]]
        self.MaxDiffSearch = np.argmax(self.WaveDiffSearch)
        self.WaveFindMax = self.WaveDiffSearch[self.MaxDiffSearch:-1]
        self.DiffZer0 = np.where((self.WaveDiffSearch <= self.ZeroBoarders) & (self.WaveDiffSearch >= -1* self.ZeroBoarders))
        self.PotMaxPre = np.asarray(self.DiffZer0)
        self.PotMax = self.PotMaxPre + self.MaxDiffSearch
        self.PotMax = np.insert(self.PotMax,0,1)


class SlowPeakDerivation :
    def __init__(self,Time,Wave,SampFreq,CalcWindow1 = None, CalcWindow2 = None):
        self.Time = Time
        self.Wave = Wave
        self.SampFreq = SampFreq
        self.CalcWindow1 = CalcWindow1
        self.CalcWindow2 = CalcWindow2
        self.PotMaxLength = 10*self.SampFreq
        if self.CalcWindow1 == None:
            self.CalcWindow1 = [0,int(len(self.Wave))]
            self.CalcWindow2 = [0,int(len(self.Wave))]
       
        ''' Derivation, find were zero is hit '''
        self.A = PeakDerivationAssist(self.Time,self.Wave,self.SampFreq,self.CalcWindow1,10**-5)
        self.t='0'
        self.FiltWave = self.A.WaveFilt
        
        if np.size(self.A.PotMax)<=2*self.SampFreq or not hasattr(self.A, 'PotMax'):
            self.t = '1'
            self.A = PeakDerivationAssist(self.Time,self.Wave,self.SampFreq,self.CalcWindow2,10**-5)
            self.FiltWave = self.A.WaveFilt
            if np.size(self.A.PotMax)<=2*self.SampFreq or not hasattr(self.A, 'PotMax'):
                self.t = '2'
                self.A = PeakDerivationAssist(self.Time,self.Wave,self.SampFreq,self.CalcWindow2,10**-4)
                self.FiltWave = self.A.WaveFilt
                
                if np.size(self.A.PotMax)<=2*self.SampFreq or not hasattr(self.A, 'PotMax'):
                    self.t = '3'
                    self.A = PeakDerivationAssist(self.Time,self.Wave,self.SampFreq,self.CalcWindow2,10**-3)
                    self.FiltWave = self.A.WaveFilt
                    
                    if np.size(self.A.PotMax)<=2*self.SampFreq  or not hasattr(self.A, 'PotMax'):
                        self.t='4'
                        PotMax1 = np.argmin(self.A.WaveDiffSearch)
                        self.A.PotMax = np.zeros(3)
                        self.A.PotMax[0] = int(PotMax1-1)
                        self.A.PotMax[1] = int(PotMax1)
                        self.A.PotMax[2] = int(PotMax1+1)   
                        self.FiltWave = np.zeros(len(self.Wave))
                    else:
                        pass
                else:
                    pass
                
            else:
                pass
            
        else:
            pass
        ''' Return to VStable '''
        self.PotMaxDiff = np.diff(self.A.PotMax)
    
        ''' Calculations: '''
        # A: Time Point:
        if np.size(self.A.PotMax) >=2:
            # Old:
#            self.MaxTAllmost = self.PotMaxDiff[np.where(self.PotMaxDiff > 1)] # [i for i in self.PotMaxDiff if i >= 1]
#            self.MaxT = self.MaxTAllmost[0]/self.SampFreq

            # New1:
#            self.MaxTAllmost = self.A.PotMax[np.where(self.PotMaxDiff > 1)] # [i for i in self.PotMaxDiff if i >= 1]
#            self.MaxT = self.MaxTAllmost[0]/self.SampFreq
            

            # New2:
            self.MaxTAllmost = self.PotMaxDiff[np.where(self.PotMaxDiff > 1)] # Check hier!!!
            self.ConVentMax = np.argmax(self.Wave[self.CalcWindow1[0]:self.CalcWindow1[-1]])
            if self.ConVentMax > (len(self.Wave[self.CalcWindow1[0]:self.CalcWindow1[-1]]))*2/3:
                self.ConVentMax = np.argmax(self.Wave[self.CalcWindow2[0]:self.CalcWindow2[-1]])
#            self.ConVentMax = np.argmax(self.Wave)
            self.FindToTakeInMaxTAllmost = np.absolute(self.MaxTAllmost - self.ConVentMax) 
            self.MaxT = self.MaxTAllmost[np.argmin(self.FindToTakeInMaxTAllmost)]/self.SampFreq

            # Continous:
            self.MaxforBoarders = int(self.MaxT*self.SampFreq)
            
            # B: Voltage:
            # Fit over length of Potential Maxima
            if np.size(self.A.PotMax) > 5*self.SampFreq:
                self.Dist = int(5*self.SampFreq)
            else:
                self.Dist = np.size(self.A.PotMax)
                
            self.MaxVmBoarders1 = self.MaxforBoarders-self.Dist
            if self.MaxVmBoarders1 < 0:
                self.MaxVmBoarders1 = 0
                
            self.MaxVmBoarders2 = self.MaxforBoarders+self.Dist
            
            if self.MaxVmBoarders2 > int(self.CalcWindow2[1]):
                self.MaxVmBoarders2 = int(self.CalcWindow2[1])-1
            
            if (self.MaxVmBoarders2-self.MaxVmBoarders1)/self.SampFreq > 10:
                self.MaxVmBoarders2 = int(self.MaxVmBoarders1+10*self.SampFreq)
                
            self.MaxVm = np.mean(self.Wave[self.MaxVmBoarders1:self.MaxVmBoarders2]) 
            
        else:
            self.MaxT = self.A.PotMax[1]/self.SampFreq
            self.MaxVm = np.mean(self.Wave[self.A.PotMax[0]:self.A.PotMax[2]])
#        print(self.t)
#        print(self.ConVentMax)
##        print(np.argmin(self.FindToTakeInMaxTAllmost))
##        plt.figure()
###        plt.plot(self.Time,self.FiltWave)
##        plt.plot(self.Time[0:-1],np.diff(self.FiltWave))
###        plt.plot(self.Time[0]+self.MaxT,self.MaxVm,'o')
##        a

class SlowPeakFit:
    def __init__(self,Time,Wave,SampFreq,StartPointFit,LowerBounds=None,UpperBounds=None):
        self.Time = Time
        self.Wave = Wave
        self.SampFreq = SampFreq
        self.StartPointFit = StartPointFit
        self.BaselinePositions = [int(len(Wave)-(10*self.SampFreq)),int(len(Wave))] 
        self.LowerBounds = LowerBounds
        self.UpperBounds = UpperBounds
            
#        ''' Filter before Fitting'''
#        self.WaveFilt = ddhelp.Filters.MovingAverage(self.Wave, window_len=int(50 * self.SampFreq), window='hanning')
#        self.WaveFilt = self.WaveFilt+np.random.normal(0,0.5,len(self.WaveFilt))

        ''' PSP Fit: '''
        self.PSPFit = ddhelp.FitPSP(self.Time, self.Wave, self.BaselinePositions,self.StartPointFit,self.LowerBounds,self.UpperBounds)

        ''' Find Peak in fit Curve: '''
        self.MaxT = np.argmax(self.PSPFit.FitCurve)/self.SampFreq
        self.MaxVm = np.max(self.PSPFit.FitCurve)
        self.FitAmplitude = self.MaxVm- np.mean(self.Wave[self.BaselinePositions[0]:self.BaselinePositions[1]])
        self.r = self.PSPFit.r

     
class MaxResponse:
    def __init__(self,Time,Wave,SampFreq):
        self.Time = Time
        self.Wave = Wave
        self.SampFreq = SampFreq
        self.StartPointFit = [1.,18.,80.,50]
        
        self.CalcWindow1 = [0,int(len(self.Wave)/3)]
        self.CalcWindow2 = [0,int((len(self.Wave)/3)*2)]
        
        self.WithBounds = 0
        
        #A: PSPFit: 
        self.PSPFit = SlowPeakFit(self.Time,self.Wave,self.SampFreq,self.StartPointFit)
        self.PSPExtrem = 0
        if self.PSPFit.r <= 0.9 or self.PSPFit.MaxT <= 1.:# or self.PSPFit.MaxT*self.SampFreq >= int((len(self.Wave)/3)*2):
            #print('1')
            self.TimeFit = self.Time[0:int((len(self.Wave)))]
            self.WaveFit = self.Wave[0:int((len(self.Wave)))]
            self.StartPointFit2 = [1.,13.,30.,self.TimeFit[np.argmax(self.WaveFit)]]
            self.PSPFit = SlowPeakFit(self.TimeFit,self.WaveFit,self.SampFreq,self.StartPointFit2) 
            self.PSPExtrem = 1

        if self.PSPFit.r <= 0.9 or self.PSPFit.MaxT <= 1.:#. or self.PSPFit.MaxT*self.SampFreq >= int((len(self.Wave)/3)*2):
            #print('2')
            self.TimeFit = self.Time[0:int((len(self.Wave)/3*2))]
            self.WaveFit = self.Wave[0:int((len(self.Wave)/3*2))]
            self.StartPointFit2 = [1.,18.,80.,np.argmax(self.Wave)/self.SampFreq]
            self.PSPFit = SlowPeakFit(self.TimeFit,self.WaveFit,self.SampFreq,self.StartPointFit2) 
            self.PSPExtrem = 2
        
        if self.PSPFit.r <= 0.9  or self.PSPFit.MaxT <= 1.:# or self.PSPFit.MaxT*self.SampFreq >= int((len(self.Wave)/3)*2):
            self.TimeFit = self.Time[0:int((len(self.Wave)/3))]
            self.WaveFit = self.Wave[0:int((len(self.Wave)/3))]
            self.StartPointFit3 = [1.,18.,40.,self.TimeFit[np.argmax(self.WaveFit)]] #[1.,8.,30.,50.]
            self.PSPFit = SlowPeakFit(self.TimeFit,self.WaveFit,self.SampFreq,self.StartPointFit3)   
            self.PSPExtrem = 3
            
        if self.PSPFit.r <= 0.9 or self.PSPFit.MaxT <= 1.:#. or self.PSPFit.MaxT*self.SampFreq >= int((len(self.Wave)/3)*2):
            self.TimeFit = self.Time[0:int((len(self.Wave)/3))]
            self.WaveFit = self.Wave[0:int((len(self.Wave)/3))]
            self.StartPointFit2 = [1.,18.,10.,np.argmax(self.Wave)/self.SampFreq]
            self.PSPFit = SlowPeakFit(self.TimeFit,self.WaveFit,self.SampFreq,self.StartPointFit2)
            self.PSPExtrem = 4
        
        if self.PSPFit.r <= 0.9  or self.PSPFit.MaxT <= 1.:# or self.PSPFit.MaxT*self.SampFreq >= int((len(self.Wave)/3)*2):
            self.LowerBounds = [0.,0.,0.,0.]
            self.UpperBounds = [np.inf,80,100,int((len(self.Wave)/3))]
            self.TimeFit = self.Time[0:int((len(self.Wave)/3))]
            self.WaveFit = self.Wave[0:int((len(self.Wave)/3))]
            self.StartPointFit2 = [1.,13.,30.,self.TimeFit[np.argmax(self.WaveFit)]]
            self.PSPFit = SlowPeakFit(self.TimeFit,self.WaveFit,self.SampFreq,self.StartPointFit2,self.LowerBounds,self.UpperBounds)  
            self.PSPExtrem = 5
            self.WithBounds = 1
#        print(self.PSPExtrem)
#        print(self.PSPFit.r)
#        print(self.PSPFit.PSPFit.FitParams)
#        print(all(self.PSPFit.PSPFit.FitParams>0))
        #B: Derivation:
        self.Devi = SlowPeakDerivation(self.Time,self.Wave,self.SampFreq,self.CalcWindow1, self.CalcWindow2)
        
        # Decision what to take:
        if self.PSPFit.r >= 0.9 and self.PSPFit.MaxT >= 1. and self.PSPFit.MaxT*self.SampFreq <= int((len(self.Wave)/3)*2) and self.PSPFit.FitAmplitude > 0.1 and all(self.PSPFit.PSPFit.FitParams>0):
            self.MaxResponseT = self.PSPFit.MaxT
            self.MaxResponseVm = self.PSPFit.MaxVm
            self.MaxFitCurve = self.PSPFit.PSPFit.FitCurve
            self.MaxFitr = self.PSPFit.r
            self.MaxFitTauOff = self.PSPFit.PSPFit.FitParams[2]
            self.Method = 'PSP Fit'
        else:
            self.MaxResponseT = self.Devi.MaxT
            self.MaxResponseVm = self.Devi.MaxVm
            #self.MaxFitCurve = np.zeros(len(self.Wave))
            self.MaxFitCurve = self.Devi.FiltWave
            self.MaxFitr = 0
            self.MaxFitTauOff = 0
            self.Method = 'Derivation'
            
        if self.WithBounds == 1 and self.PSPFit.FitAmplitude < 1:
            self.MaxResponseT = self.Devi.MaxT
            self.MaxResponseVm = self.Devi.MaxVm
            #self.MaxFitCurve = np.zeros(len(self.Wave))
            self.MaxFitCurve = self.Devi.FiltWave
            self.MaxFitr = 0
            self.MaxFitTauOff = 0
            self.Method = 'Derivation'
        
class ReboundResponse:
    def __init__(self,Time,Wave,SampFreq):
        self.Time = Time
        self.Wave = Wave
        self.SampFreq = SampFreq
        self.StartPointFit = [1.,18.,40.,self.Time[np.argmax(self.Wave)]]
        self.StartPointFit2 = [1.,25.,40.,self.Time[np.argmax(self.Wave)]]
        self.CalcWindow1 = [0,int(len(self.Wave))]
        self.CalcWindow2 = [0,int(len(self.Wave))]
        
        #A: PSPFit:
        self.PSPFit = SlowPeakFit(self.Time,self.Wave,self.SampFreq,self.StartPointFit)  
        if self.PSPFit.r <= 0.9:
             self.PSPFit = SlowPeakFit(self.Time,self.Wave,self.SampFreq,self.StartPointFit2)
             
        # Try with Bounds:
        if self.PSPFit.r <= 0.9:
            self.LowerBounds = [1,.5,5,(self.Time[np.argmax(self.Wave)]-25)]
            self.UpperBounds = [15,30,100,(self.Time[np.argmax(self.Wave)]+25)]
            self.PSPFit = SlowPeakFit(self.Time,self.Wave,self.SampFreq,self.StartPointFit2,self.LowerBounds,self.UpperBounds)
            self.PSPExtrem = 1
            
        #B: Derivation:
        self.Devi = SlowPeakDerivation(self.Time,self.Wave,self.SampFreq,self.CalcWindow1, self.CalcWindow2)        
        
        # Decision what to take:
        if self.PSPFit.r >= 0.9 and self.PSPFit.MaxT >= 1. and self.PSPFit.MaxT*self.SampFreq <= int(len(self.Wave)*0.9) and self.PSPFit.FitAmplitude > 1 and self.PSPFit.PSPFit.FitParams[1] <= 25:
            self.MaxResponseT = self.PSPFit.MaxT
            self.MaxResponseVm = self.PSPFit.MaxVm
            self.MaxFitCurve = self.PSPFit.PSPFit.FitCurve
            self.MaxFitr = self.PSPFit.r
            self.MaxFitTauOff = self.PSPFit.PSPFit.FitParams[2]
            self.Method = 'PSP Fit'
        else:
            self.MaxResponseT = self.Devi.MaxT
            self.MaxResponseVm = self.Devi.MaxVm
            self.MaxFitCurve = np.zeros(len(self.Wave))
            self.MaxFitr = 0
            self.MaxFitTauOff = 0
            self.Method = 'Derivation'
       
class Tau:
    ''' Take Two Times: First: Maximum Response, then, Stable Response! '''
    # Add Max Response? Response to Max or VStable with difference! 
    def __init__(self,Time,Wave,SampFreq,TauTo,StimOnset):
        # Needs: Wave from 0 to MaxResponse + 10 ms!!!
        # Tau To: Max Response, or VStable
        self.Time = Time
        self.Wave = Wave
        self.SampFreq = SampFreq
        self.TauTo = TauTo
        self.StimOnset = StimOnset
        self.BaseVm = np.mean(self.Wave[0:int(99*SampFreq)])
        #self.BaseVm = self.Wave[0]
        self.TauPotential = ((1-(1/np.exp(1)))*(self.TauTo-self.BaseVm))+self.BaseVm
        # Filtered Trace Zeroed to 63% Percent: 
        self.WaveFilt = ddhelp.Filters.MovingAverage(self.Wave, window_len=int(5 * self.SampFreq), window='hanning')
        self.WaveZeroed = self.WaveFilt - self.TauPotential 
        
        # A: Fit Expotential function to it
        self.TimeExp = self.Time[int(self.StimOnset*self.SampFreq):-1]
        self.WaveExp = self.Wave[int(self.StimOnset*self.SampFreq):-1]-self.TauPotential 
        self.BaselinePositions = [-int(10*SampFreq),-1]
        self.StartPointFit = [0.,0.,0.]
        self.MonoExFit = ddhelp.FitMonoEx(self.TimeExp,self.WaveExp,self.BaselinePositions,self.StartPointFit)
        if not self.MonoExFit.FitParams[1] == 0:
            self.TauFit = -1/self.MonoExFit.FitParams[1] 

        self.TauVmFit = self.MonoExFit.FitCurve+self.TauPotential
        self.TauTimeFit = self.TimeExp
        
        # B: Find Zero:
        self.WaveZeroedSearch = self.WaveZeroed[int(self.StimOnset*self.SampFreq):-1]
        self.TauMaxAlmost= np.where(np.absolute(self.WaveZeroedSearch)<=(np.std(self.WaveFilt[0:int(99*self.SampFreq)]/8)));    
        if len(self.TauMaxAlmost) == 0:   
            self.TauMaxAlmost= np.where(np.absolute(self.WaveZeroedSearch)<=(np.std(self.WaveFilt[0:int(99*self.SampFreq)]/7)));    
        if len(self.TauMaxAlmost) == 0:   
            self.TauMaxAlmost= np.where(np.absolute(self.WaveZeroedSearch)<=(np.std(self.WaveFilt[0:int(99*self.SampFreq)]/6)));    
        if len(self.TauMaxAlmost) == 0:
            self.TauMaxAlmost= np.where(np.absolute(self.WaveZeroedSearch)<=(np.std(self.WaveFilt[0:int(99*self.SampFreq)]/5)));    
        if len(self.TauMaxAlmost)== 0:
            self.TauMaxAlmost= np.where(np.absolute(self.WaveZeroedSearch)<=(np.std(self.WaveFilt[0:int(99*self.SampFreq)]/4)));    
        if len(self.TauMaxAlmost)== 0:
            self.TauMaxAlmost= np.where(np.absolute(self.WaveZeroedSearch)<=(np.std(self.WaveFilt[0:int(99*self.SampFreq)]/3)));    
        if len(self.TauMaxAlmost)== 0:
            self.TauMaxAlmost= np.where(np.absolute(self.WaveZeroedSearch)<=(np.std(self.WaveFilt[0:int(99*self.SampFreq)]/2)));    
        if len(self.TauMaxAlmost)== 0:
            self.TauMaxAlmost= np.where(np.absolute(self.WaveZeroedSearch)<=(np.std(self.WaveFilt[0:int(99*self.SampFreq)]/1)));    
        
        self.TauVmSearch = self.TauPotential
        self.TauTimeSearch = (np.median(self.TauMaxAlmost)/self.SampFreq) + self.StimOnset
        self.TauSearch = self.TauTimeSearch - self.StimOnset
        
        # What to Take:
        self.FitSearchThresFit = np.absolute(np.sum(self.MonoExFit.FitCurve) - np.sum(self.WaveExp))
        if self.MonoExFit.r >= 0.9 and self.FitSearchThresFit < 50:
            self.TauTime = self.TauTimeFit[int(len(self.TauTimeFit)*0.63)]
            self.TauVm = self.TauVmFit[int(len(self.TauTimeFit)*0.63)]
            self.Tau = self.TauFit
            self.Method = 'Exp Fit'
        else:
            self.TauTime = self.TauTimeSearch
            self.TauVm = self.TauVmSearch
            self.Tau = self.TauSearch
            self.Method = 'Search'    
        
class SagCalc:
    def __init__(self,Time,Wave,SampFreq,VBaseline,MaxAmp=None,VStableAmp=None,Fitr=0,FitCurve=0,FitTauOff=0,MaxResponseTP=None):
        ''' Time and Wave to Calc Maximum Response/Rebound + VStable  '''
        # Wave has to rise!: from StimOnset to StimOffset #
        
        self.Time = Time
        self.Time = np.asarray(self.Time)
        self.SampFreq = SampFreq
        self.Wave = Wave
        self.Wave = np.asarray(self.Wave)
        self.VBaseline = VBaseline
        self.MaxAmp = MaxAmp
        self.VStableAmp = VStableAmp
        self.Fitr = Fitr
        self.FitCurve = FitCurve
        self.FitTauOff = FitTauOff
        self.MaxResponseTP = int(MaxResponseTP *self.SampFreq)
        
        # Get Values if None: 
        if self.MaxAmp is None:
            self.MaxValues = MaxResponse(self.Time,self.Wave,self.SampFreq)
            self.MaxAmp = self.MaxValues.MaxResponseVm*-1 - self.VBaseline
            self.Fitr = self.MaxValues.MaxFitr
            self.FitCurve = self.MaxValues.MaxFitCurve*-1
            self.FitTauOff = self.MaxValues.MaxFitTauOff
        if self.MaxResponseTP is None:
            self.MaxValues = MaxResponse(self.Time,self.Wave,self.SampFreq)
            self.MaxResponseTP = self.MaxValues.MaxResponseT + int(self.Time[0]*self.SampFreq)
        if self.VStableAmp is None: 
            self.VStableAmp = np.mean(self.Wave[int(-100*self.SampFreq):-1])*-1 - self.VBaseline
            
        # Calc Index:
        self.Index = ((self.MaxAmp-self.VStableAmp)/self.MaxAmp)*100
        
        # New Calculation of Sag Area: From Max Response to End of Stimulus
#        self.Change = 1
        self.TimeCalcArea = self.Time[self.MaxResponseTP:-1]
        self.WaveCalcAreaPre = self.Wave[self.MaxResponseTP:-1]
        self.WaveCalcArea = self.WaveCalcAreaPre + self.VBaseline + self.MaxAmp
        if np.mean(self.Wave[int(-100*self.SampFreq):-1]) <= 0 :
             self.WaveCalcArea = self.WaveCalcAreaPre - self.VBaseline - self.MaxAmp
#             self.Change = -1
        
        self.Area = -1 * sp.integrate.simps(self.WaveCalcArea,self.TimeCalcArea)
#        self.Area = self.Area/(len(self.TimeCalcArea)/self.SampFreq)
#        print(self.Area)
        self.TauOff = None
        self.TimeFit = self.TimeCalcArea
        self.AreaThresholdVm = np.zeros(len(self.TimeCalcArea))       
        self.AreaThresholdVm[:] = self.VBaseline+self.MaxAmp
        
        # Artificial TauOff Measurment:
        if np.sum(self.FitCurve) !=0 and self.Fitr >= 0.8 and self.Index > 5:
            self.TauOff = self.FitTauOff
        
        
#        # Calculation only if self.FitCurve Exists:
#        if np.sum(self.FitCurve) !=0 and self.Fitr >= 0.8 and self.Index > 5:
#            self.FitCurveZeroed = self.FitCurve - np.mean(self.FitCurve[-100:-1])    
#            if np.sum(self.FitCurveZeroed) < 0:
#                self.FitCurveZeroed = self.FitCurveZeroed *-1
#                self.Change = -1
#            else:
#                self.Change  = 1
#            
#            if self.Fitr >= 0.8 and self.Index > 5: # and np.sum(self.FitCurve) < 0:
#                self.VMax = self.MaxAmp - self.VStableAmp #- np.mean(self.FitCurve[-100:-1])
#                self.FitCurveIndex = np.where(self.FitCurveZeroed >= (np.absolute(self.VMax)*0.10))    
##                print(self.FitCurveIndex)
#                if np.size(self.FitCurveIndex) > 10:
#                    self.TimeFit = self.Time[self.FitCurveIndex]
##                    print(self.TimeFit)
#                    self.FitCurveZeroedforInt = self.FitCurve[self.FitCurveIndex]-np.mean(self.FitCurve[-100:-1])  
#                    self.Area = sp.integrate.simps(self.FitCurveZeroedforInt,self.TimeFit)
#                    self.AreaThresholdVm = np.zeros(len(self.TimeFit))
#                    self.AreaThresholdVm[:] = (self.Change*((np.absolute(self.VMax)*0.10)) + np.mean(self.FitCurve[-100:-1]))
#                    self.WaveIn = ddhelp.Filters.MovingAverage(self.Wave[0:self.MaxResponseTP],window_len=int(10*SampFreq),window='hanning')
#                    if len(self.WaveIn)>0:
#                        self.WaveIn = np.absolute(self.WaveIn-self.Change*self.AreaThresholdVm[0])
#                        self.TP = np.argmin(self.WaveIn)
#                        self.MinDiff = np.min(self.WaveIn)
#                        self.TimeDiff = np.absolute(self.TimeFit[0] - self.Time[int(self.TP)])
#                        self.TimeFit = self.TimeFit+self.TimeDiff
#                    self.TauOff = self.FitTauOff    
#                else:
#                    self.Area = 0
#                    self.TauOff = None
#                    self.TimeFit = None
#                    self.AreaThresholdVm = None
#    
#        # Calculate Nummerically:            
#        elif self.Index > 5:
##            print('Used')
#            self.VMax = self.MaxAmp - self.VStableAmp #- np.mean(self.FitCurve[-100:-1])
#            self.CurveFilt = ddhelp.Filters.MovingAverage(self.Wave,window_len=int(50*SampFreq),window='hanning')
#            self.CurveFilt = self.CurveFilt - np.mean(self.CurveFilt[-int(100*SampFreq):-1])
#            if self.VMax > 0:
#                self.Change = 1
#            else:
#                self.Change  = -1
#            self.CurveFilt1 = self.CurveFilt - 0.1*np.absolute(self.VMax) 
#            i = 0
#            while i < len(self.CurveFilt1):
#                if self.CurveFilt1[i]< 0:
#                    self.CurveFilt1[i] = 0
#                if i > len(self.CurveFilt1)*3/5 and self.CurveFilt1[i-1]==0:
#                    self.CurveFilt1[i] = 0
#                i += 1
#            self.CurveStartWave = self.CurveFilt1[0:self.MaxResponseTP]
#            self.CurveStartTime = self.Time[0:self.MaxResponseTP]
#            self.StartPre1, = np.where(np.diff(self.CurveStartWave)>0)            
#            self.StartPre2, = np.where(np.diff(self.StartPre1[1:-1])>1)
#            self.CurveStopWave = self.CurveFilt1[self.MaxResponseTP:-1]
#            self.CurveStopTime = self.Time[self.MaxResponseTP:-1]
#            self.StopPre1, =  np.where(np.diff(self.CurveStopWave)<0)
#            self.StopPre2, = np.where(np.diff(self.StopPre1[1:-1])>0)
##            print(self.StopPre2,40*self.SampFreq,self.StopPre2[-1]>40*self.SampFreq)
#            if len(self.StartPre1) > 0 and len(self.StopPre1) > 0:
#                if len(self.StartPre2) > 0:
#                    self.StartP = self.StartPre1[self.StartPre2[0]-1]
#                else:
#                    self.StartP = self.StartPre1[0]
#                if len(self.StopPre2) > 0 and self.StopPre2[-1]>40*self.SampFreq:
##                    print('DONE')
#                    self.StopP = self.StopPre1[self.StopPre2[-1]-1]
#                else:
#                    self.StopP = self.StopPre1[-1]
##                self.StopP = self.StopPre1[-1]
#                self.TimeArea = self.Time[self.StartP:(self.StopP+self.MaxResponseTP)]
#                self.WaveArea = self.CurveFilt1[self.StartP:(self.StopP+self.MaxResponseTP)]
#                self.Area = self.Change*sp.integrate.simps(self.WaveArea,self.TimeArea)
#                self.TimeFit = self.TimeArea
#                self.AreaThresholdVm = np.zeros(len(self.TimeArea))
#                self.AreaThresholdVm[:] = self.Change*(((np.absolute(self.VMax)*0.10)) + np.mean(self.Wave[-int(100*SampFreq):-1]))
#                self.TauOff = None
#            else:
#                self.Area = 0
#                self.TauOff = None
#                self.TimeFit = None
#                self.AreaThresholdVm = None  
#        else:
#            self.Area = 0
#            self.TauOff = None
#            self.TimeFit = None
#            self.AreaThresholdVm = None

#        plt.figure()
#        plt.ion()
#        plt.plot(self.Time,self.Wave)
#        plt.plot(self.TimeCalcArea,self.WaveCalcArea)
###        plt.plot(self.Time,self.CurveFilt1)
###        plt.plot(self.CurveStopTime,self.CurveStopWave)
####        plt.plot(self.CurveStopTime[0:-1],np.diff(self.CurveStopWave))
####        plt.plot(self.CurveStopTime[0:-2],np.diff(np.diff(self.CurveStopWave)))
###        plt.plot(self.TimeFit,self.AreaThresholdVm)
#        a 
        
class ReboundCalc:
    def __init__(self,Time,Wave,SampFreq,VBaseline,MaxAmp,ReboundAmp=None,Fitr=0,FitCurve=0,FitTauOff=0,ReboundT=None,VStableAmp=None):
        ''' Time and Wave to Calc Maximum Response/Rebound + VStable  '''
        # Wave has to rise!: from StimOffset to end #
        
        self.Time = Time
        self.Time = np.asarray(self.Time)
        self.SampFreq = SampFreq
        self.Wave = Wave
        self.Wave = np.asarray(self.Wave)
        self.VBaseline = VBaseline
        self.MaxAmp = MaxAmp*-1
        self.ReboundAmp = ReboundAmp
        self.Fitr = Fitr
        self.FitCurve = FitCurve
        self.FitTauOff = FitTauOff
        self.StableAmp = VStableAmp
        
        # Get Values if None: 
                  
        if self.ReboundAmp is None: 
            self.ReboundValues = ReboundResponse(self.Time,self.Wave,self.SampFreq)
            self.ReboundAmp = self.ReboundValues.MaxResponseVm- self.VBaseline
            self.Fitr = self.ReboundValues.MaxFitr
            self.FitCurve = self.ReboundValues.MaxFitCurve
            self.FitTauOff = self.ReboundValues.MaxFitTauOff
        if ReboundT is None: 
            self.ReboundValues = ReboundResponse(self.Time,self.Wave,self.SampFreq)
            self.ReboundTP = int(self.ReboundValues.MaxResponseT*self.SampFreq)
        else:
            self.ReboundTP = int(ReboundT*self.SampFreq)
            
        # Calculation:
        self.Index = ((self.ReboundAmp)/self.MaxAmp)*100
        if np.sum(self.FitCurve) !=0 and self.Fitr >= 0.8 and np.absolute(self.Index) > 5:
            self.FitCurveZeroed = self.FitCurve - np.mean(self.FitCurve[-100:-1])    
        
            if np.sum(self.FitCurveZeroed) < 0:
                self.FitCurveZeroed = self.FitCurveZeroed *-1
                self.Change = -1
            else:
                self.Change  = 1
            
            if self.Fitr >= 0.8 and np.absolute(self.Index) > 5: # and np.sum(self.FitCurve) < 0:
                self.VRebound = self.ReboundAmp
                self.FitCurveIndex = np.where(self.FitCurveZeroed >= (np.absolute(self.VRebound)*0.10))    

                self.TimeFit = self.Time[self.FitCurveIndex]
                self.FitCurveZeroedforInt = self.FitCurve[self.FitCurveIndex]-np.mean(self.FitCurve[-100:-1])  
                try: 
                    self.Area = sp.integrate.simps(self.FitCurveZeroedforInt,self.TimeFit)
                
                    self.AreaThresholdVm = np.zeros(len(self.TimeFit))
            
                    self.AreaThresholdVm[:] = (self.Change*((np.absolute(self.VRebound)*0.10)) + np.mean(self.FitCurve[-100:-1]))
           
                    self.TauOff = self.FitTauOff
            
#                except:
#                    self.Area = 0
#                    self.TauOff = None   
#                    self.TimeFit = None
#                    self.AreaThresholdVm = None
                    
                except (RuntimeError,RuntimeWarning):
                    pass
    
        # Calculate Nummerically:           
        elif np.absolute(self.Index) > 5:
#            print('Index',self.Index)
#            print('MaxAmp',self.MaxAmp)
#            print('Baseline',self.VBaseline)
#            print('ReboundAmp',self.ReboundAmp)
            self.CurveFilt = ddhelp.Filters.MovingAverage(self.Wave,window_len=int(50*SampFreq),window='hanning')
            self.CurveFilt = self.CurveFilt - np.mean(self.CurveFilt[-int(10*SampFreq):-1])
            if self.StableAmp < 0:
                self.Change = 1
            else:
                self.Change  = -1

            self.CurveFilt1 = self.CurveFilt - self.Change*0.1*np.absolute(self.ReboundAmp) 
            i = 0
            while i < len(self.CurveFilt1):
                if self.CurveFilt1[i]< 0:
                    self.CurveFilt1[i] = 0
                if i > len(self.CurveFilt1)*3/4 and self.CurveFilt1[i-1]==0:
                    self.CurveFilt1[i] = 0
                i += 1
            self.CurveStartWave = self.CurveFilt1[0:self.ReboundTP]
            self.CurveStartTime = self.Time[0:self.ReboundTP]
            self.StartPre1, = np.where(np.diff(self.CurveStartWave)>0)            
            self.StartPre2, = np.where(np.diff(self.StartPre1[1:-1])>1)
            self.CurveStopWave = self.CurveFilt1[self.ReboundTP:-1]
            self.CurveStopTime = self.Time[self.ReboundTP:-1]
            self.StopPre1, =  np.where(np.diff(self.CurveStopWave)<0)
            self.StopPre2, = np.where(np.diff(self.StopPre1[1:-1])>0)
            if len(self.StartPre1) > 0 and len(self.StopPre1) > 0:
                if len(self.StartPre2) > 0:
                    self.StartP = self.StartPre1[self.StartPre2[0]-1]
                else:
                    self.StartP = self.StartPre1[0]
                if len(self.StopPre2) > 0 and self.StopPre2[-1]>40*self.SampFreq:
                    self.StopP = self.StopPre1[self.StopPre2[-1]-1]
                else:
                    self.StopP = self.StopPre1[-1]
                self.TimeArea = self.Time[self.StartP:(self.StopP+self.ReboundTP)]
                self.WaveArea = self.CurveFilt1[self.StartP:(self.StopP+self.ReboundTP)] 
                self.Area = self.Change*sp.integrate.simps(self.WaveArea,self.TimeArea)
                self.TimeFit = self.TimeArea
                self.AreaThresholdVm = np.zeros(len(self.TimeArea))
                self.AreaThresholdVm[:] = self.Change*((np.absolute(self.ReboundAmp)*0.10) + np.mean(self.Wave[-int(10*SampFreq):-1]))
                self.TauOff = None
            else:
                self.Area = 0
                self.TauOff = None
                self.TimeFit = None
                self.AreaThresholdVm = None  
        else:
            self.Area = 0
            self.TauOff = None   
            self.TimeFit = None
            self.AreaThresholdVm = None
#        print(self.Change)
#        plt.figure()
#        plt.ion()
##        plt.plot(self.Time,self.Wave)
#        plt.plot(self.Time,self.CurveFilt)
#        plt.plot(self.Time,self.CurveFilt1)
#        plt.plot(self.CurveStopTime,self.CurveStopWave)
#        plt.plot(self.CurveStartTime,self.CurveStartWave)
###        plt.plot(self.CurveStopTime[0:-1],np.diff(self.CurveStopWave))
###        plt.plot(self.CurveStopTime[0:-2],np.diff(np.diff(self.CurveStopWave)))
##        plt.plot(self.TimeFit,self.AreaThresholdVm)
#        a             

class Main:    
    def __init__ (self,Names,Time,Wave,Stimulus,SampFreq,PrintShow=0):
        self.Time = Time
        self.Wave = Wave
        self.Names = Names
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
        self.VStable = np.mean(self.Wave[(self.StimOffset-int(100*self.SampFreq)):(self.StimOffset)])
        self.VStableP = np.zeros((2,int(100*self.SampFreq)))
        self.VStableP[0] = self.Time[(self.StimOffset-int(100*self.SampFreq)):(self.StimOffset)]
        self.VStableP[1]= self.VStable
        self.VStableAmplitude = self.VStable - self.Baseline
        
        #Change Waves according to Depolarisation or Hyperpolarisation:
        if (self.Baseline - self.VStable) < 0:
            self.WaveMulti = 1
        else:
            self.WaveMulti = -1
            
        ''' Calc Maximum Response:  '''    
        self.MaxResponseTime = self.Time[self.StimOnset:self.StimOffset]
        self.MaxResponseWave = self.Wave[self.StimOnset:self.StimOffset]*self.WaveMulti
        
        self.MaxResponseValues = MaxResponse(self.MaxResponseTime,self.MaxResponseWave,self.SampFreq)
        
        self.MaxResponseTP = self.MaxResponseValues.MaxResponseT+self.StimOnsetTime
        self.MaxResponseT = self.MaxResponseValues.MaxResponseT
        self.MaxResponseVm = self.MaxResponseValues.MaxResponseVm*self.WaveMulti
        self.MaxFitCurve = self.MaxResponseValues.MaxFitCurve*self.WaveMulti
        self.MaxFitr = self.MaxResponseValues.MaxFitr
        self.MaxFitCurveT = self.MaxResponseValues.Time
        self.MaxFitTauOff = self.MaxResponseValues.MaxFitTauOff
        self.MaxMethod = self.MaxResponseValues.Method
        self.MaxAmplitude = self.MaxResponseVm - self.Baseline
        
        ''' Calc Tau To VMax:  '''
        self.TauTime = self.Time[0:int(self.MaxResponseTP*SampFreq)+int(10*SampFreq)]
        self.TauWave = self.Wave[0:int(self.MaxResponseTP*SampFreq)+int(10*SampFreq)]*self.WaveMulti
        
        #A: Tau to VMax:
        self.TauMaxValues = Tau(self.TauTime,self.TauWave,self.SampFreq,(self.MaxResponseVm*self.WaveMulti),self.StimOnsetTime)
        self.TauMaxTime = self.TauMaxValues.TauTime
        self.TauMaxVm = self.TauMaxValues.TauVm*self.WaveMulti
        self.TauMax = self.TauMaxValues.Tau
        self.TauMaxMethod = self.TauMaxValues.Method
        
        #A: Tau to VStable:
        self.TauVStableValues = Tau(self.TauTime,self.TauWave,self.SampFreq,(self.VStable*self.WaveMulti),self.StimOnsetTime)
        self.TauVStableTime = self.TauVStableValues.TauTime
        self.TauVStableVm = self.TauVStableValues.TauVm*self.WaveMulti
        self.TauVStable = self.TauVStableValues.Tau
        self.TauVStableMethod = self.TauVStableValues.Method
        
        ''' Rebound '''
        self.ReboundTime = self.Time[self.StimOffset:-1]
        self.ReboundWave = self.Wave[self.StimOffset:-1]*-1*self.WaveMulti
        
        self.ReboundValues = ReboundResponse(self.ReboundTime,self.ReboundWave,self.SampFreq)
        self.ReboundT = self.ReboundValues.MaxResponseT+self.StimOffTime
        self.ReboundVm = self.ReboundValues.MaxResponseVm*-1*self.WaveMulti
        self.ReboundAmplitude = (np.absolute(self.ReboundValues.MaxResponseVm)-np.absolute(self.Baseline))*self.WaveMulti
        self.ReboundFitCurve = self.ReboundValues.MaxFitCurve*-1*self.WaveMulti
        self.ReboundFitCurveT = self.ReboundTime
        self.ReboundFitr = self.ReboundValues.MaxFitr
        self.ReboundTauOff = self.ReboundValues.MaxFitTauOff
        self.ReboundMethod = self.ReboundValues.Method        
        
        ''' Sag '''
        self.SagValues = SagCalc(self.MaxResponseTime,self.MaxResponseWave,self.SampFreq,self.Baseline,self.MaxAmplitude,self.VStableAmplitude,self.MaxFitr,self.MaxFitCurve,self.MaxFitTauOff,self.MaxResponseT)
        self.SagIndex = self.SagValues.Index
        self.SagArea = self.SagValues.Area
        self.SagTauOff = self.SagValues.TauOff
        self.SagAreaT = self.SagValues.TimeFit
        self.SagAreaVm = self.SagValues.AreaThresholdVm

        ''' Rebound Calc '''                      
        self.ReboundsValues = ReboundCalc(self.ReboundTime,self.ReboundWave,self.SampFreq,self.Baseline,self.MaxAmplitude,self.ReboundAmplitude,self.ReboundFitr,self.ReboundFitCurve,self.ReboundTauOff,self.ReboundValues.MaxResponseT,self.VStableAmplitude)
        self.ReboundIndex = self.ReboundsValues.Index
        self.ReboundArea = self.ReboundsValues.Area
        self.ReboundTauOff = self.ReboundsValues.TauOff
        self.ReboundAreaT = self.ReboundsValues.TimeFit
        self.ReboundAreaVm = self.ReboundsValues.AreaThresholdVm
        
        ''' Rebound APs '''
        if np.max(self.ReboundWave*-1*self.WaveMulti) > 0:
#            print('Done')
            self.ReboundAPValues = SuprathresholdProperties.FindAPs2(self.ReboundTime,self.ReboundWave,self.SampFreq,2,1,0)
            self.ReboundAPsTimes = self.ReboundAPValues.APTimes+self.StimOffTime
            self.ReboundAPsVm = self.ReboundAPValues.APPeaks
            self.NumReboundAPs = self.ReboundAPValues.APNum
        else:
            self.ReboundAPsTimes = np.nan
            self.ReboundAPsVm = np.nan
            self.NumReboundAPs = 0  
        
        ''' Figure '''
        # Prepare for outside plotting: PlotWave [x] and Annot(ations) [x]!!!!!
        # Each Wave and Annotation as: ''' matplotlib.lines.Line2D '''
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
            self.Figure = plt.figure()
            self.Figure.set_dpi(300)
        # GridSpec:
        self.gs = gridspec.GridSpec(1,1)
        self.gs.update(left=0.05, bottom= 0.05, top = 0.95, right=0.95)
        
        ### Plot IV: 
        self.ax = plt.subplot(self.gs[0,0])
        
        # Each Line in list: PlotWave:
        self.PlotWave = [None] * 10
        
        self.PlotWave[0], = self.ax.plot(self.Time,self.Wave,'k')
        self.PlotWave[1], = self.ax.plot(self.BaselineP[0],self.BaselineP[1],color = [0.149, 0.670, 0.239])
        self.PlotWave[2], = self.ax.plot(self.VStableP[0],self.VStableP[1],color = [0.945, 0.874, 0.274])
        self.PlotWave[3], = self.ax.plot(self.MaxResponseTP,self.MaxResponseVm,'o',color = [0.945, 0.168, 0.117])
        self.PlotWave[4], = self.ax.plot(self.ReboundT,self.ReboundVm,'o', color = [0.274, 0.8, 0.945])
        
        # Conditional Plotting:
        if np.size(self.TauVStableTime) >= 2:
            self.PlotWave[5] = self.ax.plot(self.TauVStableTime,self.TauVStableVm,'-', color = [0.976, 0.949, 0.705])
            self.PHTauStableVm = np.mean(self.TauVStableVm)*1.05
            self.PHTauStableT = np.mean(self.TauVStableTime)*0.8
            self.PlotWave[5] = self.PlotWave[5][0]
        else:
            self.PlotWave[5] = self.ax.plot(self.TauVStableTime,self.TauVStableVm,'o', color = [0.976, 0.949, 0.705])
            self.PHTauStableVm = self.TauVStableVm
            self.PHTauStableT = self.TauVStableTime
            self.PlotWave[5] = self.PlotWave[5][0]
        
        if np.size(self.TauMaxTime) >= 2:
            self.PlotWave[6] = self.ax.plot(self.TauMaxTime,self.TauMaxVm,'-',color = [0.972, 0.545, 0.592])
            self.PHTauMaxVm = np.mean(self.TauMaxVm)*1.05
            self.PHTauMaxT = np.mean(self.TauMaxTime)*0.8
            self.PlotWave[6] = self.PlotWave[6][0]
        else:
            self.PlotWave[6] = self.ax.plot(self.TauMaxTime,self.TauMaxVm,'o',color = [0.972, 0.545, 0.592])
            self.PHTauMaxVm = self.TauMaxVm
            self.PHTauMaxT = self.TauMaxTime
            self.PlotWave[6] = self.PlotWave[6][0]
            
        if not (self.SagAreaT) is None or not self.SagAreaVm is None :  
            self.PlotWave[7] = self.ax.plot(self.SagAreaT,self.SagAreaVm,'--',color = [0.945, 0.682, 0.274])
            self.PlotWave[7] = self.PlotWave[7][0]
        else:
            self.PlotWave[7] = self.ax.plot(self.Time[0],self.Wave[0],'.k',)
            self.PlotWave[7] = self.PlotWave[7][0]
            
        if not (self.ReboundAreaT )is None or not self.ReboundAreaVm is None:
            self.PlotWave[8] = self.ax.plot(self.ReboundAreaT,self.ReboundAreaVm,'--', color = [0.274, 0.8, 0.945])
            self.PlotWave[8] = self.PlotWave[8][0]
        else:
            self.PlotWave[8] = self.ax.plot(self.Time[0],self.Wave[0],'.k')
            self.PlotWave[8] = self.PlotWave[8][0]
            
        self.PlotWave[9], = self.ax.plot(self.ReboundAPsTimes,self.ReboundAPsVm,'o', color = [0.6, 0.043, 0.215])
        
            
        # Each Annotation in list: Annot:
        self.Annot = [None] * 8
        self.Annot[0] = self.ax.annotate("Base: %.1f mV" % self.BaselineP[1][0],xy=(50,self.BaselineP[1][0]),xytext=(-20,15),color = [0.149, 0.670, 0.239],xycoords='data',textcoords='offset points')
        if self.WaveMulti >= 0:
            self.Annot[0] = self.ax.annotate("Base: %.1f mV" % self.BaselineP[1][0],xy=(50,self.BaselineP[1][0]),xytext=(-20,15),color = [0.149, 0.670, 0.239],xycoords='data',textcoords='offset points')
            self.Annot[1] = self.ax.annotate("VStable: %.1f mV" % self.VStableP[1][0],xy=(450,self.VStableP[1][0]),xytext=(-10,-15),color = [0.945, 0.874, 0.274],xycoords='data',textcoords='offset points')
            self.Annot[2] = self.ax.annotate("Max: %.1f mV" % self.MaxResponseVm,xy=(self.MaxResponseTP,self.MaxResponseVm),xytext=(-35,-15),color = [0.945, 0.168, 0.117],xycoords='data',textcoords='offset points')
            self.Annot[3] = self.ax.annotate("SagIndex: %.1f %%" % self.SagIndex,xy=(self.MaxResponseTP,self.MaxResponseVm),xytext=(-35,-25),color = [0.945, 0.168, 0.117],xycoords='data',textcoords='offset points')
            self.Annot[4] = self.ax.annotate("TauVStable: %.1f ms" % self.TauVStable,xy=(self.PHTauStableT,self.PHTauStableVm),xytext=(1,5),color = [0.976, 0.949, 0.705],xycoords='data',textcoords='offset points')
            self.Annot[5] = self.ax.annotate("TauMax: %.1f ms" % self.TauMax,xy=(self.PHTauMaxT,self.PHTauMaxVm),xytext=(1,5),color = [0.972, 0.545, 0.592],xycoords='data',textcoords='offset points')
            self.Annot[6] = self.ax.annotate("Rebound: %.1f mV" % self.ReboundVm,xy=(self.ReboundT,self.ReboundVm),xytext=(-95,15),color = [0.274, 0.8, 0.945],xycoords='data',textcoords='offset points')
            self.Annot[7] = self.ax.annotate("RebIndex: %.1f %%" % self.ReboundIndex,xy=(self.ReboundT,self.ReboundVm),xytext=(-95,25),color = [0.274, 0.8, 0.945],xycoords='data',textcoords='offset points')
        else:
            self.Annot[0] = self.ax.annotate("Base: %.1f mV" % self.BaselineP[1][0],xy=(50,self.BaselineP[1][0]),xytext=(-20,-15),color = [0.149, 0.670, 0.239],xycoords='data',textcoords='offset points')
            self.Annot[1] = self.ax.annotate("VStable: %.1f mV" % self.VStableP[1][0],xy=(450,self.VStableP[1][0]),xytext=(-10,15),color = [0.945, 0.874, 0.274],xycoords='data',textcoords='offset points')
            self.Annot[2] = self.ax.annotate("Max: %.1f mV" % self.MaxResponseVm,xy=(self.MaxResponseTP,self.MaxResponseVm),xytext=(-35,15),color = [0.945, 0.168, 0.117],xycoords='data',textcoords='offset points')
            self.Annot[3] = self.ax.annotate("SagIndex: %.1f %%" % self.SagIndex,xy=(self.MaxResponseTP,self.MaxResponseVm),xytext=(-35,25),color = [0.945, 0.168, 0.117],xycoords='data',textcoords='offset points')
            self.Annot[4] = self.ax.annotate("TauVStable: %.1f ms" % self.TauVStable,xy=(self.PHTauStableT,self.PHTauStableVm),xytext=(1,5),color = [0.976, 0.949, 0.705],xycoords='data',textcoords='offset points')
            self.Annot[5] = self.ax.annotate("TauMax: %.1f ms" % self.TauMax,xy=(self.PHTauMaxT,self.PHTauMaxVm),xytext=(1,5),color = [0.972, 0.545, 0.592],xycoords='data',textcoords='offset points')
            self.Annot[6] = self.ax.annotate("Rebound: %.1f mV" % self.ReboundVm,xy=(self.ReboundT,self.ReboundVm),xytext=(-95,-15),color = [0.274, 0.8, 0.945],xycoords='data',textcoords='offset points')
            self.Annot[7] = self.ax.annotate("RebIndex: %.1f %%" % self.ReboundIndex,xy=(self.ReboundT,self.ReboundVm),xytext=(-95,-25),color = [0.274, 0.8, 0.945],xycoords='data',textcoords='offset points')
    
        # Self Plotting Scales:
        self.ax.set_xlabel('ms')
        self.ax.set_ylabel('mV')
        self.ax.set_xlim(self.Time[0],self.Time[-1])

        # Saving:
        if self.PrintShow == 1:
            self.SavingName = 'SingleSubThresholdPlot'
            ddPlotting.save(self.SavingName, ext="jpg", close=True, verbose=True)
            #ddPlotting.save(self.SavingName, ext="svg", close=True, verbose=True)
            plt.close('All')
            plt.ion()  


class IVHelp:
    def __init__ (self,CurrentAmps,ResponseAmps,RangeIR):
        self.CurrentAmps = CurrentAmps      # as np.array
        self.ResponseAmps = ResponseAmps    # as np.array
        self.RangeIR = RangeIR              # as np.array [Start,End] in pA
        
        # Get Boarders for 2 polynome fit:
        self.lengthSub = len(self.ResponseAmps) 
        self.Current2Poly = self.CurrentAmps[0:self.lengthSub]
        
        # Get Boarders for IR Calculation
        self.IVStart = np.where(self.CurrentAmps == self.RangeIR[0])
        self.IVStart = int(self.IVStart[0][0])
        self.IVEnd = np.where(self.CurrentAmps == self.RangeIR[1])
        if hasattr(self, 'IVEnd'):
            self.IVEnd = int(self.IVEnd[0][0]+1)

        if not self.IVEnd or self.IVEnd >= len(self.ResponseAmps)+1:
            self.IVEnd = len(self.ResponseAmps)
            
        # Get Curves
        self.CurrentAmpsPoly = self.CurrentAmps[0:self.lengthSub] 
        self.IVWaves = np.empty(shape=[2,(self.IVEnd-self.IVStart)])
        self.IVWaves[0][:] = self.CurrentAmps[self.IVStart:self.IVEnd] 
        self.IVWaves[1][:]= self.ResponseAmps[self.IVStart:self.IVEnd] 
        
        # Fit 2 polynome fit
        self.PolyFit = np.polyfit(self.CurrentAmpsPoly,self.ResponseAmps,3)
        
        self.PolyFitCurveT = np.linspace(np.min(self.CurrentAmpsPoly),np.max(self.CurrentAmpsPoly),50)
        self.PolyFitCurve = self.PolyFit[0]*self.PolyFitCurveT**3+self.PolyFit[1]*self.PolyFitCurveT**2+self.PolyFit[2]*self.PolyFitCurveT+self.PolyFit[3]

        # Fit IR:
        def f(x,a):
            return a*x
        self.FitParams, self.FitCov = curve_fit(f,self.IVWaves[0],self.IVWaves[1])  
        self.IRCurve = self.IVWaves[0]*self.FitParams[0]
        self.IRCurveAll = self.CurrentAmpsPoly*self.FitParams[0]
        self.IR = self.FitParams[0]*1000  
        
        # Calculation of Fast Rectification from Stimulation of -100pA and justsub:
        self.VExpectHypo = -100*self.IR/1000
        self.FastRecHypo = ((self.VExpectHypo-self.ResponseAmps[1])/self.VExpectHypo)*100
        
        self.VexpectDepo = self.CurrentAmpsPoly[-1]*self.IR/1000
        self.FastRecDepo = ((self.VexpectDepo-self.ResponseAmps[-1])/self.VexpectDepo)*100
        
class IVCalc:
    def __init__ (self,CurrentAmps,MaxAmps,VStableAmps,RangeIR,PrintShow=0):
        # Does All IV Calculations
        # Needs MaxAmplitudes,VStable Amplitudes of all Subthreshold Traces
        self.CurrentAmps = CurrentAmps      # as np.array
        self.MaxAmps = MaxAmps              # as np.array
        self.VStableAmps = VStableAmps      # as np.array
        self.RangeIR = RangeIR              # as np.array
        self.PrintShow = PrintShow
        
        # Calculate IV for Max Responses:
        self.IVMaxValues = IVHelp(self.CurrentAmps,self.MaxAmps,self.RangeIR) 
        self.IRMax = self.IVMaxValues.IR
        self.MaxFastRecHypoMax = self.IVMaxValues.FastRecHypo
        self.MaxFastRecDepoMax = self.IVMaxValues.FastRecDepo
        
        # Calculate IV for VStable Responses:
        self.IVVStableValues = IVHelp(self.CurrentAmps,self.VStableAmps,self.RangeIR)
        self.IRVStable = self.IVVStableValues.IR
        self.VStableFastRecHypoMax = self.IVVStableValues.FastRecHypo
        self.VStableFastRecDepoMax = self.IVVStableValues.FastRecDepo
                
        ''' Figure '''
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
            self.Figure = plt.figure()
            self.Figure.set_dpi(300)
        # GridSpec:
        self.gs = gridspec.GridSpec(1,1)
        self.gs.update(left=0.05, bottom= 0.05, top = 0.95, right=0.95)
        
        ### Plot IV: 
        self.ax = plt.subplot(self.gs[0,0])
        
#        # Prepare for outside plotting: PlotWave [x] and Legends [x]!!!!!
#        # Each Wave and Annotation as: ''' matplotlib.lines.Line2D '''
#        self.Figure,self.ax = plt.subplots()
#        # Prevent from popping up:
#        plt.ioff()
#        plt.close()
        
        # Each Line in list: PlotWave:
        self.PlotWave = [None] * 8
        
        self.PlotWave[0], = self.ax.plot(self.IVVStableValues.CurrentAmpsPoly,self.IVVStableValues.ResponseAmps,'o',color = [0.992, 0.647, 0.407])
        self.PlotWave[1], = self.ax.plot(self.IVVStableValues.PolyFitCurveT,self.IVVStableValues.PolyFitCurve,color = [0.992, 0.647, 0.407])
        self.PlotWave[2], = self.ax.plot(self.IVVStableValues.CurrentAmpsPoly,self.IVVStableValues.IRCurveAll,'--',color = [0.713, 0.709, 0.705])
        self.PlotWave[3], = self.ax.plot(self.IVVStableValues.IVWaves[0],self.IVVStableValues.IRCurve,'-',color = [0.713, 0.709, 0.705])
        
        self.PlotWave[4], = self.ax.plot(self.IVMaxValues.CurrentAmpsPoly,self.IVMaxValues.ResponseAmps,'o',color = [0.203, 0.247, 0.937])
        self.PlotWave[5], = self.ax.plot(self.IVMaxValues.PolyFitCurveT,self.IVMaxValues.PolyFitCurve,color = [0.203, 0.247, 0.937])
        self.PlotWave[6], = self.ax.plot(self.IVMaxValues.CurrentAmpsPoly,self.IVMaxValues.IRCurveAll,'--k')
        self.PlotWave[7], = self.ax.plot(self.IVMaxValues.IVWaves[0],self.IVMaxValues.IRCurve,'-k')
        
        # Each Legend in list: legend:
        self.legend = [None] * 2
        self.legend[0] = self.ax.legend([self.PlotWave[0],self.PlotWave[3]],['Max Response',"IR Max: %.1f MOhm" % self.IRMax],loc='upper left')
        self.legend[1] = self.ax.legend([self.PlotWave[4],self.PlotWave[7]],['VStable Response',"IR VStable: %.1f MOhm" % self.IRVStable],loc='lower right')
        #plt.gca().add_artist(self.legend[0])
        
        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['bottom'].set_position('zero')
        
        # Eliminate upper and right axes
        self.ax.spines['right'].set_color('none')
        self.ax.spines['top'].set_color('none')
        
        # Show ticks in the left and lower axes only
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')

class InstantTauHelp:
    def __init__ (self,CurrentAmps,Taus,Range):                    
        self.CurrentAmps = CurrentAmps     
        self.Taus = Taus
        self.Range = Range
        
        # Get Boarders for Tau Calculation
        self.TauCalcStart = np.where(self.CurrentAmps == self.Range[0])
        self.TauCalcStart = int(self.TauCalcStart[0][0])
        self.TauCalcEnd = np.where(self.CurrentAmps == self.Range[-1])

        if hasattr(self, 'TauCalcEnd'):
            self.TauCalcEnd = int(self.TauCalcEnd[0][0]+1)

        if self.TauCalcEnd >= len(self.Taus)+1:
            self.TauCalcEnd = len(self.Taus)
        
        # Get Curves
        self.TauWaves = np.empty(shape=[2,(self.TauCalcEnd-self.TauCalcStart)])
        self.TauWaves[0][:] = self.CurrentAmps[self.TauCalcStart:self.TauCalcEnd] 
        self.TauWaves[1][:]= self.Taus[self.TauCalcStart:self.TauCalcEnd]  
        
        self.IndexNaN = np.where(np.isnan(self.TauWaves[1]))
        self.TauWaves = np.delete(self.TauWaves,self.IndexNaN,axis =1 )

        # Fit IR:
        def f(x,a):
            return x+a
        self.FitParams, self.FitCov = curve_fit(f,self.TauWaves[0],self.TauWaves[1])  
        self.InstTau = self.FitParams[0]
        self.TauCurve = self.TauWaves[0]*0+self.FitParams[0]
        
class InstantTauCalc:
    def __init__ (self,CurrentAmps,TausMax,TausVStable,Range,PrintShow = 0):        
        self.CurrentAmps = CurrentAmps
        self.TausMax = TausMax
        self.TausVStable = TausVStable
        self.Range = Range
        self.PrintShow = PrintShow
        
        # Calculate Instant Tau for Max Responses:
        self.TauMaxValues = InstantTauHelp(self.CurrentAmps,self.TausMax,self.Range) 
        self.InstTauMax = self.TauMaxValues.InstTau
        
        # Calculate Instant Tau for Stable Responses:
        self.TauVStableValues = InstantTauHelp(self.CurrentAmps,self.TausVStable,self.Range) 
        self.InstTauStable = self.TauVStableValues.InstTau
        
        ''' Figure '''
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
            self.Figure = plt.figure()
            self.Figure.set_dpi(300)
        # GridSpec:
        self.gs = gridspec.GridSpec(1,1)
        self.gs.update(left=0.05, bottom= 0.05, top = 0.95, right=0.95)
        
        ### Plot IV: 
        self.ax = plt.subplot(self.gs[0,0])
#        
#        # Prepare for outside plotting: PlotWave [x] and Legends [x]!!!!!
#        # Each Wave and Annotation as: ''' matplotlib.lines.Line2D '''
#        self.Figure,self.ax = plt.subplots()
#        # Prevent from popping up:
#        plt.ioff()
#        plt.close()
        
        # Each Line in list: PlotWave:
        self.PlotWave = [None] * 6
        
        self.PlotWave[0], = self.ax.plot(self.TauVStableValues.TauWaves[0],self.TauVStableValues.TauWaves[1],'o',color = [0.992, 0.647, 0.407])
        self.PlotWave[1], = self.ax.plot(self.TauVStableValues.TauWaves[0],self.TauVStableValues.TauCurve, '-', color = [0.992, 0.647, 0.407])
        self.PlotWave[2], = self.ax.plot(0,self.InstTauStable,'o',color = [0.713, 0.709, 0.705])
    
        self.PlotWave[3], = self.ax.plot(self.TauMaxValues.TauWaves[0],self.TauMaxValues.TauWaves[1],'o',color = [0.203, 0.247, 0.937])
        self.PlotWave[4], = self.ax.plot(self.TauMaxValues.TauWaves[0],self.TauMaxValues.TauCurve, '-', color = [0.203, 0.247, 0.937])
        self.PlotWave[5], = self.ax.plot(0,self.InstTauMax,'ok')
        
        # Each Legend in list: legend:
        self.legend = [None] * 2
        self.legend[0] = self.ax.legend([self.PlotWave[0],self.PlotWave[2]],['Max Response',"Tau Max: %.1f ms" % self.InstTauMax],loc='upper left')
        self.legend[1] = self.ax.legend([self.PlotWave[3],self.PlotWave[5]],['VStable Response',"Tau VStable: %.1f ms" % self.InstTauStable],loc='lower right')
#        plt.gca().add_artist(self.legend[0])

class PassiveGroundValues:
    def __init__ (self,MainValues,Names):
        self.MainValues = MainValues
        self.Names = Names
        
        self.Lens = len(self.MainValues)
        i = 0
        self.MaxResponseMethod = [None] * self.Lens
        self.TauMaxMethod = [None] * self.Lens
        self.TauStableMethod = [None] * self.Lens
        
        while i < self.Lens:
            self.MaxResponseMethod[i] = self.MainValues[i].MaxMethod
            self.TauMaxMethod[i] = self.MainValues[i].TauMaxMethod
            self.TauStableMethod[i] = self.MainValues[i].TauVStableMethod
            i += 1  
        # Single Values:    
        self.BaselineMean = ddhelp.Extract(self.MainValues,'Baseline')
        self.BaselineSD = ddhelp.Extract(self.MainValues,'BaselineSD')
        
        self.MaxAmplitude = ddhelp.Extract(self.MainValues,'MaxResponseVm')
        self.MaxR = ddhelp.Extract(self.MainValues,'MaxFitr')
        self.StableAmplitude = ddhelp.Extract(self.MainValues,'VStableAmplitude')
        
        self.TauMaxVm = ddhelp.Extract(self.MainValues,'TauMaxVm')
        self.TauMax = ddhelp.Extract(self.MainValues,'TauMax')
        self.TauStableVm = ddhelp.Extract(self.MainValues,'TauVStableVm')
        self.TauStable = ddhelp.Extract(self.MainValues,'TauVStable')
         
        # Matrix for Table as panda Object: 
        self.Header = ['WaveName','BaselineMean[mV]','BaselineSD[mV]','MaxAmplitude[mV]',\
                       'MaxMethod','MaxR','StableResponse[mV]','TauMax[mV]',\
                       'TauMaxMethod','TauMax[ms]','TauStable[mV]','TauStableMethod',\
                       'TauStable[ms]']
        self.Table = pd.DataFrame(self.Names)
        self.Table = pd.concat([self.Table,\
                                pd.DataFrame(self.BaselineMean.All),\
                                pd.DataFrame(self.BaselineSD.All),\
                                pd.DataFrame(self.MaxAmplitude.All),\
                                pd.DataFrame(self.MaxResponseMethod),\
                                pd.DataFrame(self.MaxR.All),\
                                pd.DataFrame(self.StableAmplitude.All),\
                                pd.DataFrame(self.TauMaxVm.All),\
                                pd.DataFrame(self.TauMaxMethod),\
                                pd.DataFrame(self.TauMax.All),\
                                pd.DataFrame(self.TauStableVm.All),\
                                pd.DataFrame(self.TauStableMethod),\
                                pd.DataFrame(self.TauStable.All),\
                                ],axis=1) # join_axes=[self.Table.index]
        self.Table.columns = self.Header


class PassiveValues:
    def __init__ (self,IVValues,TauValues,VRest=0):
        self.IVValues = IVValues
        self.TauValues = TauValues
        # Get Values:
        self.VRest = VRest
        self.IRmax = self.IVValues.IRMax
        self.IRVStable = self.IVValues.IRVStable 
        self.TauMax = self.TauValues.InstTauMax
        self.TauVStable = self.TauValues.InstTauStable
        self.FastRecHypo = self.IVValues.MaxFastRecHypoMax # At Current Input -100
        self.FastRecDepo= self.IVValues.MaxFastRecDepoMax # At Current Input -100
        
        # Get Values for Table:
        self.VRest1 = [None]*1
        self.IRmax1 = [None]*1
        self.IRVStable1 = [None]*1
        self.TauMax1 = [None]*1
        self.TauVStable1 = [None]*1
        self.FastRecHypo1 = [None]*1
        self.FastRecDepo1 = [None]*1 
        self.VRest1[0] = VRest
        self.IRmax1[0] = self.IVValues.IRMax
        self.IRVStable1[0] = self.IVValues.IRVStable 
        self.TauMax1[0] = self.TauValues.InstTauMax
        self.TauVStable1[0] = self.TauValues.InstTauStable
        
        self.FastRecHypo1[0] = self.IVValues.MaxFastRecHypoMax # At Current Input -100
        self.FastRecDepo1[0] = self.IVValues.MaxFastRecDepoMax # At Current Input -100
       
        self.FastRecHypoStable = self.IVValues.VStableFastRecHypoMax # At Current Input JustSub
        self.FastRecDepoStable = self.IVValues.VStableFastRecDepoMax # At Current Input JustSub
        
        self.TableForMain =  pd.DataFrame(self.VRest1)
        self.TableForMain = pd.concat([self.TableForMain,\
                                pd.DataFrame(self.IRmax1),\
                                pd.DataFrame(self.IRVStable1),\
                                pd.DataFrame(self.TauMax1),\
                                pd.DataFrame(self.TauVStable1),\
                                pd.DataFrame(self.FastRecHypo1),\
                                pd.DataFrame(self.FastRecDepo1),\
                                ],axis=1) #join_axes=[self.TableForMain.index
        self.Header = ['VRest[mV]','IRMax[MOhm]','IRStable[MOhm]','TauMax[ms]','TauStable[ms]',\
                       'FastRecHypo','FastRecDepo']
        self.TableForMain.columns = self.Header


class PlotPassive:
    def __init__ (self,Waves,IVCalc,TauCalc,ValuesToPrint,PrintOrShow = 0,CellName = 'NA'):
        # Values to Print: IRMax/VStable, TauMax/VStable, ... Resting Potential
        self.SingleWaves = Waves
        self.Waves = [None] * len(self.SingleWaves)
        self.WaveNames = [None] * len(self.SingleWaves)
        i = 0
        while i < len(Waves):
            self.Waves[i] = ddPlotting.ExtractPlotting(self.SingleWaves[i])
            self.WaveNames[i] = self.SingleWaves[i].Names
            i += 1
        
        self.IVCalc = IVCalc
        self.IV = ddPlotting.ExtractPlotting(self.IVCalc) 
        self.TauCalc = TauCalc
        self.Tau = ddPlotting.ExtractPlotting(self.TauCalc)
        self.PrintShow = PrintOrShow
        self.CurrentInputs = self.IVCalc.CurrentAmps
        self.Values = ValuesToPrint
        self.CellName = CellName
    
        ''' Subplot Config: All: (6,3,X)'''
        self.SubpWaves = [1,4,7,10,13,16,14,17,3,6,9,12,15,18]
        self.NumWaves = len(self.Waves)
        self.SubWavesTaken = self.SubpWaves[0:self.NumWaves]
        self.SubWavesNotTaken = self.SubpWaves[self.NumWaves:-1]
        self.WavesPlot = len(self.Waves[0].Waves)
        self.WavesAnnot = len(self.Waves[0].Annot)
        self.SubIV = [2,5]
        self.SubIVWaves = len(self.IV.Waves)
        self.SubIVLegends = len(self.IV.Legend)
        self.SubTau = [11]
        self.SubTauWaves = len(self.Tau.Waves)
        self.SubTauLegends = len(self.Tau.Legend)
        self.Markersize = 5
    
    
        ''' Figure '''
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
            self.Figure = plt.figure()
            self.Figure.set_size_inches(11.69, 8.27, forward=True)
        if self.PrintShow == 1:
            self.Figure.set_dpi(300)    
        # GridSpec:
        self.AX = [None] * (self.NumWaves)
        self.gs = gridspec.GridSpec(6, 3)
        self.gs.update(left=0.05, bottom= 0.05, top = 0.95, right=0.95, wspace=0.2)
        
        # Set Grid for Plotting: Row/Colum
        if self.NumWaves >= 1:
            self.AX[0] = plt.subplot(self.gs[0,0])
        if self.NumWaves >= 2:
            self.AX[1] = plt.subplot(self.gs[1,0])
        if self.NumWaves >= 3:
            self.AX[2] = plt.subplot(self.gs[2,0])
        if self.NumWaves >= 4:
            self.AX[3] = plt.subplot(self.gs[3,0])
        if self.NumWaves >= 5:
            self.AX[4] = plt.subplot(self.gs[4,0])
        if self.NumWaves >= 6:
            self.AX[5] = plt.subplot(self.gs[5,0])
        if self.NumWaves >= 7:
            self.AX[6] = plt.subplot(self.gs[4,1])
        if self.NumWaves >= 8:
            self.AX[7] = plt.subplot(self.gs[5,1])
        if self.NumWaves >= 9:
            self.AX[8] = plt.subplot(self.gs[0,2])
        if self.NumWaves >= 10:
            self.AX[9] = plt.subplot(self.gs[1,2])
        if self.NumWaves >= 11:
            self.AX[10] = plt.subplot(self.gs[2,2])
        if self.NumWaves >= 12:    
            self.AX[11] = plt.subplot(self.gs[3,2])
        if self.NumWaves >= 13:    
            self.AX[12] = plt.subplot(self.gs[4,2])
        if self.NumWaves >= 14:    
            self.AX[13] = plt.subplot(self.gs[5,2])
            
        self.axIV = plt.subplot(self.gs[1:3, 1])
        self.axTau = plt.subplot(self.gs[3, 1])
        
        # Plot Waves:
        i = 0
        
        while i < self.NumWaves:
            
            j = 0
            while j < self.WavesPlot:
                
                # Set Markersize
                if self.Waves[i].Waves[j].get_marker() == '.' or self.Waves[i].Waves[j].get_marker() == 'o':
                    msset = self.Markersize
                else:
                    msset = 1.0
                    
                self.AX[i].plot(self.Waves[i].Waves[j].get_data()[0], self.Waves[i].Waves[j].get_data()[1],self.Waves[i].Waves[j].get_marker(), linestyle = self.Waves[i].Waves[j].get_linestyle(), color = self.Waves[i].Waves[j].get_color(), ms = msset)
                j +=1   
                
            # Setting Axes,...
            self.AX[i].set_xlim(0,800)
            self.AX[i].xaxis.set_ticks(np.arange(0,800,100))
            self.AX[i].set_ylabel('mV',fontsize = 6, rotation = 90)
            self.AX[i].spines["top"].set_visible(False)
            self.AX[i].spines["right"].set_visible(False)
            
            plt.setp(self.AX[i].get_yticklabels(),rotation = 'vertical', fontsize = 6)
            if i == 6 or i is 8 or i == self.NumWaves:
                #print('set')
                self.AX[i].set_xlabel('ms',fontsize = 6)
                plt.setp(self.AX[i].get_xticklabels(), fontsize = 6)
            else:
#                print('deleted')
                self.AX[i].set_xticklabels([])
                
            # Title:
            self.AX[i].set_title(self.WaveNames[i]+" Current Input: %i pA" % self.CurrentInputs[i], fontsize = 6, y=0.9)
            
            
            i +=1    
             
        # Plot Annotations
        i = 0
        self.PlottedAnnot = []
        
        while i < self.NumWaves:
            self.PlottedAnnot.append([i])
            k = 0
            while k < self.WavesAnnot:
                self.PlottedAnnot[i].append(k)
                self.PlottedAnnot[i][k] = self.AX[i].annotate(self.Waves[i].Annot[k]._text, self.Waves[i].Annot[k].xy,self.Waves[i].Annot[k].xyann, self.Waves[i].Annot[k].xycoords, self.Waves[i].Annot[k]._textcoords, fontsize = 4,color = self.Waves[i].Annot[k].get_color())
                a = self.PlottedAnnot[i][k].get_position()
                if k == 0:
                    self.PlottedAnnot[i][k].set_position((a[0]/2,a[1]/3))
                if k == 1:
                    self.PlottedAnnot[i][k].set_position((a[0]/5,a[1]/4))
                if k == 2:
                    self.PlottedAnnot[i][k].set_position((a[0]/5,a[1]/5))
                if k == 3:
                    self.PlottedAnnot[i][k].set_position((a[0]/5,a[1]/4))
                if k == 4:
                    self.PlottedAnnot[i][k].set_position((a[0]/1,a[1]/3))
                if k == 5:
                    self.PlottedAnnot[i][k].set_position((a[0]/1,a[1]/3))
                if k == 6:
                    self.PlottedAnnot[i][k].set_position((a[0]/4,a[1]/2))
                if k == 7:
                    self.PlottedAnnot[i][k].set_position((a[0]/4,a[1]/2))
                k +=1   
            i+=1
        
        # Plotting IV Curve
        j = 0
        self.YLIMIT = np.zeros((2, 2))
        
        while j < self.SubIVWaves:
            if  self.IV.Waves[j].get_marker() == '.' or  self.IV.Waves[j].get_marker() == 'o':
                    msset = self.Markersize
            else:
                msset = 0.5
            # Printing:    
            self.axIV.plot(self.IV.Waves[j].get_data()[0], self.IV.Waves[j].get_data()[1], self.IV.Waves[j].get_marker(), linestyle = self.IV.Waves[j].get_linestyle(), color = self.IV.Waves[j].get_color(), ms = msset) 
            
            if j == 0: 
                self.YLIMIT[0][0] = np.max(self.IV.Waves[j].get_data()[1])
                self.YLIMIT[1][0] = np.min(self.IV.Waves[j].get_data()[1])
            if j == 4:
                self.YLIMIT[0][1] = np.max(self.IV.Waves[j].get_data()[1])
                self.YLIMIT[1][1] = np.min(self.IV.Waves[j].get_data()[1])
            j +=1
        j = 0
        
        # Legends:
        self.IVLegend0 = self.axIV.legend([self.IV.Waves[4],self.IV.Waves[7],self.IV.Waves[0],self.IV.Waves[3]],['Max Responses',"IR Max: %.1f MOhm" % self.Values.IRmax,'VStable Responses',"IR VStable: %.1f MOhm" % self.Values.IRVStable],loc='upper left',fontsize = 6)

        # IV Axes:
        self.axIV.spines['left'].set_position('zero')
        self.axIV.spines['bottom'].set_position('zero')
        # Eliminate upper and right axes
        self.axIV.spines['right'].set_color('none')
        self.axIV.spines['top'].set_color('none')
        # Show ticks in the left and lower axes only
        self.axIV.xaxis.set_ticks_position('bottom')
        self.axIV.yaxis.set_ticks_position('left')    
        # Set Labels and Sizes:
        self.axIV.set_ylabel('mV',fontsize = 6, rotation = 90)
        plt.setp(self.axIV.get_yticklabels(), fontsize = 6)
        self.axIV.set_xlabel('pA',fontsize = 6)
        plt.setp(self.axIV.get_xticklabels(), fontsize = 6) 
        self.axIV.set_ylim([np.min(self.YLIMIT[1])+(0.2*np.min(self.YLIMIT[1])),np.max(self.YLIMIT[0])+0.2*np.max(self.YLIMIT[0])])
        self.axIV.set_xlim([np.min(self.IV.Waves[0].get_data()[0]),np.max(self.IV.Waves[0].get_data()[0])])
        
        self.OVTitle = 'Passive Properties of \n' + self.CellName + ':'
        self.axIV.set_title(self.OVTitle,fontsize = 14, y = 1.25,fontweight='bold',loc='left')
        
        # Plotting Tau Curve
        j = 0
        while j < self.SubTauWaves:
            if  self.Tau.Waves[j].get_marker() == '.' or  self.Tau.Waves[j].get_marker() == 'o':
                    msset = self.Markersize
            else:
                msset = 0.5
            # Printing:    
            self.axTau.plot(self.Tau.Waves[j].get_data()[0], self.Tau.Waves[j].get_data()[1], self.Tau.Waves[j].get_marker(), linestyle = self.Tau.Waves[j].get_linestyle(), color = self.Tau.Waves[j].get_color(), ms = msset) 
            j +=1
        j = 0
        
        # Legends:
        self.TauLegend0 = self.axTau.legend([self.Tau.Waves[3],self.Tau.Waves[5],self.Tau.Waves[0],self.Tau.Waves[2]],['Max Responses',"Tau Max: %.1f ms" % self.Values.TauMax,'VStable Responses',"Tau VStable: %.1f ms" % self.Values.TauVStable],loc='upper left',fontsize = 6)

        # IV Axes:
        self.axTau.spines['left'].set_position('zero')
        #self.axTau.spines['bottom'].set_position('zero')
        # Eliminate upper and right axes
        self.axTau.spines['right'].set_color('none')
        self.axTau.spines['top'].set_color('none')
        # Show ticks in the left and lower axes only
        self.axTau.xaxis.set_ticks_position('bottom')
        self.axTau.yaxis.set_ticks_position('left')    
        # Set Labels and Sizes:
        self.axTau.set_ylabel('ms',fontsize = 6, rotation = 90)
        plt.setp(self.axTau.get_yticklabels(), fontsize = 6)
        self.axTau.set_xlabel('pA',fontsize = 6)
        plt.setp(self.axTau.get_xticklabels(), fontsize = 6)   
        self.axTau.set_xlim([np.min(self.Tau.Waves[0].get_data()[0]),np.max(self.Tau.Waves[0].get_data()[0])])    
        
        # Saving
        if self.PrintShow == 1:
            self.SavingName = self.CellName+'_PassiveCalculations'
            ddPlotting.save(self.SavingName, ext="png", close=True, verbose=True)
#            ddPlotting.save(self.SavingName, ext="svg", close=True, verbose=True)
            plt.ion()
        

#Getting Subthreshold Active Values
class SubActiveValues:
    def __init__ (self,MainValues,Names,NumWaves):
        self.MainValues = MainValues
        self.Names1 = Names
        self.Lens = NumWaves
        
        self.Names =[None]*self.Lens
        if self.Lens <2:
            self.Names[0] = self.Names1
        else:
            self.Names = self.Names1
        
        self.SagIndex = ddhelp.Extract(self.MainValues,'SagIndex')
        self.SagArea = ddhelp.Extract(self.MainValues,'SagArea')
        self.SagTauOff = ddhelp.Extract(self.MainValues,'SagTauOff')
        
        self.ReboundIndex = ddhelp.Extract(self.MainValues,'ReboundIndex')
        self.ReboundArea = ddhelp.Extract(self.MainValues,'ReboundArea')
        self.ReboundTauOff = ddhelp.Extract(self.MainValues,'ReboundTauOff')
        self.ReboundAPs = ddhelp.Extract(self.MainValues,'NumReboundAPs')
        
        # Matrix for Table as panda Object:  
        self.BaselineMean = ddhelp.Extract(self.MainValues,'Baseline')
        self.BaselineSD = ddhelp.Extract(self.MainValues,'BaselineSD')
        self.StableAmplitude = ddhelp.Extract(self.MainValues,'VStableAmplitude')
        
        self.MaxAmplitude = ddhelp.Extract(self.MainValues,'MaxAmplitude')
        self.MaxR = ddhelp.Extract(self.MainValues,'MaxFitr')
        self.ReboundAmplitude = ddhelp.Extract(self.MainValues,'ReboundAmplitude')
        self.ReboundR = ddhelp.Extract(self.MainValues,'ReboundFitr')
        
        # Table:
        i = 0
        self.MaxResponseMethod = [None] * self.Lens
        self.ReboundMethod = [None] * self.Lens
        while i < self.Lens:
            self.MaxResponseMethod[i] = self.MainValues[i].MaxMethod
            self.ReboundMethod[i] = self.MainValues[i].ReboundMethod
            i += 1  
        
        self.Header = ['WaveName','BaselineMean[mV]','BaselineSD[mV]','StableResponse[mV]',\
                       'MaxAmplitude[mV]','MaxMethod','MaxR','SagIndex[%]','SagArea[mV2/ms]',\
                       'SagTauOf[ms]','ReboundAmp[mV]','ReboundMethod','ReboundR',\
                       'ReboundIndex[%]','ReboundArea[mV2/ms]','ReboundTauOff[ms]','ReboundAPs[#]']
        self.Table = pd.DataFrame(self.Names)
        self.Table = pd.concat([self.Table,\
                                pd.DataFrame(self.BaselineMean.All),\
                                pd.DataFrame(self.BaselineSD.All),\
                                pd.DataFrame(self.StableAmplitude.All),\
                                pd.DataFrame(self.MaxAmplitude.All),\
                                pd.DataFrame(self.MaxResponseMethod),\
                                pd.DataFrame(self.MaxR.All),\
                                
                                pd.DataFrame(self.SagIndex.All),\
                                pd.DataFrame(self.SagArea.All),\
                                pd.DataFrame(self.SagTauOff.All),\
                                
                                pd.DataFrame(self.ReboundAmplitude.All),\
                                pd.DataFrame(self.ReboundMethod),\
                                pd.DataFrame(self.ReboundR.All),\
                                
                                pd.DataFrame(self.ReboundIndex.All),\
                                pd.DataFrame(self.ReboundArea.All),\
                                pd.DataFrame(self.ReboundTauOff.All),\
                                pd.DataFrame(self.ReboundAPs.All),\
                                ],axis=1) # join_axes=[self.Table.index]
        self.Table.columns = self.Header
        

# Plotting SubActive Plot
class SubActivePlot:
    def __init__ (self,Waves,Values,PrintShow = 0, CellName = 'NA'):  
        self.PrintShow = PrintShow
        self.CellName = CellName
        self.SingleWaves = Waves
        self.Waves = [None] * len(self.SingleWaves)
        self.Values = Values
        self.WaveNames = [None] * len(self.SingleWaves)
        i = 0
        while i < len(Waves):
            self.Waves[i] = ddPlotting.ExtractPlotting(self.SingleWaves[i])
            self.WaveNames[i] = self.SingleWaves[i].Names
            i += 1
        self.NumWaves = len(self.SingleWaves)
        if self.NumWaves > 10:
            self.NumWaves = 10
        self.WavesPlot = len(self.Waves[0].Waves)
        self.WavesAnnot = len(self.Waves[0].Annot)

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
            
        self.AX = [None] * (self.NumWaves)
        self.gs = gridspec.GridSpec(5, 4)
        self.gs.update(left=0.05, bottom= 0.05, top = 0.95, right=0.95, wspace=0.2)
        self.Markersize = 5
        # Set Grid for Plotting: Row/Colum
        if self.NumWaves >= 1:
            self.AX[0] = plt.subplot(self.gs[0,0])
        if self.NumWaves >= 2:
            self.AX[1] = plt.subplot(self.gs[1,0])
        if self.NumWaves >= 3:
            self.AX[2] = plt.subplot(self.gs[2,0])
        if self.NumWaves >= 4:
            self.AX[3] = plt.subplot(self.gs[3,0])
        if self.NumWaves >= 5:
            self.AX[4] = plt.subplot(self.gs[4,0])
        if self.NumWaves >= 6:
            self.AX[5] = plt.subplot(self.gs[0,3])
        if self.NumWaves >= 7:
            self.AX[6] = plt.subplot(self.gs[1,3])
        if self.NumWaves >= 8:
            self.AX[7] = plt.subplot(self.gs[2,3])
        if self.NumWaves >= 9:
            self.AX[8] = plt.subplot(self.gs[3,3])
        if self.NumWaves >= 10:
            self.AX[9] = plt.subplot(self.gs[4,3])
                
        # Plot Waves:
        i = 0
        while i < self.NumWaves:
            
            j = 0
            while j < self.WavesPlot:
                
                # Set Markersize
                if self.Waves[i].Waves[j].get_marker() == '.' or self.Waves[i].Waves[j].get_marker() == 'o':
                    msset = self.Markersize
                else:
                    msset = 1.0
                    
                self.AX[i].plot(self.Waves[i].Waves[j].get_data()[0], self.Waves[i].Waves[j].get_data()[1],self.Waves[i].Waves[j].get_marker(), linestyle = self.Waves[i].Waves[j].get_linestyle(), color = self.Waves[i].Waves[j].get_color(), ms = msset)
                j +=1   
                
            # Setting Axes,...
            self.AX[i].set_xlim(0,800)
            self.AX[i].xaxis.set_ticks(np.arange(0,800,100))
            self.AX[i].set_ylabel('mV',fontsize = 6, rotation = 90)
            self.AX[i].spines["top"].set_visible(False)
            self.AX[i].spines["right"].set_visible(False)
            
            plt.setp(self.AX[i].get_yticklabels(),rotation = 'vertical', fontsize = 6)
            if i == 5 or i == self.NumWaves-1:
                #print('set')
                self.AX[i].set_xlabel('ms',fontsize = 6)
                plt.setp(self.AX[i].get_xticklabels(), fontsize = 6)
            else:
                #print('deleted')
                self.AX[i].set_xticklabels([])
                self.AX[i].set_xticks([], [])
                
            # Title:
            self.AX[i].set_title(self.WaveNames[i]+": %i " % i, fontsize = 6, y=0.9)
            
            i +=1    
             
        # Plot Annotations
        i = 0
        self.PlottedAnnot = []
        
        while i < self.NumWaves:
            self.PlottedAnnot.append([i])
            k = 0
            while k < self.WavesAnnot:
                self.PlottedAnnot[i].append(k)
                self.PlottedAnnot[i][k] = self.AX[i].annotate(self.Waves[i].Annot[k]._text, self.Waves[i].Annot[k].xy,self.Waves[i].Annot[k].xyann, self.Waves[i].Annot[k].xycoords, self.Waves[i].Annot[k]._textcoords, fontsize = 4,color = self.Waves[i].Annot[k].get_color())
                a = self.PlottedAnnot[i][k].get_position()
                if k == 0:
                    self.PlottedAnnot[i][k].set_position((a[0]/2,a[1]/3))
                if k == 1:
                    self.PlottedAnnot[i][k].set_position((a[0]/5,a[1]/4))
                if k == 2:
                    self.PlottedAnnot[i][k].set_position((a[0]/5,a[1]/5))
                if k == 3:
                    self.PlottedAnnot[i][k].set_position((a[0]/5,a[1]/4))
                if k == 4:
                    self.PlottedAnnot[i][k].set_position((a[0]/1,a[1]/3))
                if k == 5:
                    self.PlottedAnnot[i][k].set_position((a[0]/1,a[1]/3))
                if k == 6:
                    self.PlottedAnnot[i][k].set_position((a[0]/4,a[1]/2))
                if k == 7:
                    self.PlottedAnnot[i][k].set_position((a[0]/4,a[1]/2))
                k +=1   
            i+=1
            
        
# Plot Sag and Rebound:
        self.gs1 = gridspec.GridSpec(1, 3)
        self.gs1.update(left=0.35, bottom= 0.475, top = 0.85, right=0.65, wspace=0.5)
        self.gs2 = gridspec.GridSpec(1, 4)
        self.gs2.update(left=0.35, bottom= 0.05, top = 0.375, right=0.65, wspace=1)
        self.FS = 6
        self.FSTitle = 6
# Sag Index:
        self.axSagIndex = plt.subplot(self.gs1[0,0])
        self.XValuesSag = np.ones(len(self.Values.SagIndex.All))
        self.axSagIndex.plot(np.ones(len(self.Values.SagIndex.All)),self.Values.SagIndex.All,'o') 
        self.axSagIndex.errorbar(1,self.Values.SagIndex.Mean,self.Values.SagIndex.SD,linestyle='None', marker='o')
            # Annotate: 
        #self.FS = 8
        i = 0
        while i < len(self.Values.SagIndex.All):
            self.axSagIndex.annotate('%.0f ' % i,xy = (1,self.Values.SagIndex.All[i]),xytext=(-10,0),xycoords='data',textcoords='offset points', fontsize = 6)
            i +=1
        self.axSagIndex.annotate('%.1f %%' % self.Values.SagIndex.Mean,xy = (1,self.Values.SagIndex.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 6)
        self.axSagIndex.annotate('+/- %.1f' % self.Values.SagIndex.SD,xy = (1,self.Values.SagIndex.Mean),xytext=(5,-7.5),xycoords='data',textcoords='offset points', fontsize = 6)
        self.axSagIndex.set_title("Sag Index ", fontsize = self.FSTitle)
        
            # Set Axes:
        self.axSagIndex.set_ylabel('%',fontsize = self.FS, rotation = 90)
        
        self.SagIndexyLim = [None]*2
        if min(self.Values.SagIndex.All) < 0:
            self.SagIndexyLim[0] = ddhelp.rounddown(min(self.Values.SagIndex.All))
        else:
            self.SagIndexyLim[0] = 0
        if max(self.Values.SagIndex.All) < 0:   
            self.SagIndexyLim[1] = 0
        else:
            self.SagIndexyLim[1] = ddhelp.roundup(max(self.Values.SagIndex.All))
        
        self.axSagIndex.set_ylim ([self.SagIndexyLim[0],self.SagIndexyLim[1]])
        self.axSagIndex.set_xticklabels([])
        self.axSagIndex.set_xticks =[]
        plt.setp(self.axSagIndex.get_yticklabels(),rotation = 'vertical', fontsize = self.FS)
        self.axSagIndex.spines["top"].set_visible(False)
        self.axSagIndex.spines["right"].set_visible(False)
        
        # Title of the Whole plot:
        if self.SingleWaves[0].Baseline > self.SingleWaves[0].VStable:
            self.CellName2 = 'HypoProperties'
        elif self.SingleWaves[0].Baseline < self.SingleWaves[0].VStable:
            self.CellName2 = 'JustSubProperties'  
        self.OVTitle = self.CellName2 + '\nof ' + self.CellName +':'
        self.axSagIndex.set_title(self.OVTitle,fontsize = 14, y = 1.15,fontweight='bold',loc='left')
        
# Sag Area:
        if not np.all(np.isnan(self.Values.SagArea.All)):
            self.axSagArea = plt.subplot(self.gs1[0,1])
            self.XValuesSagArea = np.ones(len(self.Values.SagArea.All))
            self.axSagArea.plot(np.ones(len(self.Values.SagArea.All)),self.Values.SagArea.All,'o') 
            self.axSagArea.errorbar(1,self.Values.SagArea.Mean,self.Values.SagArea.SD,linestyle='None', marker='o')
                # Annotate: 
            #self.FS = 8
            i = 0
            while i < len(self.Values.SagArea.All):
                self.axSagArea.annotate('%.0f ' % i,xy = (1,self.Values.SagArea.All[i]),xytext=(-10,0),xycoords='data',textcoords='offset points', fontsize = 6)
                i +=1
            self.axSagArea.annotate('%.1f mV$^{2}$/ms' % self.Values.SagArea.Mean,xy = (1,self.Values.SagArea.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 6)
            self.axSagArea.annotate('+/- %.1f' % self.Values.SagArea.SD,xy = (1,self.Values.SagArea.Mean),xytext=(5,-7.5),xycoords='data',textcoords='offset points', fontsize = 6)
            self.axSagArea.set_title("Sag Area ", fontsize = self.FSTitle)
            
                # Set Axes:
            self.axSagArea.set_ylabel('mV$^{2}$/ms',fontsize = self.FS, rotation = 90)
            
            self.SagAreayLim = [None]*2
            if np.nanmin(self.Values.SagArea.All) < 0:
                self.SagAreayLim[0] = ddhelp.rounddown(np.nanmin(self.Values.SagArea.All))
            else:
                self.SagAreayLim[0] = 0                
            if np.nanmax(self.Values.SagArea.All) < 0:
                self.SagAreayLim[1] = ddhelp.rounddown(np.nanmax(self.Values.SagArea.All))
            else:
                self.SagAreayLim[1] = ddhelp.roundup(np.nanmax(self.Values.SagArea.All))
            
            
            self.axSagArea.set_ylim ([self.SagAreayLim[0],self.SagAreayLim[1]])
            self.axSagArea.set_xticklabels([])
            self.axSagArea.set_xticks =[]
            plt.setp(self.axSagArea.get_yticklabels(),rotation = 'vertical', fontsize = self.FS)
            self.axSagArea.spines["top"].set_visible(False)
            self.axSagArea.spines["right"].set_visible(False)

# Sag TauOff:
        if not np.all(np.isnan(self.Values.SagTauOff.All)):
            self.axSagTauOff = plt.subplot(self.gs1[0,2])
            self.XValuesSagTauOff = np.ones(len(self.Values.SagTauOff.All))
            self.axSagTauOff.plot(np.ones(len(self.Values.SagTauOff.All)),self.Values.SagTauOff.All,'o') 
            self.axSagTauOff.errorbar(1,self.Values.SagTauOff.Mean,self.Values.SagTauOff.SD,linestyle='None', marker='o')
            #self.axSagIndex.plot(1,self.Values.SagIndex.Mean,'or')
            #self.axSagIndex.boxplot(self.Values.SagIndex.All, 1)
                # Annotate: 
            #self.FS = 8
            i = 0
            while i < len(self.Values.SagTauOff.All):
                self.axSagTauOff.annotate('%.0f ' % i,xy = (1,self.Values.SagTauOff.All[i]),xytext=(-10,0),xycoords='data',textcoords='offset points', fontsize = 6)
                i +=1
            self.axSagTauOff.annotate('%.1f ms' % self.Values.SagTauOff.Mean,xy = (1,self.Values.SagTauOff.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 6)
            self.axSagTauOff.annotate('+/- %.1f' % self.Values.SagTauOff.SD,xy = (1,self.Values.SagTauOff.Mean),xytext=(5,-7.5),xycoords='data',textcoords='offset points', fontsize = 6)
            self.axSagTauOff.set_title("Sag Tau Off ", fontsize = self.FSTitle)
            
                # Set Axes:
            self.axSagTauOff.set_ylabel('ms',fontsize = self.FS, rotation = 90)
            
            self.SagTauyLim = [None]*2
            
            if min(self.Values.SagTauOff.All) < 0:
                self.SagTauyLim[0] = ddhelp.rounddown(np.nanmin(self.Values.SagTauOff.All))
            else:
                self.SagTauyLim[0] = 0      
                
            if max(self.Values.SagTauOff.All) < 0:
                self.SagTauyLim[1] = ddhelp.rounddown(np.nanmax(self.Values.SagTauOff.All))
            else:
                self.SagTauyLim[1] = ddhelp.roundup(np.nanmax(self.Values.SagTauOff.All))
                
            self.axSagTauOff.set_ylim ([self.SagTauyLim[0],self.SagTauyLim[1]])
            
            self.axSagTauOff.set_xticklabels([])
            self.axSagTauOff.set_xticks =[]
            plt.setp(self.axSagTauOff.get_yticklabels(),rotation = 'vertical', fontsize = self.FS)
            self.axSagTauOff.spines["top"].set_visible(False)
            self.axSagTauOff.spines["right"].set_visible(False)

# Rebound Index:
        self.axRebIndex = plt.subplot(self.gs2[0,0])
        self.XValuesRebound = np.ones(len(self.Values.ReboundIndex.All))
        self.axRebIndex.plot(np.ones(len(self.Values.ReboundIndex.All)),self.Values.ReboundIndex.All,'o') 
        self.axRebIndex.errorbar(1,self.Values.ReboundIndex.Mean,self.Values.ReboundIndex.SD,linestyle='None', marker='o')
            # Annotate: 
        #self.FS = 8
        i = 0
        while i < len(self.Values.ReboundIndex.All):
            self.axRebIndex.annotate('%.0f ' % i,xy = (1,self.Values.ReboundIndex.All[i]),xytext=(-10,0),xycoords='data',textcoords='offset points', fontsize = 6)
            i +=1
        self.axRebIndex.annotate('%.1f %%' % self.Values.ReboundIndex.Mean,xy = (1,self.Values.ReboundIndex.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 6)
        self.axRebIndex.annotate('+/- %.1f' % self.Values.ReboundIndex.SD,xy = (1,self.Values.ReboundIndex.Mean),xytext=(5,-7.5),xycoords='data',textcoords='offset points', fontsize = 6)
        self.axRebIndex.set_title("Rebound Index ", fontsize = self.FSTitle)
        
            # Set Axes:
        self.axRebIndex.set_ylabel('%',fontsize = self.FS, rotation = 90)
        
        self.RebIndexyLim = [None]*2
        if min(self.Values.ReboundIndex.All) < 0:
            self.RebIndexyLim[0] = ddhelp.rounddown(min(self.Values.ReboundIndex.All))
        else:
            self.RebIndexyLim[0] = 0
            
        if max(self.Values.ReboundIndex.All) < 0:   
            self.RebIndexyLim[1] = 0
        else:
            self.RebIndexyLim[1] = ddhelp.roundup(max(self.Values.ReboundIndex.All))
        
        self.axRebIndex.set_ylim ([self.RebIndexyLim[0],self.RebIndexyLim[1]])
        self.axRebIndex.set_xticklabels([])
        self.axRebIndex.set_xticks =[]
        plt.setp(self.axRebIndex.get_yticklabels(),rotation = 'vertical', fontsize = self.FS)
        self.axRebIndex.spines["top"].set_visible(False)
        self.axRebIndex.spines["right"].set_visible(False)
        
# Rebound Area:
        if not np.all(np.isnan(self.Values.ReboundArea.All)):
            self.axRebArea = plt.subplot(self.gs2[0,1])
            self.XValuesSagArea = np.ones(len(self.Values.ReboundArea.All))
            self.axRebArea.plot(np.ones(len(self.Values.ReboundArea.All)),self.Values.ReboundArea.All,'o') 
            self.axRebArea.errorbar(1,self.Values.ReboundArea.Mean,self.Values.ReboundArea.SD,linestyle='None', marker='o')
                # Annotate: 
            #self.FS = 8
            i = 0
            while i < len(self.Values.ReboundArea.All):
                self.axRebArea.annotate('%.0f ' % i,xy = (1,self.Values.ReboundArea.All[i]),xytext=(-10,0),xycoords='data',textcoords='offset points', fontsize = 6)
                i +=1
            self.axRebArea.annotate('%.1f mV$^{2}$/ms' % self.Values.ReboundArea.Mean,xy = (1,self.Values.ReboundArea.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 6)
            self.axRebArea.annotate('+/- %.1f' % self.Values.ReboundArea.SD,xy = (1,self.Values.ReboundArea.Mean),xytext=(5,-7.5),xycoords='data',textcoords='offset points', fontsize = 6)
            self.axRebArea.set_title("Rebound Area ", fontsize = self.FSTitle)
            
                # Set Axes:
            self.axRebArea.set_ylabel('mV$^{2}$/ms',fontsize = self.FS, rotation = 90)
            
            self.RebAreayLim = [None]*2
            if min(self.Values.ReboundArea.All) < 0:
                self.RebAreayLim[0] = ddhelp.rounddown(np.nanmin(self.Values.ReboundArea.All))
            else:
                self.RebAreayLim[0] = 0                
            if max(self.Values.ReboundArea.All) < 0:
                self.RebAreayLim[1] = ddhelp.rounddown(np.nanmax(self.Values.ReboundArea.All))
            else:
                self.RebAreayLim[1] = ddhelp.roundup(np.nanmax(self.Values.ReboundArea.All))
            
            
            self.axRebArea.set_ylim ([self.RebAreayLim[0],self.RebAreayLim[1]])
            self.axRebArea.set_xticklabels([])
            self.axRebArea.set_xticks =[]
            plt.setp(self.axRebArea.get_yticklabels(),rotation = 'vertical', fontsize = self.FS)
            self.axRebArea.spines["top"].set_visible(False)
            self.axRebArea.spines["right"].set_visible(False)
            
# Rebound TauOff:
        if not np.all(np.isnan(self.Values.ReboundTauOff.All)):
            self.axRebTauOff = plt.subplot(self.gs2[0,2])
            self.XValuesSagTauOff = np.ones(len(self.Values.ReboundTauOff.All))
            self.axRebTauOff.plot(np.ones(len(self.Values.ReboundTauOff.All)),self.Values.ReboundTauOff.All,'o') 
            self.axRebTauOff.errorbar(1,self.Values.ReboundTauOff.Mean,self.Values.ReboundTauOff.SD,linestyle='None', marker='o')
            #self.axSagIndex.plot(1,self.Values.SagIndex.Mean,'or')
            #self.axSagIndex.boxplot(self.Values.SagIndex.All, 1)
                # Annotate: 
            #self.FS = 8
            i = 0
            while i < len(self.Values.ReboundTauOff.All):
                self.axRebTauOff.annotate('%.0f ' % i,xy = (1,self.Values.ReboundTauOff.All[i]),xytext=(-10,0),xycoords='data',textcoords='offset points', fontsize = 6)
                i +=1
            self.axRebTauOff.annotate('%.1f ms' % self.Values.ReboundTauOff.Mean,xy = (1,self.Values.ReboundTauOff.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 6)
            self.axRebTauOff.annotate('+/- %.1f' % self.Values.ReboundTauOff.SD,xy = (1,self.Values.ReboundTauOff.Mean),xytext=(5,-7.5),xycoords='data',textcoords='offset points', fontsize = 6)
            self.axRebTauOff.set_title("Sag Tau Off ", fontsize = self.FSTitle)
            
                # Set Axes:
            self.axRebTauOff.set_ylabel('ms',fontsize = self.FS, rotation = 90)
            
            self.RebTauyLim = [None]*2
            
            if min(self.Values.ReboundTauOff.All) < 0:
                self.RebTauyLim[0] = ddhelp.rounddown(np.nanmin(self.Values.ReboundTauOff.All))
            else:
                self.RebTauyLim[0] = 0      
                
            if max(self.Values.ReboundTauOff.All) < 0:
                self.RebTauyLim[1] = ddhelp.rounddown(np.nanmax(self.Values.ReboundTauOff.All))
            else:
                self.RebTauyLim[1] = ddhelp.roundup(np.nanmax(self.Values.ReboundTauOff.All))
                
            self.axRebTauOff.set_ylim ([self.RebTauyLim[0],self.RebTauyLim[1]])
            
            self.axRebTauOff.set_xticklabels([])
            self.axRebTauOff.set_xticks =[]
            plt.setp(self.axRebTauOff.get_yticklabels(),rotation = 'vertical', fontsize = self.FS)
            self.axRebTauOff.spines["top"].set_visible(False)
            self.axRebTauOff.spines["right"].set_visible(False)
            
        # Rebound TauOff:
        if not np.all(np.isnan(self.Values.ReboundAPs.All)):
            self.axReboundAPs = plt.subplot(self.gs2[0,3])
            self.XValuesReboundAPs = np.ones(len(self.Values.ReboundAPs.All))
            self.axReboundAPs.plot(np.ones(len(self.Values.ReboundAPs.All)),self.Values.ReboundAPs.All,'o') 
            self.axReboundAPs.errorbar(1,self.Values.ReboundAPs.Mean,self.Values.ReboundAPs.SD,linestyle='None', marker='o')
            #self.FS = 8
            i = 0
            while i < len(self.Values.ReboundAPs.All):
                self.axReboundAPs.annotate('%.0f ' % i,xy = (1,self.Values.ReboundAPs.All[i]),xytext=(-10,0),xycoords='data',textcoords='offset points', fontsize = 6)
                i +=1
            self.axReboundAPs.annotate('%.1f ms' % self.Values.ReboundAPs.Mean,xy = (1,self.Values.ReboundAPs.Mean),xytext=(5,0),xycoords='data',textcoords='offset points', fontsize = 6)
            self.axReboundAPs.annotate('+/- %.1f' % self.Values.ReboundAPs.SD,xy = (1,self.Values.ReboundAPs.Mean),xytext=(5,-7.5),xycoords='data',textcoords='offset points', fontsize = 6)
            self.axReboundAPs.set_title("Rebound APs ", fontsize = self.FSTitle)
            
                # Set Axes:
            self.axReboundAPs.set_ylabel('Number',fontsize = self.FS, rotation = 90)
            
            self.ReboundAPsyLim = [None]*2
            
            if min(self.Values.ReboundAPs.All) < 0:
                self.ReboundAPsyLim[0] = ddhelp.rounddown(np.nanmin(self.Values.ReboundAPs.All))
            else:
                self.ReboundAPsyLim[0] = 0      
                
            if max(self.Values.ReboundAPs.All) < 0:
                self.ReboundAPsyLim[1] = ddhelp.rounddown(np.nanmax(self.Values.ReboundAPs.All))
            else:
                self.ReboundAPsyLim[1] = ddhelp.roundup(np.nanmax(self.Values.ReboundAPs.All))
                
            self.axReboundAPs.set_ylim ([self.ReboundAPsyLim[0],self.ReboundAPsyLim[1]])
            
            self.axReboundAPs.set_xticklabels([])
            self.axReboundAPs.set_xticks =[]
            plt.setp(self.axReboundAPs.get_yticklabels(),rotation = 'vertical', fontsize = self.FS)
            self.axReboundAPs.spines["top"].set_visible(False)
            self.axReboundAPs.spines["right"].set_visible(False)
            
        #Saving:
        if self.PrintShow == 1:
            if self.SingleWaves[0].Baseline > self.SingleWaves[0].VStable:
                self.CellName3 = '_HypoProperties'
            elif self.SingleWaves[0].Baseline < self.SingleWaves[0].VStable:
                self.CellName3 = '_JustSubProperties'  
            self.SavingName = self.CellName+self.CellName3
            ddPlotting.save(self.SavingName, ext="png", close=True, verbose=True)
            #ddPlotting.save(self.SavingName, ext="svg", close=True, verbose=True)
            plt.close('All')
            plt.ion()

''' For Testing: '''        
#A = Main(Time,Wave,Stimulus,SampFreq)
 

