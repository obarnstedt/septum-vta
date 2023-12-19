#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Different Functions for Analysis
"""

''' Importing scripts '''

#import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#import scipy.signal as spsignal
import numpy as np
import math
import importlib
from collections import Counter
import pandas as pd
#from matplotlib import pyplot as plt
#import warnings

## Suppress warnings
#def SupressWarning():
#    warnings.warn("deprecated", DeprecationWarning)
#
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    SupressWarning()

class FitMonoEx:
    def f(x,a,b,c):
        return (a*np.exp(b*x))+c
    
    def __init__(self,Time,Wave,BaselinePositions,StartPointFit):
        self.Baseline = BaselinePositions
        self.Time = Time
        self.Wave = Wave
        self.StartPointFit = StartPointFit
        self.WaveZeroed = np.zeros(len(Wave))
        self.FitParams = np.zeros(3)
        self.FitCoev = np.zeros((3,3))
        self.FitCurveZero = np.zeros(len(Wave))
        self.FitCurve = np.zeros(len(Wave))
        self.r = 0.
        self.SST = 0
        self.SSR = 0
        
        ''' Get Wave to zero with Baseline: '''
        if len(self.Baseline) >=2:
            BaselineMean = np.mean(self.Wave[int(self.Baseline[0]):int(self.Baseline[1])])
        else:
            BaselineMean = self.Wave[int(self.Baseline[0])]
        self.WaveZeroed = self.Wave - BaselineMean
        self.WaveZeroed[self.WaveZeroed < 0] = 0
        
        ''' Get Rid of NaNs '''
        self.WaveZeroed = np.nan_to_num(self.WaveZeroed)      
        
        ''' Try Fitting '''         
        try:
            self.FitParams, self.FitCov = curve_fit(FitMonoEx.f,self.Time,self.WaveZeroed,p0=self.StartPointFit)   
        
            self.FitCurveZero = FitMonoEx.f(self.Time,self.FitParams[0],self.FitParams[1],self.FitParams[2])
            self.SST = sum((self.WaveZeroed-np.mean(self.Wave))**2)
            self.SSR = sum((self.WaveZeroed - FitMonoEx.f(self.Time, self.FitParams[0],self.FitParams[1],self.FitParams[2]))**2)
            self.r = np.sqrt(1-(self.SSR/self.SST))
           
            if np.isnan(self.r):
                self.r = 0
        except (RuntimeError, RuntimeWarning):
            pass 

        ''' FitCurve to Original Value '''
        self.FitCurve = self.FitCurveZero + BaselineMean
    
class FitMonoEx2:
    def f(x,a,b):
        return (np.exp(a*x))+b
    
    def __init__(self,Time,Wave,BaselinePositions,StartPointFit):
        self.Baseline = BaselinePositions
        self.Time = Time
        self.Wave = Wave
        self.StartPointFit = StartPointFit
        self.WaveZeroed = np.zeros(len(Wave))
        self.FitParams = np.zeros(3)
        self.FitCoev = np.zeros((3,3))
        self.FitCurveZero = np.zeros(len(Wave))
        self.FitCurve = np.zeros(len(Wave))
        self.r = 0.
        self.SST = 0
        self.SSR = 0
        
        ''' Weight'''
        # Create Weight for Fit:
        self.WeightFit = np.ones(len(self.Wave))
        self.WeightFit[2] = 4
        self.WeightFit[1] = 4
        self.WeightFit[0] = 4
        self.WeightFit[-1] = 1
#        i = 0
#        while i < len(self.WeightFit):
#            self.WeightFit[i] = self.WeightFit[i]/16
#            i +=1
##        print(self.WeightFit)
        
        ''' Get Wave to zero with Baseline '''
        if len(self.Baseline) >=2:
            BaselineMean = np.mean(self.Wave[int(self.Baseline[0]):int(self.Baseline[1])])
        else:
            BaselineMean = self.Wave[int(self.Baseline[0])]
        self.WaveZeroed = self.Wave - BaselineMean
        self.WaveZeroed[self.WaveZeroed < 0] = 0
        
        ''' Get Rid of NaNs '''
        self.WaveZeroed = np.nan_to_num(self.WaveZeroed)      
        ''' Try Fitting '''         
        try:
            self.FitParams, self.FitCov = curve_fit(FitMonoEx2.f,self.Time,self.WaveZeroed,p0=self.StartPointFit,sigma=self.WeightFit, absolute_sigma=True)   
        
            self.FitCurveZero = FitMonoEx2.f(self.Time,self.FitParams[0],self.FitParams[1])
            self.SST = sum((self.WaveZeroed-np.mean(self.Wave))**2)
            self.SSR = sum((self.WaveZeroed - FitMonoEx2.f(self.Time, self.FitParams[0],self.FitParams[1]))**2)
            self.r = np.sqrt(1-(self.SSR/self.SST))
           
            if np.isnan(self.r):
                self.r = 0
        except (RuntimeError, RuntimeWarning):
            pass 

        ''' FitCurve to Original Value '''
        self.FitCurve = self.FitCurveZero #+ BaselineMean

class FitMonoExLinearised:

    def __init__(self,Time,Wave):
        self.Time = Time
        self.Wave = Wave
        # Set Wave to allmost Zero: 
        self.MinAllmostWave = np.min(self.Wave)-1
        self.WaveZeroed = self.Wave -self.MinAllmostWave
        # Linearise with log:
        self.WaveToFit = np.log(self.WaveZeroed)-1
        # Create Weight for Fit:
#        self.WeightFit = np.flip(np.arange(1,len(self.Wave)+1,1),0)
        self.WeightFit = np.ones(len(self.Wave))
        self.WeightFit[0] = 4
        self.WeightFit[-1] = 4
        if len(self.WeightFit) > 2:
            self.WeightFit[1] =  2
        if len(self.WeightFit) > 3:
            self.WeightFit[1] = 2
            
        i = 0
        while i < len(self.WeightFit):
            self.WeightFit[i] = self.WeightFit[i]/len(self.Wave)*8
            i +=1
        ''' Try Fitting '''         
        try:
            self.FitParams = np.polyfit(self.Time,self.WaveToFit,1, w = self.WeightFit)
            # Create FitCurve:
            self.FitCurvePrePre = np.polyval(self.FitParams,self.Time)+1
            self.FitCurvePre = np.exp(self.FitCurvePrePre)
            self.FitCurve = self.FitCurvePre + self.MinAllmostWave
            
            # Calculate r:
            self.r_squared = 1 - (sum((self.WaveToFit - (self.FitParams[0] * self.Time + self.FitParams[1]))**2) / ((len(self.WaveToFit) - 1) * np.var(self.WaveToFit, ddof=1)))
            self.r = np.sqrt(self.r_squared)        
#            p = np.poly1d(self.FitParams)
#            yhat = p(self.Time)                         # or [p(z) for z in x]
#            ybar = np.sum(self.WaveToFit)/len(self.WaveToFit)          # or sum(y)/len(y)
#            ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
#            sstot = np.sum((self.WaveToFit - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
#            self.r =  ssreg/sstot
#            print(self.r)
            if np.isnan(self.r):
                self.r = 0
        except (RuntimeError, RuntimeWarning):
            pass 

class FitDoubleEx:
    def f(x,a,b,c,d):
        return (a*np.exp(b*x))+c*np.exp(d*x)
    
    def __init__(self,Time,Wave,BaselinePositions,StartPointFit):
        self.Baseline = BaselinePositions
        self.Time = Time
        self.Wave = Wave
        self.StartPointFit = StartPointFit
        self.WaveZeroed = np.zeros(len(Wave))
        self.FitParams = np.zeros(3)
        self.FitCoev = np.zeros((3,3))
        self.FitCurveZero = np.zeros(len(Wave))
        self.FitCurve = np.zeros(len(Wave))
        self.r = 0.
        self.SST = 0
        self.SSR = 0
        
        ''' Get Wave to zero with Baseline: '''
        if len(self.Baseline) >=2:
            BaselineMean = np.mean(self.Wave[int(self.Baseline[0]):int(self.Baseline[1])])
        else:
            BaselineMean = self.Wave[int(self.Baseline[0])]
#        self.WaveZeroed = self.Wave 
        self.WaveZeroed = self.Wave - BaselineMean
#        self.WaveZeroed[self.WaveZeroed < 0] = 0
        
        ''' Get Rid of NaNs '''
        self.WaveZeroed = np.nan_to_num(self.WaveZeroed)      
        
        ''' Try Fitting '''         
        try:
            self.FitParams, self.FitCov = curve_fit(FitDoubleEx.f,self.Time,self.WaveZeroed,p0=self.StartPointFit)   
        
            self.FitCurveZero = FitDoubleEx.f(self.Time,self.FitParams[0],self.FitParams[1],self.FitParams[2],self.FitParams[3])
            self.SST = sum((self.WaveZeroed-np.mean(self.Wave))**2)
            self.SSR = sum((self.WaveZeroed - FitDoubleEx.f(self.Time, self.FitParams[0],self.FitParams[1],self.FitParams[2],self.FitParams[3]))**2)
            self.r = np.sqrt(1-(self.SSR/self.SST))
           
            if np.isnan(self.r):
                self.r = 0
        except (RuntimeError, RuntimeWarning):
            pass 

        ''' FitCurve to Original Value '''
        self.FitCurve = self.FitCurveZero + BaselineMean
            
class FitPSP:
    def f(x,a,b,c,d):
        return a*((1-(np.exp((-(x-d))/b)))**5)*(np.exp((-(x-d))/c))
    
    def __init__(self,Time,Wave,BaselinePositions,StartPointFit,LowerBounds=None,UpperBounds=None):
        self.Baseline = BaselinePositions
        self.Time = Time
        self.SamppingFreq = 1/(self.Time[1]-self.Time[0])
        self.Wave = Wave
        self.StartPointFit = StartPointFit
        self.WaveZeroed = np.zeros(len(Wave))
        self.FitParams = np.zeros(4)
        self.FitCoev = np.zeros((4,4))
        self.FitCurveZero = np.zeros(len(Wave))
        self.FitCurve = np.zeros(len(Wave))
        self.r = 0.
        self.SST = 0
        self.SSR = 0
        
        self.LowerBounds = LowerBounds 
        self.UpperBounds = UpperBounds
        
        ''' Get Wave to zero with Baseline: '''
        BaselineMean = np.mean(self.Wave[int(self.Baseline[0]):int(self.Baseline[1])])
        self.WaveZeroed = self.Wave - BaselineMean
        self.WaveZeroed[self.WaveZeroed < 0] = 0
#        self.WaveZeroed = self.WaveZeroed+np.random.normal(0,0.2,len(self.WaveZeroed))
        
        ''' Get Rid of NaNs '''
        self.WaveZeroed = np.nan_to_num(self.WaveZeroed)             
        
        ''' Try Fitting '''
        try:
            if self.LowerBounds is not None:
                self.FitParams, self.FitCov = curve_fit(FitPSP.f,self.Time,self.WaveZeroed,p0=self.StartPointFit, bounds = [self.LowerBounds,self.UpperBounds]) 
            else:
                self.FitParams, self.FitCov = curve_fit(FitPSP.f,self.Time,self.WaveZeroed,p0=self.StartPointFit)       
            self.FitCurveZero = FitPSP.f(self.Time,self.FitParams[0],self.FitParams[1],self.FitParams[2],self.FitParams[3])
            self.SST = sum((self.WaveZeroed-np.mean(self.Wave))**2)
            self.SSR = sum((self.WaveZeroed - FitPSP.f(self.Time, self.FitParams[0],self.FitParams[1],self.FitParams[2],self.FitParams[3]))**2)
            self.r = np.sqrt(1-(self.SSR/self.SST))
           
            if np.isnan(self.r):
                self.r = 0
        except (RuntimeError,RuntimeWarning):
            pass
        
        ''' FitCurve to Original Value '''
        self.FitCurve = self.FitCurveZero + BaselineMean

#        ''' Figure: '''
#        self.fig = plt.figure()
#        plt.plot(self.Time,self.Wave)
#        plt.plot(self.Time,self.FitCurve)
#        plt.close();
        

class Filters:
    def MovingAverage (x,window_len=11,window='hanning'):
        
        s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        if window == 'flat': #moving averag
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')
            y=np.convolve(w/w.sum(),s,mode='valid')
            
        return y[int(window_len/2-1):-int(window_len/2)]

class Deri:
    def SlowPeak(Time, Wave, SampFreq):
        WaveFilt = Filters.MovingAverage(Wave,window_len=int(50*SampFreq),window='hanning')
        WaveDiff = np.diff(WaveFilt)
        return WaveFilt, WaveDiff

    def AHP(Time, Wave, SampFreq):
        WaveFilt = Filters.MovingAverage(Wave,window_len=int(10),window='hanning')
        WaveDiff = np.diff(WaveFilt)
        return WaveFilt, WaveDiff
        
class Extract:
    ''' Extract Values from List of Classes!!! '''
    def __init__ (self,Class,Name):

        self.Class = Class
        self.ClassLength = len(self.Class)
        self.Name = Name
        
        i = 0
        self.VariableList = [None]*self.ClassLength
        while i < self.ClassLength:
            self.VariableList[i] = getattr(self.Class[i], self.Name)
            if self.VariableList[i] is None:
                self.VariableList[i] = np.nan    
            i +=1
     
        self.All = np.asarray(self.VariableList)
#        self.ArrayColumns = self.All.shape[1]
#        if self.ArrayColumns > 1
#            self.All = self.All[:,0]
        
        if np.any(self.All):
            self.Mean = np.nanmean(self.All)
            self.SD = np.nanstd(self.All)
        else:
            self.Mean = np.mean(self.All)
            self.SD = np.std(self.All)
            
        self.Differences = self.All - self.Mean
        self.TraceToTake = np.argmin(self.Differences)
            
            
def roundup(x):
    if not np.isnan(x):
        output = int(math.ceil(x / 10.0)) * 10 
    else:
        output = 0
    return output

def rounddown(x):
    return int(math.ceil(np.absolute(x) / 10.0)) * -10  

def Indexing(bigValue,smallValue):
    output = []
    output = (np.absolute(smallValue)-np.absolute(bigValue))/np.absolute(bigValue)  
    return output 

def SearchKeyDic(Dict,keyPart):
    output = None
    for key in Dict:
        if keyPart in key:
            output = Dict.get(key)    
    return output
def SearchCompleteKeyDic (Dict,FullKey):
    output = None
    for key in Dict:
        if FullKey == key:
            output = Dict.get(key)    
    return output
    

def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c   


def ListofSubAttributes(Object,SearchAttribute):
    MemberList = [attr for attr in dir(Object) if not callable(getattr(Object, attr)) and not attr.startswith("__")]  
    ValueList = [None]*len(MemberList)
    i = 0
    while i < len(MemberList):
        if hasattr(getattr(Object,MemberList[i]),SearchAttribute):
            #print(getattr(getattr(Object,MemberList[i]),SearchAttribute))
            ValueList[i] = getattr(getattr(Object,MemberList[i]),SearchAttribute)
        i += 1
    ValueListClean = [x for x in ValueList if x != None]
    return ValueListClean

def CountListAndSort(List):
    a = Counter(List)    
    NPArray = np.array(list(a.items()))
    SortedList =NPArray[NPArray[:, 1].argsort()]
    SortedIndicies = np.flip(SortedList[:,0],0)
    return SortedIndicies

def SortBySD(ValuesPdTable):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    ValuesPdTable1 = ValuesPdTable.select_dtypes(include=numerics) 
#    ValuesPdTable1 = np.absolute(ValuesPdTable1)
    TableSubStractedMean = np.absolute(ValuesPdTable1 - ValuesPdTable1.mean(axis=0))
    Mins = TableSubStractedMean.idxmin(axis=0, skipna=True)
    Counting = Mins.value_counts()
    SortedTraces = Counting.index.values
    SortedTraces = SortedTraces.astype(int)
    return SortedTraces

def most_common(lst):
    return max(set(lst), key=lst.count)

def SaveAsExcelFile(File,Name):
    # File as Variable
    # Name as string wihout ending!
    Writer = pd.ExcelWriter(Name+'.xlsx', engine='xlsxwriter')
    File.to_excel(Writer, sheet_name='Sheet1',na_rep = "NaN",merge_cells=False)
    Writer.save()
    return print(Name +'Saved as ExcelFile')

class UpdateAllTable:
    def __init__ (self, NewEntry, ExistingTable):
        self.NewEntry = NewEntry
        self.ExistingTable = ExistingTable
        self.NumNewColumns = self.NewEntry.shape[1]
        self.NumOldColumns = self.ExistingTable.shape[1]
        
        if self.NumNewColumns != self.NumOldColumns:
            return print('New and old Table do not have the same length')
        self.ListColumnNames = list(self.ExistingTable)
        
        self.NewName = self.NewEntry.index
        self.OldNames = self.ExistingTable.index 
        
        if self.NewName.isin(self.OldNames):
            self.LineToDrop = self.NewName[0]
            self.ExistingTable = self.ExistingTable[self.ExistingTable.index  != self.LineToDrop]

        # Convert Index to Numbers:
        self.ExistingTable.reset_index(level=0, inplace=True)
        self.NewEntry.reset_index(level=0, inplace=True)
        # New Entry to Dict:
        self.NewEntryDic1 = self.NewEntry.to_dict(orient='index') 
        self.NewEntryDic = self.NewEntryDic1[0]
        # Append Dic:
        self.NewTable = self.ExistingTable.append(self.NewEntryDic,ignore_index=True)
        # Back: Index to CellName
        self.NewTable.set_index('index', inplace=True)
        # Sorting:
        self.NewTable.sort_index()

def get_num(x):
    return int(''.join(ele for ele in x if ele.isdigit()))        


        

        
        
        
        
        
        
        
        
        
        
        
        
        