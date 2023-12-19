import os
import sys
sys.path.append('path')
import DDImport
import Kalman_Filt_v as Kalman
from matplotlib import pyplot as plt 
import numpy as np
from scipy import signal as sig
import pandas as pd
import seaborn as sns

#DOWNSAMPLING

def downsample(PTimeVec,SWaves):

    #set time from ms to s
    
    PTimeVec_np = np.array(PTimeVec)
    PTimeVec_sec= PTimeVec_np/1000
    TimeList = []
    for i in range(len(PTimeVec_sec)):
        PTimeVec_sliced = PTimeVec_sec[i][slice(1,600000,10)]
        TimeList.append(PTimeVec_sliced)
        Time = np.array(TimeList)

    #downsample stim from 10 to 1kHz
    
    Stim_np = np.array(SWaves)
    Stim_sec= Stim_np/1
    StimList = []
    for i in range(len(Stim_sec)):
        Stim_sliced = Stim_sec[i][slice(1,600000,10)]
        StimList.append(Stim_sliced)
        Stim = np.array(StimList)
    
    return (Time, Stim)

#CORRECT POSITION
    
def position(PWaves, PWaves_scaled, STimeVec, plot=True):
    
    num_igor_S = len(PWaves)
    PWaves_scaled_np = np.array(PWaves_scaled)
    
    corrected_position_list = []
    for i in range(num_igor_S):
        print(i)
        
        posdiff = np.diff(PWaves_scaled[i])
        
        #define beginning of each lap (drop bigger than 8)
        inds=np.where(posdiff < -8)
        inds2=inds[0] #get positions as numpy array from tuple
        
        #if the mouse does do an entire lap in the sweep
        if len(inds2) == 0 :
            corrected_position_list.append(PWaves_scaled[i])
        
        #when there is at least a lap
        else:
            
            inds2=np.insert(inds2,0,0)
            inds2=np.append(inds2,len(PWaves_scaled_np[0,:]))
       
            vecList = []
            for j in range(len(inds2)-1):
                vec = PWaves_scaled_np[i,inds2[j]:inds2[j+1]+1]
                vecList.append(vec)
    
            corrected_vector = []
            corrected_vector.append(vecList[0])
            correction_factor=vecList[0][-1]
    
            for c in range(0,len(vecList)-1):
                vecCorrected = (vecList[c+1][1:]-vecList[c+1][1])+correction_factor
                corrected_vector.append(vecCorrected)
                posvec_corrected = np.concatenate(corrected_vector)
                correction_factor=vecCorrected[-1]
    
            corrected_position_list.append(posvec_corrected)
            
    if plot:
        for i in range (len(PWaves_scaled)):
            FigAx2 = plt.figure()
            FigAx2 = plt.subplot(1,1,1)
            FigAx2.plot(STimeVec[i],corrected_position_list[i])
            
    return corrected_position_list

#SPEED CALCULATION
    
def speed(corrected_position_list, FolderName, plot=True):
    
    velocityList = []
    for i in range(len(corrected_position_list)):
        posvec_downsampled = sig.decimate(corrected_position_list[i],10)
        posvec_filtered = Kalman.Kalman_Filt_v(posvec_downsampled,1e-3)
        velocityList.append(posvec_filtered[1])
    velocityArray = np.asarray(velocityList)
    
    if plot:
        for i in range (len(velocityArray)):
            FigAx1 = plt.subplot(1,1,1)
            FigAx1.plot(velocityArray[i])
            FigAx1.axvspan(20000,40000,facecolor='b', alpha=0.05) #blue shadow below stim time
            FigAx1.set_xlabel('Time (s)')
            FigAx1.set_ylabel('Speed (cm/s)')
            FigAx1.set_title(FolderName)
            FigAx1.set_xlim([1, 60000]) #fix x axis range
            FigAx1.set_ylim([-10, 40]) #fix y axis range
        
    return velocityArray

#Opto stimulation frequency, length and pulse width

def Stim_frequency (df):
    
    pulse_on = df.loc[df.StimON == 1, ['Time']].head(2) #identifies time of first 2 pulses onsets
    pulse_off = df.loc[df.StimON == -1,['Time']].head(1) #identifies time og first pulse offset
    pulse_last = df.loc[df.StimON == -1,['Time']].tail(1) #identifies time of last pulse offset
    pulses = pd.concat([pulse_on, pulse_off, pulse_last]) #merges all time values
    pulse_np = pulses.to_numpy() #converts into array (.values for version 23, .to_numpy for version 24)
    pulse_width = pulse_np[2]-pulse_np[0] #subtracts first onset and first offset time
    frequency = np.around((1/(pulse_np[1]-pulse_np[0]))) #subtracts first and second pulse onset 
    length = np.ceil(pulse_np[-1] - pulse_np[0]) #subtracts last offset and first onset
    print (frequency, pulse_width, length)

    return (frequency, length, pulse_width)

#format means dataframe

def table (Means, plot=True):
    
    subset1 = Means[['Sweep','Frequency','MeanPre']]
    subset2 = Means[['Sweep','Frequency','MeanStim']]
    subset3 = Means[['Sweep','Frequency','MeanPost']]
    
    Means_1 = pd.DataFrame(columns = ['Sweep','Frequency','MeanPre', 'Condition'])
    Means_1 = Means_1.append(subset1, ignore_index = True)
    Means_1['Condition'] = 0
    Means_1.loc[Means_1['MeanPre'] != 0, 'Condition'] = 'Pre'
    Means_1 = Means_1.rename(columns={"MeanPre": "mean"})
    
    Means_2 = pd.DataFrame(columns = ['Sweep','Frequency','MeanStim', 'Condition'])
    Means_2 = Means_2.append(subset2, ignore_index = True)
    Means_2['Condition'] = 0
    Means_2.loc[Means_2['MeanStim'] != 0, 'Condition'] = 'Stim'
    Means_2 = Means_2.rename(columns={"MeanStim": "mean"})
    
    Means_3 = pd.DataFrame(columns = ['Sweep','Frequency','MeanPost', 'Condition'])
    Means_3 = Means_3.append(subset3, ignore_index = True)
    Means_3['Condition'] = 0
    Means_3.loc[Means_3['MeanPost'] != 0, 'Condition'] = 'Post'
    Means_3 = Means_3.rename(columns={"MeanPost": "mean"})
    
    Final = pd.DataFrame(columns = ['Sweep','Frequency','mean', 'Condition'])
    Final = Final.append(Means_1, ignore_index = True)
    Final = Final.append(Means_2, ignore_index = True)
    Final = Final.append(Means_3, ignore_index = True)
    
    if plot:
        plt.figure(figsize=(15, 7))
        speed_chart = sns.swarmplot(x ="Condition", y="mean", hue="Frequency", data=Final)
        plt.show()
        plt.savefig("swarmplot.png")
        plt.savefig("swarmplot.pdf")

    return Final

def Ratio (Means, plot=True):
    
    Means['StimvsPre'] = Means['MeanStim'] - Means['MeanPre']
    Means['PostvsPre'] = Means['MeanPost'] - Means['MeanPre']

    Ratio = Means[['Sweep','Frequency','StimvsPre','PostvsPre',]].copy()
    
    subset1 = Means[['Sweep','Frequency','StimvsPre']]
    subset2 = Means[['Sweep','Frequency','PostvsPre']]
    
    Means_1 = pd.DataFrame(columns = ['Sweep','Frequency','StimvsPre', 'Condition'])
    Means_1 = Means_1.append(subset1, ignore_index = True)
    Means_1['Condition'] = 0
    Means_1.loc[Means_1['StimvsPre'] != 0, 'Condition'] = 'StimvsPre'
    Means_1 = Means_1.rename(columns={"StimvsPre": "ratio"})
    
    Means_2 = pd.DataFrame(columns = ['Sweep','Frequency','PostvsPre', 'Condition'])
    Means_2 = Means_2.append(subset2, ignore_index = True)
    Means_2['Condition'] = 0
    Means_2.loc[Means_2['PostvsPre'] != 0, 'Condition'] = 'PostvsPre'
    Means_2 = Means_2.rename(columns={"PostvsPre": "ratio"})
    
    Final2 = pd.DataFrame(columns = ['Sweep','Frequency','ratio', 'Condition'])
    Final2 = Final2.append(Means_1, ignore_index = True)
    Final2 = Final2.append(Means_2, ignore_index = True)
    
    if plot:
        plt.figure(figsize=(15, 7))
        speed_chart = sns.boxplot(x ="Condition", y="ratio", hue="Frequency", data=Final2)
        plt.show()   
        plt.savefig("violinplot.png")
        plt.savefig("violinplot.pdf")

    return Final2
