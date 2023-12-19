import os
import sys
sys.path.append('path')
import DDImport
import Treadmill_Opto_Functions as PM_Opto_functions
import pandas as pd

#define directory
filesdir ='filesdir'
os.chdir(filesdir)

#define folder 
FolderName = 'FolderName'
os.chdir(FolderName)

#define variables
PFiles, PWaves,PTimeVec,PSampFreq,PRecTime = DDImport.ImportFolder(FolderName,'ad2','.ibw') #position
SFiles, SWaves,STimeVec,SSampFreq,SRecTime = DDImport.ImportFolder(FolderName,'Stim2','.ibw') #opto stimulation

#downsample time and stim from 10 to 1 kHz
Time, Stim = PM_Opto_functions.downsample(PTimeVec,SWaves)

#POSITION CALCULATIONS
#reset y axis from -4.5 to +4.5 into 0 to 360 cm
PWaves_scaled = [(n + 4.5)/9*360 for n in PWaves]

#calculate correct position
corrected_position_list = PM_Opto_functions.position(PWaves, PWaves_scaled, STimeVec, plot=True) 

#SPEED CALCULATIONS

#calculate speed
velocityArray = PM_Opto_functions.speed(corrected_position_list, FolderName, plot=True)

#create dataframes

Overview=pd.DataFrame()
Means = pd.DataFrame()
for i in range(len(velocityArray)):
    name = "ad2_" + str(i+1)
    df=pd.DataFrame()
    print("ad2_" + str(i+1))
    df['Time'] = Time[i]
    df['Speed'] = velocityArray[i]
    df['Stim'] = Stim[i]
    df['StimON'] = df['Stim'].diff(periods=1)
    frequency, length, pulse_width = PM_Opto_functions.Stim_frequency(df)
    df['Frequency'] = 0
    df.loc[df['Time'] != 'NaN', 'Frequency'] = frequency
    df['Sweep'] = (name)
    ind = df.loc[df.StimON != 0].index
    df['PreStimPost'] = 'Post'
    df.loc[(ind[0]):(ind[1]), 'PreStimPost'] = 'Pre'
    df.loc[(ind[1]):(ind[-1]), 'PreStimPost'] = 'Stim'
    MeanPre = df.loc[df['PreStimPost'] == 'Pre', 'Speed'].mean()
    MeanStim = df.loc[df['PreStimPost'] == 'Stim', 'Speed'].mean()
    MeanPost = df.loc[df['PreStimPost'] == 'Post', 'Speed'].mean()
    dF=pd.DataFrame(columns = ['Sweep','Frequency','MeanPre', 'MeanStim', 'MeanPost'], index = [i])
    dF['Sweep'] = name
    dF['MeanPre'] = MeanPre
    dF['MeanStim'] = MeanStim
    dF['MeanPost'] = MeanPost 
    dF['Frequency'] = frequency
    Means = Means.append(dF,ignore_index = True)
    Overview = Overview.append(df, ignore_index = True)

#format the dataframe with mean speed for each sweep/condition

Final = PM_Opto_functions.table(Means, plot = True)
Final2 = PM_Opto_functions.Ratio(Means, plot=True)

Means.to_excel(FolderName + ".xlsx") 
