''' Script for Importing Data '''

''' Importing scripts '''    
import os
from igor import binarywave as bw
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt 
import math
import pandas as pd
import natsort
import imageio


''' To Test
    cwd = os.getcwd()   
    filesdir = os.path.abspath('..')
    os.chdir(filesdir)    
    filesdir = '/Users/DennisDa/Desktop/DataAnalaysis'

'''    

''' 8ung: Import Sampling Frequency in kHz '''

#class Folder:    
#    def ImpC(filesdir):
#        ''' Imports Current Clamp Traces from Igor Binary Files '''
#        ''' With Wave, Sampling Rate and Recording Time [ms] '''
#        files = [f for f in os.listdir(filesdir) if (f.endswith('.ibw') and f.startswith('ad1'))]
#        files.sort(key=len)
#        A=[i for i in range(len(files))]
#        Waves=[i for i in range(len(files))]
#        SampFreq=[i for i in range(len(files))]
#        Head=[i for i in range(len(files))]
#        RecTime=[i for i in range(len(files))]
#        count=0
#        
#        for Wave in files:
#            A[count] = bw.load(Wave)
#            Waves[count] = A[count]['wave']['wData']
#            Head[count] = A[count]['wave']['wave_header']
#            if 'hsA' in Head[count]:
#                SampFreq[count] = 1/Head[count]['hsA'] 
#                RecTime[count] = Head[count]['hsA']*Head[count]['npnts']
#            elif 'sfA' in Head[count]:
#                SampFreq[count] = 1/Head[count]['sfA'][0]
#                RecTime[count] = Head[count]['sfA'][0] *Head[count]['npnts'] 
#            else:
#                SampFreq[count] = float('nan')
#                RecTime[count] = float('nan')
#            
#            count +=1
#        
#        TimeVec=[i for i in range(len(SampFreq))]
#        count=0
#        for W in Waves:
#            TimeVec[count] = np.linspace(0.0,RecTime[count], num=(len(Waves[count])))
#            count +=1
#              
#        return Waves,TimeVec,SampFreq,RecTime
    
#def PlotFolder(Foldername):
#    ''' Imports Current Clamp Traces from Igor Binary Files and ... '''
#    ''' ... Plots all in One Window '''
#    import DDImport as FI
#    
#    Names,Waves,TimeVec,SampFreq,RecTimes = FI.ImportFolder(Foldername)
#    NumPlots = len(Waves)
#
#    # Plotting:
#    NumSubplt = math.ceil(math.sqrt(NumPlots))
#    fig, ax = plt.subplots(nrows=NumSubplt,ncols=NumSubplt)
#    
#    count=0
#    for W in Waves:
#        plt.subplot(NumSubplt,NumSubplt,count+1)
#        plt.plot(TimeVec[count],Waves[count])
#        plt.title(Names[count])
#        count +=1
#
#    return plt.show(fig)

def ImportFolder(Foldername):
    filesdir = os.getcwd()
    a = len(Foldername)
    if filesdir[-a:] != Foldername:
        os.chdir(Foldername)
        filesdir = os.getcwd()
    
    ''' Imports Current Clamp Traces from Igor Binary Files '''
    ''' With Wave, Sampling Rate and Recording Time [ms] '''
    files = [f for f in os.listdir(filesdir) if (f.endswith('.ibw') and f.startswith('ad1'))]
    i = 0
    length_files = [i for i in range(len(files))]
    while i < len(files):
        length_files[i] = len(files[i])
        i += 1
#    print(length_files)
    if all(x==length_files[0] for x in length_files):
        files.sort()
#        print('no',files)    
    else:
#        files.sort(key=lambda s: int(s[4:-4]))
        files = natsort.natsorted(files)
#        print('yes',files)
    A=[i for i in range(len(files))]
    Waves=[i for i in range(len(files))]
    SampFreq=[i for i in range(len(files))]
    Head=[i for i in range(len(files))]
    RecTime=[i for i in range(len(files))]
    count=0
    
    for Wave in files:
        A[count] = bw.load(Wave)
        Waves[count] = A[count]['wave']['wData']
        Head[count] = A[count]['wave']['wave_header']
        if 'hsA' in Head[count]:
            SampFreq[count] = 1/Head[count]['hsA'] 
            RecTime[count] = Head[count]['hsA']*Head[count]['npnts']
        elif 'sfA' in Head[count]:
            SampFreq[count] = 1/Head[count]['sfA'][0]
            RecTime[count] = Head[count]['sfA'][0] *Head[count]['npnts'] 
        else:
            SampFreq[count] = float('nan')
            RecTime[count] = float('nan')
        
        count +=1
    
    TimeVec=[i for i in range(len(SampFreq))]
    count=0
    for W in Waves:
        TimeVec[count] = np.linspace(0.0,RecTime[count], num=(len(Waves[count])))
        count +=1
    
    os.chdir("..")      
    return files, Waves,TimeVec,SampFreq,RecTime

''' Import Pics '''
def ImportPicsFolder(Foldername):
    filesdir = os.getcwd()
    a = len(Foldername)
    if filesdir[-a:] != Foldername:
        os.chdir(Foldername)
        filesdir = os.getcwd()
        
    ''' Imports Pics as numpy.ndarray '''
    
    files = [f for f in os.listdir(filesdir) if (f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.TIF'))]
    Pics =[i for i in range(len(files))]
    count = 0
    for P in Pics:
        
        Pics[count]=imageio.imread(files[count]) #imred substituted with imageio
        count +=1
    
    os.chdir("..")      
    return Pics

''' Import ExcelFiles '''
def ImportExcel(Name):
    Folderpath = os.getcwd()
#    print(Folderpath)
#    print(os.listdir(Folderpath))
    files = [f for f in os.listdir(Folderpath) if (f.endswith('.xlsx'))]
    if Name+'.xlsx' in files:
        output = pd.read_excel(Name+'.xlsx',index_col=0)
        output1 = 1
        return output, output1
    else:
        output = None
        output1 = 0
        return output, output1
        return print('Excel-File does not exists in Dictionary')
    
''' Import TextFiles '''
def ImportTxt(Name):
    Folderpath = os.getcwd() 
    files = [f for f in os.listdir(Folderpath) if (f.endswith('.txt'))]
    if Name+'.txt' in files:
        output = pd.read_csv(Name+'.txt',sep='\t', lineterminator='\r',header=None)
        output1 = 1
        return output, output1
    else:
        output = None
        output1 = 0
        return output, output1
        return print('Text-File does not exists in Dictionary')
    
''' Import Single ibw for VClamp '''
def SingleVClampTrace(WaveName):
    A = bw.load(WaveName)
    Wave = A['wave']['wData']
    Head = A['wave']['wave_header']
    if 'hsA' in Head:
        SampFreq = 1/Head['hsA'] 
        RecTime = Head['hsA']*Head['npnts']
    elif 'sfA' in Head:
        SampFreq = 1/Head['sfA'][0]
        RecTime = Head['sfA'][0] *Head['npnts'] 
    else:
        SampFreq = float('nan')
        RecTime = float('nan')
        
    TimeVec = np.linspace(0.0,RecTime, num=(len(Wave)))
    
    os.chdir("..")      
    return Wave,TimeVec,SampFreq,RecTime
        
    



 