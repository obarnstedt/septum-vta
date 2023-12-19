# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:11:24 2022

@author: Kevin Luxem, Petra Mocellin and Dennis Dalügge
"""

import os
import neo
import tqdm
import natsort
import numpy as np
from pathlib import Path
from igor import binarywave as bw


class load():
    def __init__(self, simpleLFP_directory, filesdir, FolderName, ad=None, filetype=None):
        self.simpleLFP_directory = simpleLFP_directory
        self.filesdir = filesdir
        self.FolderName = FolderName
        self.ad = ad
        self.filetype = filetype
        
        os.chdir(self.filesdir)
    
    def ImportData(self):
        if self.filetype == ".ibw":
            files, Waves, TimeVec, _ = self.ImportFolder()
            return  files, Waves, TimeVec, _
    
        if self.filetype == ".rhd":
            files, Waves, _, analogsignal_all = self.INTANreader()
            return files, Waves, _, analogsignal_all
    
    
    def get_folder_directory(self):
        filesdir = os.getcwd()
        a = len(self.FolderName)
        if filesdir[-a:] != self.FolderName:
            os.chdir(self.FolderName)
            filesdir = os.getcwd()
        
        return filesdir
        
    def INTANreader(self):
        filesdir = self.get_folder_directory()
        p = Path(filesdir)
        filenames = [i.stem for i in p.glob('**/*.rhd')]
                
        signal_waves = []
        analogsignal_all = []
        for i, file in enumerate(tqdm.tqdm(filenames)):
            IO = neo.io.IntanIO(os.path.join(filesdir, file+'.rhd'))
            Block = IO.read_block()
            
            analogsignal_array = Block.segments[0].analogsignals
            
            # detect lfp signal channels 
            time_points = 0
            channels = 0
            for i, analogsignal in enumerate(analogsignal_array):
                if analogsignal.shape[0] > time_points and analogsignal.shape[1] > channels:
                    signal = analogsignal
                    time_points = analogsignal.shape[0]
                    channels = analogsignal.shape[1]
        
            signal_waves.append(signal)
            analogsignal_all.append(analogsignal_array)
            
        for signal in signal_waves:
            print(signal.shape[0], signal.shape[1])
            
        os.chdir(self.simpleLFP_directory)   
            
        return filenames, signal_waves, 0, analogsignal_all
    
    def ImportFolder(self):
        filesdir = self.get_folder_directory()
        
        ''' Imports Current Clamp Traces from Igor Binary Files '''
        ''' With Wave, Sampling Rate and Recording Time [ms] '''
        files = [f for f in os.listdir(filesdir) if (f.endswith(self.filetype) and (f.startswith(self.ad)))]
        i = 0
        length_files = [i for i in range(len(files))]
        while i < len(files):
            length_files[i] = len(files[i])
            i += 1
        if all(x==length_files[0] for x in length_files):
            files.sort() 
        else:
            files = natsort.natsorted(files)

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
        
        os.chdir(self.simpleLFP_directory)      
        return files, Waves, TimeVec
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    