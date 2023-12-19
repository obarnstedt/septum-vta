#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:52:35 2017

''' Script to Sort Traces for CellCharacterisation '''

''' Needs: TextFile With Stimulations and Raw Traces:

@author: DennisDa
"""

''' Import Scripts '''
import os
import DDImport
import ddhelp
import numpy as np
import math
import pandas as pd
import shutil as sh
from SuprathresholdProperties import FindAPs2 as FindAPs
''' For Testing '''
Testing = 0
if Testing == 1:
    
    ## Get Path To Data:
    PathToData = '/Users/DennisDa/Desktop/DataAnalaysis/Test Sorting'
    os.chdir(PathToData)
    
    ## Get To CellFolder:
    Cellname = 'DD_20170823_S2C1'
    #'DD_20170621_S1C1'
    ''' Go To CellFolder '''
    os.chdir(Cellname)
    
    ''' Variables for what to Analyse:'''
    Conditions =[]
    Conditions = {'NumIRTraces':15}
    Conditions ['OVSortingTableName'] = 'SortingOV.xlsx'
    
    
''' Helpers'''
def FindStrInMixedTable(Table,String):
    NewTable = Table.select_dtypes(exclude=['floating'])
    mask = np.column_stack([NewTable[col].str.contains(String, na=False) for col in NewTable])
    Index = Table.loc[mask.any(axis=1)]
    return Index

def SpecifyTracesByCurrentInput(TableOriginal,MaxMin):
    # Max = 1, Min = -1:
    Table = TableOriginal.select_dtypes(exclude=['floating'])
    NewTable1 = Table.convert_objects(convert_numeric=True)
    NewTable = NewTable1.iloc [:,:-1]
    if MaxMin > 0: 
        Value1 = NewTable.max(skipna=1,numeric_only=float)
        Value = Value1.tolist()
        Value = np.nanmax(Value)
    elif MaxMin < 0:
        Value1 = NewTable.min(skipna=1,numeric_only=float)
        Value = Value1.tolist()
        Value = np.nanmin(Value) 
    
    OutPutTable = FindStrInMixedTable(TableOriginal,str(int(Value)))
    NotIncluded = pd.merge(TableOriginal, OutPutTable, how='outer', indicator=True).query('_merge=="left_only"').drop('_merge',1)
    return OutPutTable,Value,NotIncluded

def FindLPRXTimes(TableOriginal,LPRValue):
    Table = TableOriginal.select_dtypes(exclude=['floating'])   
    NewTable = Table.convert_objects(convert_numeric=True)
    CurrentInputs = NewTable._get_numeric_data()
    XValues = CurrentInputs.dropna(axis=1, how='any')
    OnlyCurrentInputsT = XValues.iloc[:,:-1]
    OnlyCurrentInputsV = OnlyCurrentInputsT.as_matrix()
    CurrentInput = np.unique(OnlyCurrentInputsV)
    if len(CurrentInput) > 1:    
        i = 0
        XLPR = [i for i in range(len(CurrentInput))]
        OutPutTable = [None] * len (CurrentInput)
        while i < len(CurrentInput):
            XLPR[i] = math.ceil(CurrentInput[i]/LPRValue)
            OutPutTable[i] = FindStrInMixedTable(TableOriginal,str(int(CurrentInput[i]))) 
            i += 1
    else:
        XLPR = math.ceil(CurrentInput/LPRValue)
        OutPutTable = FindStrInMixedTable(TableOriginal,str(int(CurrentInput)))    
        
    return CurrentInput, XLPR, OutPutTable

def findDayAndTime(TableOriginal):
    for col in TableOriginal.columns:
        if TableOriginal[col].dtype == 'object':
            try:
                Date = pd.to_datetime(TableOriginal[col])
                Time = pd.to_timedelta(TableOriginal[col])
            except ValueError:
                pass
    A = Date.iloc[0]
    DayString = A.strftime('%Y %m %d')
    B = str(Time.iloc[0])
    DayTime = B[7:]
    return DayString, DayTime

def CopyGoodFilesB(FileNameArray,FolderName,CellName):
    if not os.path.isdir(os.getcwd()+'/'+FolderName):
        os.makedirs(os.getcwd()+'/'+FolderName) 
    destination = os.getcwd()+'/'+FolderName     
    # Move Files: 
    i = 0
    while i < len (FileNameArray):
        sh.move(FileNameArray[i],destination)
        i += 1
    return print(FolderName+' Saved')

def CopyBadFilesB(FileNameArray,FolderName,FolderNameDirect,CellName):
    os.chdir(FolderNameDirect)
    if not os.path.isdir(os.getcwd()+'/'+FolderName):
        os.makedirs(os.getcwd()+'/'+FolderName) 
    destination = os.getcwd()+'/'+FolderName
    os.chdir("..")       
    # Move Files: 
    i = 0
    while i < len (FileNameArray):
        sh.move(FileNameArray[i],destination)
        i += 1
    return print(FolderName+' Saved')

def QualityAPs(FolderName,APThreshold):
    Names, Waves,Times,SampFreq,RecTime = DDImport.ImportFolder(FolderName)
     # Create Stimulus:
    Stimulus = [i for i in range(len(Waves))]
    count = 0
    for W in Waves:
        Stimulus[count] = np.zeros(len(Waves[count]))
        Stimulus[count][int(100*SampFreq[count]):int(600*SampFreq[count])] = 1
        count +=1    
    StimDiffWave = np.diff(Stimulus[0])
    StimDiffPoints = np.where(StimDiffWave != 0)
    StimOnset = np.asarray(StimDiffPoints[0][0])
    StimOffset = np.asarray(StimDiffPoints[0][1])
    
    # FindAPs:
    AP = [None]*len(Waves)
    NoAP = [None]*len(Waves)
    count = 0
    for W in Waves:
        if np.max(Waves[count][StimOnset:StimOffset]) > APThreshold:
            AP[count] = Names[count]
        else:
            NoAP[count] = Names[count]  
        count +=1
    AP = [x for x in AP if x is not None]
    NoAP = [x for x in NoAP if x is not None]
    
    return NoAP, AP 

def QualityToManyAPs(FolderName,APThreshold):
    Names, Waves,Times,SampFreq,RecTime = DDImport.ImportFolder(FolderName)
     # Create Stimulus:
    Stimulus = [i for i in range(len(Waves))]
    count = 0
    for W in Waves:
        Stimulus[count] = np.zeros(len(Waves[count]))
        Stimulus[count][int(100*SampFreq[count]):int(600*SampFreq[count])] = 1
        count +=1    
    StimDiffWave = np.diff(Stimulus[0])
    StimDiffPoints = np.where(StimDiffWave != 0)
    StimOnset = np.asarray(StimDiffPoints[0][0])
    StimOffset = np.asarray(StimDiffPoints[0][1])
    
    # FindAPs:
    APs = [None]*len(Waves)
    APsClass = [None]*len(Waves)
    count = 0
    for W in Waves:
        if np.max(Waves[count][StimOnset:StimOffset]) > APThreshold:
            APsClass[count] = FindAPs(Times[count][StimOnset:StimOffset],Waves[count][StimOnset:StimOffset],SampFreq[count],2,1,0)
            APs[count] = APsClass[count].APNum
#            APFreq[count] = (1/APsClass[count].APIntervals)*1000
        count +=1
    MultipleAPs = [None]*len(Waves)
    count = 0
    for A in APs:
        if APs[count] > 1:
            # New:
            Frequencies = (1/APsClass[count].APIntervals)*1000
            if np.any(Frequencies < 100):
                MultipleAPs[count]=Names[count]  
        count += 1
    MultipleAPs = [x for x in MultipleAPs if x is not None]        
    
    return MultipleAPs

   
''' Start of the MainScript '''
class SortingCellCharacteristics:
    def __init__ (self,Cellname,Conditions): 
        self.CellName = Cellname
        self.Conditions = Conditions
        
        ''' UnRavel Conditions '''
        self.NumIRTraces = ddhelp.SearchKeyDic(self.Conditions,'NumIRTraces')
        self.OVSortingTableName = ddhelp.SearchKeyDic(self.Conditions,'OVSortingTableName')
         
        ''' Import OVSorting Table '''
        os.chdir("..")
        if self.OVSortingTableName is not None:
    #        print(os.getcwd())
    #        self.OVSortingTableName = [f for f in os.listdir(os.getcwd()) if (f.endswith('.xlsx') and f.startswith('Sorting'))] 
            self.OVSortingTable = pd.read_excel(self.OVSortingTableName,index_col=0)
        os.chdir(self.CellName)
        
        ''' Copy complete Folder into New Folder 'Sorted': '''
        self.OriginalPath = os.getcwd()
        os.chdir("..")
        # Create Sorted Folder: 
        if not os.path.isdir(os.getcwd()+'/Sorted'):
            os.makedirs(os.getcwd()+'/Sorted')   
        os.chdir(os.getcwd()+'/Sorted')
        self.NewPath = os.getcwd()        
        # Delete Folder if exists:
        self.Folders =  [name for name in os.listdir(self.NewPath) if os.path.isdir(os.path.join(self.NewPath, name))]
        if self.CellName in self.Folders:
            sh.rmtree(self.CellName)

        # Copy Whole Folder:
        self.NewPath = self.NewPath + '/' + self.CellName
        sh.copytree(self.OriginalPath,self.NewPath)
        os.chdir(self.CellName)
        
        ''' Import TextFile '''
        self.OVTable,self.Done = DDImport.ImportTxt('StimHistoryTable')
        self.OVTable.drop(self.OVTable.index[len(self.OVTable)-1],inplace=True)
        self.OVTable.drop(self.OVTable.columns[len(self.OVTable.columns)-1], axis=1, inplace=True)
        
        ''' Look for Pictures in Folders and Copy to New Folder Pic '''
        self.PicFolders = [name for name in os.listdir(self.NewPath)\
                           if os.path.isdir(os.path.join(self.NewPath, name))]
        if self.PicFolders: 
            if not os.path.isdir(os.getcwd()+'/Pics'):
                os.makedirs(os.getcwd()+'/Pics')
            self.PathPicFolder = os.getcwd()+'/Pics'
            
            self.PicInFolders = [None]*len(self.PicFolders)
            i = 0
            while i < len(self.PicFolders):
                self.PicInFolders[i] = [f for f in os.listdir(self.PicFolders[i]) if (f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.TIF'))] 
                if len(self.PicInFolders[i]) < 4:
                    if len(self.PicInFolders[i]) > 1:
                        j = 0
                        while j < len(self.PicInFolders[i]):
                            Filepath = [self.PicFolders[i] +'/' + self.PicInFolders[i][j]]
                            CopyGoodFilesB(Filepath,'Pics',self.CellName)
                            j += 1
                        sh.rmtree(self.PicFolders[i])
                    else:
                        Filepath = [self.PicFolders[i] + '/' + self.PicInFolders[i][0]]
                        CopyGoodFilesB(Filepath,'Pics',self.CellName)
                        sh.rmtree(self.PicFolders[i])
                else:
                    FolderToMove = os.getcwd()+'/'+self.PicFolders[i]
                    Destination = os.getcwd()+'/Pics'
                    sh.move(FolderToMove,Destination)    
                i +=1
        ''' Look for Pictures and Copy New Folder Pic: '''
        self.PossiblePics = [f for f in os.listdir(os.getcwd()) if (f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.TIF'))] 
        if self.PossiblePics:
            CopyGoodFilesB(self.PossiblePics,'Pics',self.CellName)  
        
        ''' Table for Current Clamp:'''
        self.OVCellCharac = FindStrInMixedTable(self.OVTable,'CC')
        # Trace not in Table:
        self.IndiciesNotInList = list(self.OVCellCharac.index.values)
        self.NotInList, = np.where(np.diff(self.IndiciesNotInList)>1)
        
        ''' Add Filenames To Table: '''
        self.filenames = [f for f in os.listdir(os.getcwd()) if (f.endswith('.ibw') and f.startswith('ad1'))]
        self.filenames.sort(key = len)
        self.fileindicies = [i for i in range(len(self.filenames))]
        i = 0
#        print(len(self.OVTable.index))
#        print(len(self.filenames))
        while i < len(self.filenames):
            self.fileindicies[i] = ddhelp.get_num(self.filenames[i])
            self.fileindicies[i] = str(self.fileindicies[i])
            self.fileindicies[i] = self.fileindicies[i][1:]
            i += 1
#        print(len(self.fileindicies))
#        print(len(self.OVCellCharac.index))
#        print(self.fileindicies )
        if len(self.fileindicies) == (len(self.OVCellCharac.index))+1:
            self.fileindicies =self.fileindicies[:-1]
        self.OVCellCharac['FileNames'] = self.fileindicies 
        
        ''' Sort Table by Stimulation Protocols '''
        self.IRTraces = FindStrInMixedTable(self.OVCellCharac,'IR')
        self.HypoTraces = FindStrInMixedTable(self.OVCellCharac,'RecordingForSag')
        self.JustSubTraces = FindStrInMixedTable(self.OVCellCharac,'Sub Thers')
        self.APTraces = FindStrInMixedTable(self.OVCellCharac,'Sup Thers')
        self.FiringTraces = FindStrInMixedTable(self.OVCellCharac,'Mult LPR')
        
        ''' Rest of the Traces: '''
        self.UndefinedTraces = FindStrInMixedTable(self.OVCellCharac,'MultiStim')

        ''' Find Exact Traces To Put in Folder from current input: '''
        self.QualityComplete = ['IR']
        if self.HypoTraces.size > 0:
            self.HypoTracesConcrete,self.HypoInput,self.HypoTracesNot = SpecifyTracesByCurrentInput(self.HypoTraces,-1)
            self.QualityComplete.append(', Hypo50')
        else:
             self.HypoInput = np.nan   
        if self.JustSubTraces.size > 0:
            self.JustSubConcrete,self.JustSubInput,self.JustSubTracesNot = SpecifyTracesByCurrentInput(self.JustSubTraces,1)
            self.QualityComplete.append(', JustSub')
        else:
             self.JustSubInput = np.nan
        if self.APTraces.size > 0:    
            self.APTracesConcrete,self.APInput,self.APTracesNot = SpecifyTracesByCurrentInput(self.APTraces,1)
            self.QualityComplete.append(', LPR')
        else:
             self.APInput = np.nan
        if self.FiringTraces.size > 0: 
            # Find LPR X Times:
            self.FiringInput,self.XLPR,self.FiringTracesConcrete = FindLPRXTimes(self.FiringTraces,self.APInput)
        
        ''' Sort To New Created Folders: Traces To it: '''        
        # Sort and Copy IR:
        if self.IRTraces.size > 0:
            self.NumIR,a = self.IRTraces.shape
            self.NumIRRepetitions = int(self.NumIR/self.NumIRTraces)
            self.IRTraceNumbers = pd.Series.as_matrix(self.IRTraces.iloc[:,-1].astype('float64'))
            self.BasicText = [None]*self.NumIRRepetitions 
            self.BasicText[0]='Basic'
            self.IRfilenames = [i for i in range(self.NumIRRepetitions)]
            # Prepare for Copying:
            if self.NumIRRepetitions > 1:
                i = 0
                while i < self.NumIRRepetitions:
                    self.IRfilenames[i] = [j for j in range(self.NumIRTraces)]
                    self.BasicText[i]='Basic '+str(i+1)
                    j = 0
                    while j < self.NumIRTraces: 
                        self.IRfilenames[i][j] = 'ad1_' + str(int(self.IRTraceNumbers[(i*self.NumIRTraces)+j])) +'.ibw'
                        j += 1
                    i += 1
                # CopyFunction: 
                i = 0
                self.BasicText[0]='Basic'
                while i < self.NumIRRepetitions:
                    CopyGoodFilesB(self.IRfilenames[i],self.BasicText[i],self.CellName)
                    i += 1
                
        # Sort and Copy Hypo: 
        if self.HypoTraces.size > 0:
            self.NumHypoTraces = len(self.HypoTracesConcrete.index)
            self.HypoNumbers = pd.Series.as_matrix(self.HypoTracesConcrete.iloc[:,-1].astype('float64'))
            self.HypoNames = [i for i in range(self.NumHypoTraces)]
            i = 0
            while i < self.NumHypoTraces:
                self.HypoNames[i] = 'ad1_' + str(int(self.HypoNumbers[i])) +'.ibw'
                i += 1
            CopyGoodFilesB(self.HypoNames,'Hypo50',self.CellName)
            
            if self.HypoTracesNot.size > 0:
                self.NumHypoBadTraces = len(self.HypoTracesNot.index)
                self.HypoBadNumbers = pd.Series.as_matrix(self.HypoTracesNot.iloc[:,-1].astype('float64'))
                self.HypoBadNames = [i for i in range(self.NumHypoBadTraces)]
                i = 0
                while i < self.NumHypoBadTraces:
                    self.HypoBadNames[i] = 'ad1_' + str(int(self.HypoBadNumbers[i])) +'.ibw'
                    i += 1
                CopyBadFilesB(self.HypoBadNames,'Excluded','Hypo50',self.CellName)

        # Sort and Copy JustSubTraces: 
        if self.JustSubTraces.size > 0:
            self.NumJustSubTraces = len(self.JustSubConcrete.index)
            self.JustSubNumbers = pd.Series.as_matrix(self.JustSubConcrete.iloc[:,-1].astype('float64'))
            self.JustSubNames = [i for i in range(self.NumJustSubTraces)]
            i = 0
            while i < self.NumJustSubTraces:
                self.JustSubNames[i] = 'ad1_' + str(int(self.JustSubNumbers[i])) +'.ibw'
                i += 1
            CopyGoodFilesB(self.JustSubNames,'JustSub',self.CellName)
            
            if self.JustSubTracesNot.size > 0:
                self.NumJustSubBadTraces = len(self.JustSubTracesNot.index)
                self.JustSubBadNumbers = pd.Series.as_matrix(self.JustSubTracesNot.iloc[:,-1].astype('float64'))
                self.JustSubBadNames = [i for i in range(self.NumJustSubBadTraces)]
                i = 0
                while i < self.NumJustSubBadTraces:
                    self.JustSubBadNames[i] = 'ad1_' + str(int(self.JustSubBadNumbers[i])) +'.ibw'
                    i += 1
                CopyBadFilesB(self.JustSubBadNames,'Excluded','JustSub',self.CellName)
                
        # Sort and Copy APTraces: 
        if self.APTraces.size > 0:
            self.NumAPTraces = len(self.APTracesConcrete.index)
            self.APNumbers = pd.Series.as_matrix(self.APTracesConcrete.iloc[:,-1].astype('float64'))
            self.APNames = [i for i in range(self.NumAPTraces)]
            i = 0
            while i < self.NumAPTraces:
                self.APNames[i] = 'ad1_' + str(int(self.APNumbers[i])) +'.ibw'
                i += 1
            CopyGoodFilesB(self.APNames,'LPR',self.CellName)
            
            if self.APTracesNot.size > 0:
                self.NumAPBadTraces = len(self.APTracesNot.index)
                self.APBadNumbers = pd.Series.as_matrix(self.APTracesNot.iloc[:,-1].astype('float64'))
                self.APBadNames = [i for i in range(self.NumAPBadTraces)]
                i = 0
                while i < self.NumAPBadTraces:
                    self.APBadNames[i] = 'ad1_' + str(int(self.APBadNumbers[i])) +'.ibw'
                    i += 1
                CopyBadFilesB(self.APBadNames,'Excluded','LPR',self.CellName)
                
        # Sort and Copy Firing Traces:
        if self.FiringTraces.size > 0:
            if isinstance(self.XLPR, list):
                i = 0
                self.FiringNames = [None]*len(self.XLPR)
                self.TextLPRX = [None]*len(self.XLPR)
                self.FiringNumbers = [None]*len(self.XLPR)
                while i < len(self.XLPR):
                    self.NumFiringTraces = len(self.FiringTracesConcrete[i].index)
                    self.FiringNames[i] = [j for j in range(self.NumFiringTraces)]
                    self.TextLPRX [i]= 'LPR'+str(int(self.XLPR[i]))
                    self.FiringNumbers[i] = pd.Series.as_matrix(self.FiringTracesConcrete[i].iloc[:,-1].astype('float64'))
                    j = 0
                    while j < self.NumFiringTraces:
                        
                        self.FiringNames[i][j] = 'ad1_' + str(int(self.FiringNumbers[i][j])) +'.ibw'
                        j +=1
                    CopyGoodFilesB(self.FiringNames[i],self.TextLPRX[i],self.CellName)
                    i +=1
                self.QualityFiringComplete = ', '.join(self.TextLPRX)
            else:
                self.NumFiringTraces = len(self.FiringTracesConcrete.index)
                self.FiringNumbers = pd.Series.as_matrix(self.FiringTracesConcrete.iloc[:,-1].astype('float64'))   
                self.FiringNames = [i for i in range(self.NumFiringTraces)] 
                self.TextLPRX = 'LPR'+str(int(self.XLPR))
                i = 0
                while i < self.NumFiringTraces:
                    self.FiringNames[i] = 'ad1_' + str(int(self.FiringNumbers[i])) +'.ibw'
                    i += 1    
                CopyGoodFilesB(self.FiringNames,self.TextLPRX,self.CellName)
                self.QualityFiringComplete = self.TextLPRX
            # QualityComplete:
            self.QualityComplete.append(', ')
            self.QualityComplete.append(self.QualityFiringComplete)
        
        ''' Rest of the Traces '''
        self.StimFiles = [f for f in os.listdir(os.getcwd()) if (f.endswith('.ibw') and f.startswith('ad7'))]
        CopyGoodFilesB(self.StimFiles,'Z_StimFiles',self.CellName)
        self.VClampFiles = [f for f in os.listdir(os.getcwd()) if (f.endswith('.ibw') and f.startswith('ad0'))]
        CopyGoodFilesB(self.VClampFiles,'Z_VClampFiles',self.CellName)
        self.CClampFiles = [f for f in os.listdir(os.getcwd()) if (f.endswith('.ibw') and f.startswith('ad1'))]
        CopyGoodFilesB(self.CClampFiles,'Z_CClampFiles',self.CellName)
        self.Notes1 = [f for f in os.listdir(os.getcwd()) if "StimHist" in f or "ExperimentNote" in f]
        CopyGoodFilesB(self.Notes1,'Z_Notes',self.CellName)
        self.Averages = [f for f in os.listdir(os.getcwd()) if (f.endswith('.ibw') and f.startswith('avg_'))]
        CopyGoodFilesB(self.Averages,'Z_Averages',self.CellName)
        
        ''' Create XFile: '''
        self.XFile = pd.DataFrame({'Variables': ['Date','StartTime','Animal Number','Birth',\
                                                 'Genotype','Virus Injection','InjectionDate',\
                                                 'Tissue','Region','Layer',\
                                                 'Celltype','Morphology','Immunostaining',\
                                                 'Recording Quality','RecordingComplete',\
                                                 'Restingpotential (mV)','Access Resistance Beginning (MOhm)','Access Resistance End (MOhm)',\
                                                 'HyperStim (pA)','JustSubStim (pA)','Rheobase LPR (pA)','Rheobase SPR (pA)',\
                                                 'Comment']})
       

        self.RecDate,self.RecTime = findDayAndTime(self.OVTable) 
        
        if self.OVSortingTableName is not None and self.CellName in self.OVSortingTable.index:
            self.XFile.loc[0,1] = self.RecDate
            self.XFile.loc[1,1] = self.RecTime
            self.XFile.loc[2,1] = self.OVSortingTable.loc[self.CellName,'Animal Number']
            self.XFile.loc[3,1] = self.OVSortingTable.loc[self.CellName,'Birth']
            self.XFile.loc[4,1] = self.OVSortingTable.loc[self.CellName,'Genotype']
            self.XFile.loc[5,1] = self.OVSortingTable.loc[self.CellName,'Virus Injection']
            self.XFile.loc[6,1] = self.OVSortingTable.loc[self.CellName,'Injection Date']
            self.XFile.loc[7,1] = self.OVSortingTable.loc[self.CellName,'Tissue']
            self.XFile.loc[8,1] = self.OVSortingTable.loc[self.CellName,'Region']
            self.XFile.loc[9,1] = self.OVSortingTable.loc[self.CellName,'Layer']
            self.XFile.loc[10,1] = self.OVSortingTable.loc[self.CellName,'Celltype']
            self.XFile.loc[11,1] = self.OVSortingTable.loc[self.CellName,'Morphology']
            self.XFile.loc[12,1] = self.OVSortingTable.loc[self.CellName,'Immunostaining']
            self.XFile.loc[13,1] = self.OVSortingTable.loc[self.CellName,'RecordingQuality']            
            self.XFile.loc[14,1] = ''.join(self.QualityComplete)            
            self.XFile.loc[15,1] = self.OVSortingTable.loc[self.CellName,'Vrest']
            self.XFile.loc[16,1] = self.OVSortingTable.loc[self.CellName,'Access Beginning']
            self.XFile.loc[17,1] = self.OVSortingTable.loc[self.CellName,'Access End']
            self.XFile.loc[18,1] = int(self.HypoInput)
            self.XFile.loc[19,1] = int(self.JustSubInput)
            self.XFile.loc[20,1] = int(self.APInput)
            self.XFile.loc[21,1] = 'nan'
            self.XFile.loc[22,1] = self.OVSortingTable.loc[self.CellName,'Comment']
        else:
            self.XFile.loc[0,1] = 'NaN'
            self.XFile.loc[1,1] = 'NaN'
            self.XFile.loc[2,1] = 'NaN'
            self.XFile.loc[3,1] = 'NaN'
            self.XFile.loc[4,1] = 'NaN'
            self.XFile.loc[5,1] = 'NaN'
            self.XFile.loc[6,1] = 'NaN'
            self.XFile.loc[7,1] = 'NaN'
            self.XFile.loc[8,1] = 'NaN'
            self.XFile.loc[9,1] = 'NaN'
            self.XFile.loc[10,1] = 'NaN'
            self.XFile.loc[11,1] = 'NaN'
            self.XFile.loc[12,1] = 'NaN'
            self.XFile.loc[13,1] = 'NaN'           
            self.XFile.loc[14,1] = ''.join(self.QualityComplete)            
            self.XFile.loc[15,1] = 'NaN'
            self.XFile.loc[16,1] = 'NaN'
            self.XFile.loc[17,1] = 'NaN'
            self.XFile.loc[18,1] = self.HypoInput
            self.XFile.loc[19,1] = self.JustSubInput
            self.XFile.loc[20,1] = self.APInput
            self.XFile.loc[21,1] = 'nan'
            self.XFile.loc[22,1] = 'NaN' 
        # Write XFile:
        Writer = pd.ExcelWriter('XFile.xls', engine='xlsxwriter')
        self.XFile.to_excel(Writer, sheet_name='Sheet1',na_rep = "NaN",merge_cells=False,header=False, index=False)
        # Formating:
        self.Workbook = Writer.book
        self.Worksheet = Writer.sheets['Sheet1']
        self.Workbook.set_size(540, 740)
#        self.Worksheet.set_column('A:B', 30)
        FormateA = self.Workbook.add_format({'align':'right', 'bold':True})
        FormateB = self.Workbook.add_format({'align':'left'})
        self.Worksheet.set_column('A:A',30, FormateA)
        self.Worksheet.set_column('B:B',30, FormateB)
        
        # Saving:
        Writer.save()
            
        ''' Quality Control JustSub and AP: '''
        self.NewFolders =  [name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(self.NewPath, name))]
        # With APs:
        if 'JustSub' in self.NewFolders:
             self.NamesWithJustNoAPs, self.NamesWithJustAPs = QualityAPs('JustSub',-20)  
             if self.NamesWithJustAPs:
                 i = 0
                 while i < len(self.NamesWithJustAPs):
                     Original = os.getcwd()+'/'+'JustSub'+'/'+self.NamesWithJustAPs[i]
                     destination = os.getcwd()+'/'+'LPR'
                     sh.move(Original,destination)
                     i += 1
        # No APs: 
        if 'LPR' in self.NewFolders:
            self.NamesWithJustNoAPs, self.NamesWithJustAPs = QualityAPs('LPR',-20)  
            if self.NamesWithJustAPs:
                i = 0
                while i < len(self.NamesWithJustNoAPs):
                    Original = os.getcwd()+'/'+'LPR'+'/'+self.NamesWithJustNoAPs[i]
                    destination = os.getcwd()+'/'+'JustSub'
                    sh.move(Original,destination)
                    i += 1
        # To Many APs:
        if 'LPR' in self.NewFolders:
            self.NameWithToManyAps = QualityToManyAPs('LPR',-20)
            os.chdir('LPR')
            self.LPRfiles = [f for f in os.listdir(os.getcwd()) if (f.endswith('.ibw') and f.startswith('ad1'))]
            self.NumLPRFiles = len(self.LPRfiles)
            if not os.path.isdir(os.getcwd()+'/Excluded_ManyAps'):
                os.makedirs(os.getcwd()+'/Excluded_ManyAps')   
            if len(self.NameWithToManyAps) < self.NumLPRFiles or (len(self.NameWithToManyAps) - self.NumLPRFiles) > 5 :
                i = 0
                while i < len(self.NameWithToManyAps):
                    Original = os.getcwd()+'/'+self.NameWithToManyAps[i]
                    destination = os.getcwd()+'/'+'Excluded_ManyAps'
                    sh.move(Original,destination)
                    i += 1 
            os.chdir("..")
            
        # Only XTimes-LPRs with one Trace:
        if self.FiringTraces.size > 0:
            if isinstance(self.XLPR, list):
                self.XFiringTraceFolders = [name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(self.NewPath, name)) and name.startswith('LPR')]
                if self.XFiringTraceFolders[0] == 'LPR':
                    self.XFiringTraceFolders.remove('LPR')
                # Find Number of Traces:
                i = 0
                self.TraceInFiringFolders = [None]*len(self.XFiringTraceFolders)
                while i < len(self.XFiringTraceFolders):
                    Path = os.getcwd()+'/'+self.XFiringTraceFolders[i]
                    self.TraceInFiringFolders[i] = [f for f in os.listdir(Path) if (f.startswith('ad1'))] 
                    if len(self.TraceInFiringFolders[i]) < 3:
                        j = 0
                        while j < len(self.TraceInFiringFolders[i]):
                            OldPath = Path+'/'+self.TraceInFiringFolders[i][j]
                            Destination = os.getcwd()+'/Z_CClampFiles'
                            sh.move(OldPath,Destination)
                            j += 1
                        sh.rmtree((os.getcwd()+'/'+self.XFiringTraceFolders[i]))
                        stringtoremove = self.XFiringTraceFolders[i]
                        self.QualityFiringComplete = self.QualityFiringComplete.replace(stringtoremove, "")
                        print('Removed '+self.XFiringTraceFolders[i])
                    self.QualityFiringComplete = self.QualityFiringComplete.replace(" ,", "")
                      
                    i+=1
                if self.QualityFiringComplete[-2:] == ', ':
                    self.QualityFiringComplete = self.QualityFiringComplete[:-2] 
                self.QualityComplete[-1] = self.QualityFiringComplete
                self.XFile.loc[14,1] = ''.join(self.QualityComplete)   
                        
                            
''' Testing '''
if Testing == 1:
    
    A = SortingCellCharacteristics(Cellname,Conditions)
    B = A.OVTable
    B1 = A.OVSortingTable
    B2 = A.XFile
    B3 = A.QualityComplete
    B4 = A.QualityFiringComplete
#    C = A.XFiringTraceFolders
#    B1 = A.PicInFolders
#    C = A.FiringInput
#    D = A.XLPR

