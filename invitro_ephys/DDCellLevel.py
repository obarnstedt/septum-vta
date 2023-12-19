 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Main Scripts """
import os
import numpy as np
import DDImport 
import ddhelp
from matplotlib import pyplot as plt
import SubthresholdProperties
import SuprathresholdProperties
import ddPlotting
import pandas as pd
from datetime import datetime
from matplotlib import gridspec as gridspec
from collections import Counter

''' For Testing '''
Testing = 0
if Testing == 1:
    cwd = os.getcwd()
    ## Get Path To Data:
    PathToData = 'G:/Users/Hiroshi/_Projects/VTA_ChR2_Petra/SortedData copy 2'
    os.chdir(PathToData)
    
    # Set OVTableName:     
    OVTableName = 'AllCells_Characterisation'
    
     # Set CellName:
    CellName ='HK_230821_01'

#    CellName ='DD_20170613_S2C2'
#    CellName = 'DD_20170607_S3C1'
     
    os.chdir(CellName)   
    ''' Variables/ Thresholds/ ... for Analysis: '''
    Conditions = {'AmpApThres':-20} 
    Conditions['PassiveRange'] =  np.array([-30,30],dtype=np.float)
    Conditions['CurrentAmpsBasic'] =np.array([-100, -70, -50, -30, -20, -10, 10, 20, 30, 50, 100, 200, 300, 400, 500],dtype=np.float)
    #Conditions['CurrentAmpsBasic'] =np.array([-200, -100, -50, -30, -20, -10, 10, 20, 30, 50, 100, 200, 300, 400, 500],dtype=np.float)
    
    Conditions['AnalyseOnlyIR'] = 0
    
    WhichLPR = 3
    Conditions['LPRXTime'] = 'LPR'+str(WhichLPR)
    
    # Values To Calculate: 
    Conditions['ToCalcAll'] = 1
    Conditions['ToCalcPassive'] = 0
    Conditions['ToCalcHypo'] = 0
    Conditions['ToCalcJustSub'] = 0
    Conditions['ToCalcAP'] = 0
    Conditions['ToCalcFiring'] = 0
    
    # Printing/ Showing
    Conditions['PrintShowAll'] = 1   # 0 = None, 1 Print, 2 Show! 
    Conditions['PrintShowPassive'] = 0
    Conditions['PrintShowHypo'] = 0
    Conditions['PrintShowJustSub'] = 0
    Conditions['PrintShowAP'] = 0
    Conditions['PrintShowFiring'] = 0
    Conditions['PrintOV'] = 0

    # Close all open figures:
    plt.close('all')

''' Assisting classes '''
class ImportCellCheat():
    def __init__ (self,ExcelsheetName):
        self.SheetName = ExcelsheetName # As 'string'
        # Import and convert first two columns to dictionary:
        self.XFileImport = pd.read_excel(self.SheetName,index_col=0,header=None)
        self.XDict1 = self.XFileImport.to_dict()
        self.XDict = self.XDict1[1] 
        # Get Values for Characterisation: 
        self.VRest = ddhelp.SearchKeyDic(self.XDict,'Resting')
        self.HypoStim = ddhelp.SearchKeyDic(self.XDict,'HyperStim')
        if self.HypoStim is None:
            self.HypoStim = np.nan
        self.JustSubStim = ddhelp.SearchKeyDic(self.XDict,'JustSubStim')
        if self.JustSubStim is None:
            self.JustSubStim = np.nan
        self.RheoLPR = ddhelp.SearchKeyDic(self.XDict,'LPR')
        if self.RheoLPR is None:
            self.RheoLPR = np.nan
        self.RheoSPR = ddhelp.SearchKeyDic(self.XDict,'SPR')
        self.RecQuality = ddhelp.SearchKeyDic(self.XDict,'Recording Quality')
        if not hasattr(self, 'RecQuality')or self.RecQuality is np.nan:
             self.RecQuality = '?' 
        self.AccessResBeginning = float(ddhelp.SearchKeyDic(self.XDict,'Beginning (MOhm)'))
        self.AccessResEnd = float(ddhelp.SearchKeyDic(self.XDict,'End (MOhm)'))
        
        # Dates:
        self.BirthDate1 = ddhelp.SearchKeyDic(self.XDict,'Birth')
        self.Birthday = 0
        if self.BirthDate1 is not np.nan:
            self.Birthday = 1
            if self.BirthDate1[-1] == ' ':
                self.BirthDate1 = self.BirthDate1[0:-2]
            self.BirthDate = datetime.strptime(self.BirthDate1,'%Y %m %d')

        self.RecDate = datetime.strptime(ddhelp.SearchCompleteKeyDic(self.XDict,'Date'),'%Y %m %d')
        
        
        self.InjectDate =ddhelp.SearchKeyDic(self.XDict,'InjectionDate')
        if hasattr(self, 'InjectDate') and self.InjectDate is not np.float and self.InjectDate is not None :
            self.InjectDate = datetime.strptime(str(self.InjectDate),'%Y %m %d')


        self. AnimalNumber = ddhelp.SearchKeyDic(self.XDict,'Animal Number')
        if not hasattr(self, 'AnimalNumber'):
             self.AnimalNumber = np.nan 
        self.Injection = ddhelp.SearchKeyDic(self.XDict,'Virus Injection')
        if not hasattr(self, 'Injection') or self.Injection is np.nan:
             self.Injection = 'NA' 
        self.Region = ddhelp.SearchKeyDic(self.XDict,'Region')
        if not hasattr(self, 'Region') or self.Region is np.nan:
             self.Region = 'NA' 
        self.Genotype = ddhelp.SearchKeyDic(self.XDict,'Genotype')
        if not hasattr(self, 'Genotype') or self.Genotype is np.nan:
             self.Genotype = 'NA' 
        self.Tissue = ddhelp.SearchKeyDic(self.XDict,'Tissue')
        if not hasattr(self, 'Tissue') or self.Tissue is np.nan:
             self.Tissue = 'NA' 
        self.Region = ddhelp.SearchKeyDic(self.XDict,'Region')
        if not hasattr(self, 'Region') or self.Region is np.nan:
             self.Region = 'NA' 
        self.SubRegion = ddhelp.SearchKeyDic(self.XDict,'Layer')
        if not hasattr(self, 'SubRegion') or self.SubRegion is None or  self.SubRegion is np.nan:
             self.SubRegion = 'NA' 
        if type(self.SubRegion) == int:
            self.SubRegion=str(self.SubRegion)
        self.Celltype = ddhelp.SearchKeyDic(self.XDict,'CellType')
        if not hasattr(self, 'Celltype') or self.Celltype is None or self.Celltype is np.nan:
             self.Celltype = 'NA'   
        self.Morphology = ddhelp.SearchKeyDic(self.XDict,'Morpho')
        if not hasattr(self, 'Morphology')or self.Morphology is None or self.Morphology is np.nan:
             self.Morphology = 'NA'  
        self.Immuno = ddhelp.SearchKeyDic(self.XDict,'Immunostaining')
        if not hasattr(self, 'Immuno')or self.Immuno is None or self.Immuno is np.nan:
             self.Immuno = 'NA'  
        self.Comment = ddhelp.SearchKeyDic(self.XDict,'Comment')
        if not hasattr(self, 'Comment')or self.Comment is None or self.Comment is np.nan:
             self.Comment = 'Na' 
        
        #### Calculations:
        # Recording Calculations:
        self.AccessResistance = self.AccessResBeginning
        self.AccessResistanceChange = self.AccessResEnd - self.AccessResBeginning
    
        # Dates/Times:
        if self.Birthday != 0:
            self.Age = self.RecDate-self.BirthDate
            self.AgeWeeks = self.Age.days/7 # In Weeks
            self.AgeDays = self.Age.days
        else:
            self.Age = np.nan
            self.AgeWeeks = np.nan 
            self.AgeDays = np.nan
            
        if hasattr(self, 'InjectDate') and self.InjectDate is not np.float and self.InjectDate is not None:
            self.Expression = self.RecDate-self.InjectDate
            self.ExpressionDays = self.Expression.days
        else:
            self.ExpressionDays = np.nan


class ImportAndSortIR():
    def __init__ (self,FolderName,APThreshold=0):
        self.FolderName = FolderName # As 'string'
        self.AmpApThres=APThreshold
        #### IR Stimulation: 
            # With find: Subthreshold, JustSub, AP and Firing Trace: 
                # Just Sub Trace = one Trace before AP
                # Firing with higherst Number of APs#
        
        # Import Basic:
        self.Names, self.Waves,self.Times,self.SampFreq,self.RecTime = DDImport.ImportFolder(self.FolderName)
        # Create Stimulus:
        self.Stimulus = [i for i in range(len(self.Waves))]
        count = 0
        for W in self.Waves:
            self.Stimulus[count] = np.zeros(len(self.Waves[count]))
            self.Stimulus[count][int(100*self.SampFreq[count]):int(600*self.SampFreq[count])] = 1
#            print(np.where(self.Stimulus[count]==1))
            count +=1
            
        # Find First Subthreshold Trace: 
        #AmpApThres = 0
        self.StimDiffWave = np.diff(self.Stimulus[0])
        self.StimDiffPoints = np.where(self.StimDiffWave != 0)
        self.StimOnset = np.asarray(self.StimDiffPoints[0][0])
        self.StimOffset = np.asarray(self.StimDiffPoints[0][1])
        self.StimOffset = self.StimOffset+int(1*self.SampFreq[0])

        self.FirstAP = [i for i in range(len(self.Waves))]
        count = 0
        for W in self.Waves:
            self.FirstAP[count] = np.max(self.Waves[count][self.StimOnset:self.StimOffset])
            count +=1 
        self.BasicSubthreshold = [i for i in range(len(self.FirstAP)) if self.FirstAP[i] < self.AmpApThres]
#        self.BasicSubthreshold [3] = 10
#        print(self.BasicSubthreshold)
        # 8ung: BasicSubthreshold not Continous:
        self.DiffBasicSubthreshold = np.diff(self.BasicSubthreshold)
        self.Where = np.where(self.DiffBasicSubthreshold > 1)
#        print(type(self.Where))
        if np.size(self.Where) > 0:
            self.BasicSubthreshold [(int(self.Where[0])+1):] =[]
#            print(self.BasicSubthreshold)

        
        # Hypo, Just Sub and Action Potential Trace:
        self.BasicHypo = 0
        self.BasicJustSub = self.BasicSubthreshold[-1]
        self.BasicAP = self.BasicSubthreshold[-1]+1
#        print(self.BasicJustSub)
        
        # Firing Traces: 
        count = self.BasicAP
        self.BasicFindAP = [None]*(len(self.Waves)-self.BasicAP)
        self.APNums = [None]*(len(self.Waves)-self.BasicAP)
        while count < len(self.Waves):
            self.BasicFindAP[count-self.BasicAP] = SuprathresholdProperties.FindAPs2(self.Times[count][self.StimOnset:self.StimOffset],self.Waves[count][self.StimOnset:self.StimOffset],self.SampFreq[count],2,1,0)
            self.APNums[count-self.BasicAP] = self.BasicFindAP[count-self.BasicAP].APNum
            count +=1
        self.BasicFiring = np.argmax(self.APNums)+self.BasicAP   
        
        #### Basic from IR Protocol:    
        count = 0
        self.BasicSubThresNames = [i for i in range(self.BasicJustSub+1)]
        self.BasicSubThresWaves = [i for i in range(self.BasicJustSub+1)] 
        self.BasicSubThresTimes = [i for i in range(self.BasicJustSub+1)] 
        self.BasicSubThresSampFreq = [i for i in range(self.BasicJustSub+1)]  
        self.BasicSubThresRecTime = [i for i in range(self.BasicJustSub+1)]
        self.BasicSubStimulus= [i for i in range(self.BasicJustSub+1)] 
        
        while count <= self.BasicJustSub:
            self.BasicSubThresNames[count] = self.Names[count]
            self.BasicSubThresWaves[count] = self.Waves[count]
            self.BasicSubThresTimes[count] = self.Times[count]
            self.BasicSubThresSampFreq[count] = self.SampFreq[count]
            self.BasicSubThresRecTime[count] = self.RecTime[count]
            self.BasicSubStimulus[count] = self.Stimulus[count]
            count +=1
            
        self.AnalysisPassiveOn = 'IRStimulation'

class ImportOther():
    def __init__ (self,FolderName,BasicTracesFromIR,AnalsisOnlyBasic = 0):
        self.FolderName = FolderName # as 'string'
        #print(FolderName)
        self.IRTraces = BasicTracesFromIR
        self.AnalsisOnlyBasic = AnalsisOnlyBasic

        self.TracesFromIR = [None]
        if self.FolderName == 'Hypo50':
            self.TracesFromIR =  int(self.IRTraces.BasicHypo)
        if self.FolderName == 'JustSub':
            self.TracesFromIR = int(self.IRTraces.BasicJustSub)
        if self.FolderName.startswith('LPR'):
            self.TracesFromIR = int(self.IRTraces.BasicFiring)
        if self.FolderName == 'LPR':
            self.TracesFromIR = int(self.IRTraces.BasicAP)
        
        
        if os.path.exists(self.FolderName) and self.AnalsisOnlyBasic != 1 :    
            self.AnalysisOn = 'Defined Traces'
            self.Names, self.Waves,self.Times,self.SampFreq,self.RecTime = DDImport.ImportFolder(FolderName)
            # Cut Down to 10 when to Many:
            if len(self.Names) > 10:
                self.Names = self.Names[0:10]
                self.Waves = self.Waves[0:10]
                self.Times = self.Times[0:10]
                self.SampFreq = self.SampFreq[0:10]
                self.RecTime = self.RecTime[0:10]
           
#            print(self.Names[0])
#            print(len(self.Waves[0]))
#            print(type(self.SampFreq))
#            print(type(self.SampFreq[]))
            # Create Stimulus:
            self.Stimulus = [i for i in range(len(self.Names))]
            count = 0
            while count <len(self.Names):
#                print(self.Names[count])
#                print(type(self.SampFreq[count]))
                self.Stimulus[count] = np.zeros(len(self.Waves[count]))
                self.Stimulus[count][int(100*self.SampFreq[count]):int(600*self.SampFreq[count])] = 1
#                print(self.Stimulus[count])
#                print(len(self.Stimulus[count]))
                count +=1
                
            self.NumWaves = count
            
            if self.NumWaves == 1:
                if type(self.Names) == list:
                    self.Names = self.Names[0]
                if type(self.SampFreq) == list:
                    self.SampFreq = self.SampFreq[0]
                if type(self.Stimulus) == list:
                    self.Stimulus = self.Stimulus[0]
                if type(self.Waves) == list:
                    self.Waves = self.Waves[0]
                if type(self.Times) == list:
                    self.Times = self.Times[0]
                if type(self.RecTime) == list:
                    self.RecTime = self.RecTime[0]
        else:
            self.Names = self.IRTraces.Names[self.TracesFromIR]
            self.Waves = self.IRTraces.Waves[self.TracesFromIR]
            self.Times = self.IRTraces.Times[self.TracesFromIR]
            self.SampFreq = self.IRTraces.SampFreq[self.TracesFromIR]
            self.RecTime = self.IRTraces.RecTime[self.TracesFromIR]
            self.AnalysisOn = 'IRStimulation'
            self.NumWaves = 1
            # Create Stimulus:
            self.Stimulus = np.zeros(len(self.Waves))
            self.Stimulus[int(100*self.SampFreq):int(600*self.SampFreq)] = 1   
 
class AnalysisPassive():
    def __init__ (self,Traces,CurrentAmpsBasic,PassiveRange):    
        self.TraceObject = Traces
        self.NumWaves = len(self.TraceObject.BasicSubThresNames)
        self.CurrentAmpsBasic = CurrentAmpsBasic
        self.PassiveRange = PassiveRange
        
        self.SubThresValues = [i for i in range(self.NumWaves)]
        self.SubThresMaxAmplitudes = np.empty(shape=[self.NumWaves]) 
        self.SubThresVStableAmplitudes = np.empty(shape=[self.NumWaves])    
        self.SubThresTauMax = np.empty(shape=[self.NumWaves])
        self.SubThresTauVStable = np.empty(shape=[self.NumWaves])  
          
        count = 0
        while count < self.NumWaves:
            if Testing == 1:
                print(self.TraceObject.BasicSubThresNames[count])
            self.SubThresValues[count] = SubthresholdProperties.Main(\
                          self.TraceObject.BasicSubThresNames[count],\
                          self.TraceObject.BasicSubThresTimes[count],\
                          self.TraceObject.BasicSubThresWaves[count],\
                          self.TraceObject.BasicSubStimulus[count],\
                          self.TraceObject.BasicSubThresSampFreq[count]) 
            
            self.SubThresMaxAmplitudes[count] = self.SubThresValues[count].MaxAmplitude
            self.SubThresVStableAmplitudes[count] = self.SubThresValues[count].VStableAmplitude
            self.SubThresTauMax[count] = self.SubThresValues[count].TauMax
            self.SubThresTauVStable[count] = self.SubThresValues[count].TauVStable
            count +=1
            
        # Calculations IV and Tau: 
        self.IVCalc = SubthresholdProperties.IVCalc(self.CurrentAmpsBasic,self.SubThresMaxAmplitudes,self.SubThresVStableAmplitudes,self.PassiveRange)    
        self.TauCalc = SubthresholdProperties.InstantTauCalc(self.CurrentAmpsBasic,self.SubThresTauMax,self.SubThresTauVStable,self.PassiveRange)
        
        # Passive Values:
        self.Values = SubthresholdProperties.PassiveValues(self.IVCalc,self.TauCalc)     
        self.GroundValues = SubthresholdProperties.PassiveGroundValues(self.SubThresValues,self.TraceObject.BasicSubThresNames)
        
#class Analysis(Module,Class,Files as Object)
class Analysis:
    def __init__ (self,Traces,ModuleName,ClassNameMain,ClassNameGetValues):        
        self.TraceObject = Traces
        self.NumWaves = self.TraceObject.NumWaves
        self.ModuleName = ModuleName
        self.ClassName = ClassNameMain
        self.ClassName1 = ClassNameGetValues
        self.loaded_class1 = ddhelp.class_for_name(self.ModuleName,self.ClassName)
        self.loaded_class2 = ddhelp.class_for_name(self.ModuleName,self.ClassName1)
        
#        print(self.ClassName)
#        print(self.NumWaves)
        
        # Calclations:
        self.Calcs = [i for i in range(self.NumWaves)]
        if self.NumWaves > 1:
            count = 0
            #self.Calcs = [i for i in range(self.NumWaves)]
            while count < self.NumWaves:
                if Testing == 1:
                    print(self.TraceObject.Names[count])
                
                self.Calcs[count] = self.loaded_class1(\
                          self.TraceObject.Names[count],\
                          self.TraceObject.Times[count],\
                          self.TraceObject.Waves[count],\
                          self.TraceObject.Stimulus[count],\
                          self.TraceObject.SampFreq[count])    
                count +=1
        else:
            self.Calcs1 = self.loaded_class1(self.TraceObject.Names, self.TraceObject.Times, self.TraceObject.Waves,  self.TraceObject.Stimulus,  self.TraceObject.SampFreq)
            self.Calcs[0] = self.Calcs1     
        # Get Values:
        self.Values = self.loaded_class2(self.Calcs,self.TraceObject.Names,self.NumWaves) 
        
class CellTable():
    def __init__ (self,CellName,CellInfo,PassiveValues,HypoValues,JustSubValues,APValues,FiringValues):          
        self.CellName = CellName
        self.CellInfo = CellInfo
        self.PassiveValues = PassiveValues
        self.HypoValues = HypoValues
        self.JustSubValues = JustSubValues
        self.APValues = APValues
        self.FiringValues = FiringValues

        # Start of Creating Table:         
        self.index1 = [self.CellName]
        self.columns1 = ['AnimalNumber','Age[D]','Genotype','Injection','Expression[d]','Tissue','Region',\
                   'SubRegion','Celltype','Morphology','ImmunoStaining','RecordingQuality',\
                   'AccessResistance[MOhm]','AccessResistanceChange[MOhm]','RestingPotential[mV]']
                   
        self.CellTable1 = pd.DataFrame(index=self.index1, columns=self.columns1)
        self.CellTable1.loc[self.CellName,'AnimalNumber']=self.CellInfo.AnimalNumber
        self.CellTable1.loc[self.CellName,'Age[D]']=self.CellInfo.AgeDays
        self.CellTable1.loc[self.CellName,'Genotype']=self.CellInfo.Genotype
        self.CellTable1.loc[self.CellName,'Injection']=self.CellInfo.Injection
        self.CellTable1.loc[self.CellName,'Expression[d]']=self.CellInfo.ExpressionDays
        
        self.CellTable1.loc[self.CellName,'Tissue']=self.CellInfo.Tissue
        self.CellTable1.loc[self.CellName,'Region']=self.CellInfo.Region
        self.CellTable1.loc[self.CellName,'SubRegion']=self.CellInfo.SubRegion
        self.CellTable1.loc[self.CellName,'Celltype']=self.CellInfo.Celltype
        self.CellTable1.loc[self.CellName,'Morphology']=self.CellInfo.Morphology
        self.CellTable1.loc[self.CellName,'ImmunoStaining']=self.CellInfo.Immuno
        
        self.CellTable1.loc[self.CellName,'RecordingQuality']=self.CellInfo.RecQuality
        self.CellTable1.loc[self.CellName,'AccessResistance[MOhm]']=self.CellInfo.AccessResBeginning
        self.CellTable1.loc[self.CellName,'AccessResistanceChange[MOhm]']=self.CellInfo.AccessResistanceChange
        self.CellTable1.loc[self.CellName,'RestingPotential[mV]']=self.CellInfo.VRest
        
        ### Acessory Table: 
        self.Acolumns1 = ['NumHypoTraces[#]','NumJustSubTraces[#]','NumAPTraces[#]','NumFiringTraces[#]']
        self.AcessoryCellTable1 = pd.DataFrame(index=self.index1, columns=self.Acolumns1)
        self.AcessoryCellTable1.loc[self.CellName,'NumHypoTraces[#]']=self.CellInfo.NumHypoTraces
        self.AcessoryCellTable1.loc[self.CellName,'NumJustSubTraces[#]']=self.CellInfo.NumJustSubTraces
        self.AcessoryCellTable1.loc[self.CellName,'NumAPTraces[#]']=self.CellInfo.NumAPTraces
        self.AcessoryCellTable1.loc[self.CellName,'NumFiringTraces[#]']=self.CellInfo.NumFiringTraces


        # Passive Values:
        self.index2 = [self.CellName]
        self.columns2 = ['IRmax[MOhm]','IRStable[MOhm]','TauMax[ms]','TauStable[ms]','FastRecHypo',\
                    'FastRecDepo']         
        self.CellTable2 = pd.DataFrame(index=self.index2, columns=self.columns2)
        if self.PassiveValues != 0:
            self.CellTable2.loc[self.CellName,'IRmax[MOhm]']=self.PassiveValues.IRmax
            self.CellTable2.loc[self.CellName,'IRStable[MOhm]']=self.PassiveValues.IRVStable
            self.CellTable2.loc[self.CellName,'TauMax[ms]']=self.PassiveValues.TauMax
            self.CellTable2.loc[self.CellName,'TauStable[ms]']=self.PassiveValues.TauVStable
            self.CellTable2.loc[self.CellName,'FastRecHypo']=self.PassiveValues.FastRecHypo
            self.CellTable2.loc[self.CellName,'FastRecDepo']=self.PassiveValues.FastRecDepo
        else:
            self.CellTable2 = pd.DataFrame(index=self.index2, columns=self.columns2)
            self.CellTable2.loc[self.CellName,'IRmax[MOhm]']=np.nan
            self.CellTable2.loc[self.CellName,'IRStable[MOhm]']=np.nan
            self.CellTable2.loc[self.CellName,'TauMax[ms]']=np.nan
            self.CellTable2.loc[self.CellName,'TauStable[ms]']=np.nan
            self.CellTable2.loc[self.CellName,'FastRecHypo']=np.nan
            self.CellTable2.loc[self.CellName,'FastRecDepo']=np.nan
        
        ### Hypo Values:
        self.index3 = [self.CellName]
        self.columns3 = ['HypoStim[pA]','HypoSagIndex[%]','HypoSagArea[mV^2/ms]',\
                   'HypoReboundIndex[%]','HypoReboundArea[mV^2/ms]',\
                   'HypoReboundAPs[#]']
                   
        self.CellTable3 = pd.DataFrame(index=self.index3, columns=self.columns3)
        if self.HypoValues != 0:
            self.CellTable3.loc[self.CellName,'HypoStim[pA]']=self.CellInfo.HypoStim
            self.CellTable3.loc[self.CellName,'HypoSagIndex[%]']=self.HypoValues.SagIndex.Mean
            self.CellTable3.loc[self.CellName,'HypoSagArea[mV^2/ms]']=self.HypoValues.SagArea.Mean
            self.CellTable3.loc[self.CellName,'HypoReboundIndex[%]']=self.HypoValues.ReboundIndex.Mean
            self.CellTable3.loc[self.CellName,'HypoReboundArea[mV^2/ms]']=self.HypoValues.ReboundArea.Mean
            self.CellTable3.loc[self.CellName,'HypoReboundAPs[#]']=self.HypoValues.ReboundAPs.Mean
        else:
            self.CellTable3.loc[self.CellName,'HypoStim[pA]']=np.nan
            self.CellTable3.loc[self.CellName,'HypoSagIndex[%]']=np.nan
            self.CellTable3.loc[self.CellName,'HypoSagArea[mV^2/ms]']=np.nan
            self.CellTable3.loc[self.CellName,'HypoReboundIndex[%]']=np.nan
            self.CellTable3.loc[self.CellName,'HypoReboundArea[mV^2/ms]']=np.nan
            self.CellTable3.loc[self.CellName,'HypoReboundAPs[#]']=np.nan

        ### Acessory Table: 
        self.Acolumns3 = ['HypoSagTauOff[ms]','HypoReboundTauOff[ms]']
        self.AcessoryCellTable3 = pd.DataFrame(index=self.index1, columns=self.Acolumns3)
        if self.HypoValues != 0:
            self.AcessoryCellTable3.loc[self.CellName,'HypoSagTauOff[ms]']=self.HypoValues.SagTauOff.Mean
            self.AcessoryCellTable3.loc[self.CellName,'HypoReboundTauOff[ms]']=self.HypoValues.ReboundTauOff.Mean
        else:
            self.AcessoryCellTable3.loc[self.CellName,'HypoSagTauOff[ms]']=np.nan
            self.AcessoryCellTable3.loc[self.CellName,'HypoReboundTauOff[ms]']=np.nan
            
        # JustSub Values:
        self.index4 = [self.CellName]
        self.columns4 = ['JustSubStim[pA]','DepoSagIndex[%]','DepoSagArea[mV^2/ms]',\
                   'DepoReboundIndex[%]','DepoReboundAmp[mV]']
                   
        self.CellTable4 = pd.DataFrame(index=self.index4, columns=self.columns4)
        if self.JustSubValues != 0:
            self.CellTable4.loc[self.CellName,'JustSubStim[pA]']=self.CellInfo.JustSubStim
            self.CellTable4.loc[self.CellName,'DepoSagIndex[%]']=self.JustSubValues.SagIndex.Mean
            self.CellTable4.loc[self.CellName,'DepoSagArea[mV^2/ms]']=self.JustSubValues.SagArea.Mean
            self.CellTable4.loc[self.CellName,'DepoReboundIndex[%]']=self.JustSubValues.ReboundIndex.Mean
            self.CellTable4.loc[self.CellName,'DepoReboundAmp[mV]']=self.JustSubValues.ReboundAmplitude.Mean
        else:
            self.CellTable4.loc[self.CellName,'JustSubStim[pA]']=np.nan
            self.CellTable4.loc[self.CellName,'DepoSagIndex[%]']=np.nan
            self.CellTable4.loc[self.CellName,'DepoSagArea[mV^2/ms]']=np.nan
            self.CellTable4.loc[self.CellName,'DepoReboundIndex[%]']=np.nan
            self.CellTable4.loc[self.CellName,'DepoReboundAmp[mV]']=np.nan
            
        ### Acessory Table: 
        self.Acolumns4 = ['DepoSagTauOff[ms]','DepoReboundArea[mV^2/ms]','DepoReboundTauOff[ms]']
        self.AcessoryCellTable4 = pd.DataFrame(index=self.index1, columns=self.Acolumns4)
        if self.JustSubValues != 0:
            self.AcessoryCellTable4.loc[self.CellName,'DepoSagTauOff[ms]']=self.JustSubValues.SagTauOff.Mean
            self.AcessoryCellTable4.loc[self.CellName,'DepoReboundArea[mV^2/ms]']=self.JustSubValues.ReboundArea.Mean
            self.AcessoryCellTable4.loc[self.CellName,'DepoReboundTauOff[ms]']=self.JustSubValues.ReboundTauOff.Mean
        else:
            self.AcessoryCellTable4.loc[self.CellName,'DepoSagTauOff[ms]']=np.nan
            self.AcessoryCellTable4.loc[self.CellName,'DepoReboundArea[mV^2/ms]']=np.nan
            self.AcessoryCellTable4.loc[self.CellName,'DepoReboundTauOff[ms]']=np.nan
        
        ### AP Values:
        self.index5 = [self.CellName]
        self.columns5 = ['Rheobase[pA]','APType','NumAPs',\
                   'APBaseAmp[mV]','Threshold[mV]','APThresAmp[mV]','TimeToPeak[ms]',\
                   'HalfWidth[ms]','SlopeRise[mV/ms]','SlopeDecay[mV/ms]','Latency[ms]',\
                   'AHPType','AHPArea[mV^2/ms]','fAHP[Vm]','fAHPTtP[ms]','ADP[Vm]','ADPTtP[ms]','mAHP[Vm]',\
                   'mAHPTtP[ms]','sAHPAmp[mV]','BurstDuration[ms]','BurstArea[mV^2/ms]','BurstAHP[mV]']
                   
        self.CellTable5 = pd.DataFrame(index=self.index5, columns=self.columns5)
        if self.APValues != 0: 
            self.CellTable5.loc[self.CellName,'Rheobase[pA]']=self.CellInfo.RheoLPR
            self.CellTable5.loc[self.CellName,'APType']=self.APValues.APType
            self.CellTable5.loc[self.CellName,'NumAPs']=self.APValues.NumAPs.Mean
            self.CellTable5.loc[self.CellName,'APBaseAmp[mV]']=self.APValues.APAmplitudeBaseline.Mean
            self.CellTable5.loc[self.CellName,'Threshold[mV]']=self.APValues.Threshold.Mean
            self.CellTable5.loc[self.CellName,'APThresAmp[mV]']=self.APValues.APAmplitudeThreshold.Mean
            self.CellTable5.loc[self.CellName,'TimeToPeak[ms]']=self.APValues.APTtP.Mean
            self.CellTable5.loc[self.CellName,'HalfWidth[ms]']=self.APValues.HalfWidth.Mean
            self.CellTable5.loc[self.CellName,'SlopeRise[mV/ms]']=self.APValues.SlopeRise.Mean
            self.CellTable5.loc[self.CellName,'SlopeDecay[mV/ms]']=self.APValues.SlopeDecay.Mean
            self.CellTable5.loc[self.CellName,'Latency[ms]']=self.APValues.Latency.Mean
            self.CellTable5.loc[self.CellName,'AHPType']=self.APValues.AHPType
            self.CellTable5.loc[self.CellName,'AHPArea[mV^2/ms]']=self.APValues.AHPArea.Mean
            self.CellTable5.loc[self.CellName,'fAHP[Vm]']=self.APValues.fAHPVm.Mean
            self.CellTable5.loc[self.CellName,'fAHPTtP[ms]']=self.APValues.fAHPTtP.Mean
            self.CellTable5.loc[self.CellName,'ADP[Vm]']=self.APValues.ADPVm.Mean
            self.CellTable5.loc[self.CellName,'ADPTtP[ms]']=self.APValues.ADPTtP.Mean
            self.CellTable5.loc[self.CellName,'mAHP[Vm]']=self.APValues.mAHPVm.Mean
            self.CellTable5.loc[self.CellName,'mAHPTtP[ms]']=self.APValues.mAHPTtP.Mean
            self.CellTable5.loc[self.CellName,'sAHPAmp[mV]']=self.APValues.sAHPAmp.Mean
            self.CellTable5.loc[self.CellName,'BurstDuration[ms]']=self.APValues.BurstDuration.Mean
            self.CellTable5.loc[self.CellName,'BurstArea[mV^2/ms]']=self.APValues.BurstArea.Mean
            self.CellTable5.loc[self.CellName,'BurstAHP[mV]']=self.APValues.BurstAHPVm.Mean
        else:
            self.CellTable5.loc[self.CellName,'Rheobase[pA]']=np.nan
            self.CellTable5.loc[self.CellName,'APType']=np.nan
            self.CellTable5.loc[self.CellName,'NumAPs']=np.nan
            self.CellTable5.loc[self.CellName,'APBaseAmp[mV]']=np.nan
            self.CellTable5.loc[self.CellName,'Threshold[mV]']=np.nan
            self.CellTable5.loc[self.CellName,'APThresAmp[mV]']=np.nan
            self.CellTable5.loc[self.CellName,'TimeToPeak[ms]']=np.nan
            self.CellTable5.loc[self.CellName,'HalfWidth[ms]']=np.nan
            self.CellTable5.loc[self.CellName,'SlopeRise[mV/ms]']=np.nan
            self.CellTable5.loc[self.CellName,'SlopeDecay[mV/ms]']=np.nan
            self.CellTable5.loc[self.CellName,'Latency[ms]']=np.nan
            self.CellTable5.loc[self.CellName,'AHPType']=np.nan
            self.CellTable5.loc[self.CellName,'AHPArea[mV^2/ms]']=np.nan
            self.CellTable5.loc[self.CellName,'fAHP[Vm]']=np.nan
            self.CellTable5.loc[self.CellName,'fAHPTtP[ms]']=np.nan
            self.CellTable5.loc[self.CellName,'ADP[Vm]']=np.nan
            self.CellTable5.loc[self.CellName,'ADPTtP[ms]']=np.nan
            self.CellTable5.loc[self.CellName,'mAHP[Vm]']=np.nan
            self.CellTable5.loc[self.CellName,'mAHPTtP[ms]']=np.nan
            self.CellTable5.loc[self.CellName,'sAHPAmp[mV]']=np.nan
            self.CellTable5.loc[self.CellName,'BurstDuration[ms]']=np.nan
            self.CellTable5.loc[self.CellName,'BurstArea[mV^2/ms]']=np.nan
            self.CellTable5.loc[self.CellName,'BurstAHP[mV]']=np.nan
        
        ### Acessory Table: 
        self.Acolumns5 = ['APBaseAmpChange','ThresholdChange','APThresAmpChange','TimeToPeakChange',\
                          'HalfWidthChange','SlopeRiseChange','SlopeDecayChange','BurstAHPChange']
        self.AcessoryCellTable5 = pd.DataFrame(index=self.index1, columns=self.Acolumns5)
        if self.APValues != 0: 
            self.AcessoryCellTable5.loc[self.CellName,'APBaseAmpChange']=self.APValues.APAmpBaseChange.Mean
            self.AcessoryCellTable5.loc[self.CellName,'ThresholdChange']=self.APValues.ThresChange.Mean
            self.AcessoryCellTable5.loc[self.CellName,'APThresAmpChange']=self.APValues.APAmpThesChange.Mean
            self.AcessoryCellTable5.loc[self.CellName,'TimeToPeakChange']=self.APValues.APTtPChange.Mean
            self.AcessoryCellTable5.loc[self.CellName,'HalfWidthChange']=self.APValues.HalfWidthChange.Mean
            self.AcessoryCellTable5.loc[self.CellName,'SlopeRiseChange']=self.APValues.SlopeRiseChange.Mean
            self.AcessoryCellTable5.loc[self.CellName,'SlopeDecayChange']=self.APValues.SlopeDecayChange.Mean
            self.AcessoryCellTable5.loc[self.CellName,'BurstAHPChange']=self.APValues.BurstAHPVmChange.Mean
        else:
            self.AcessoryCellTable5.loc[self.CellName,'APBaseAmpChange']=np.nan
            self.AcessoryCellTable5.loc[self.CellName,'ThresholdChange']=np.nan
            self.AcessoryCellTable5.loc[self.CellName,'APThresAmpChange']=np.nan
            self.AcessoryCellTable5.loc[self.CellName,'TimeToPeakChange']=np.nan
            self.AcessoryCellTable5.loc[self.CellName,'HalfWidthChange']=np.nan
            self.AcessoryCellTable5.loc[self.CellName,'SlopeRiseChange']=np.nan
            self.AcessoryCellTable5.loc[self.CellName,'SlopeDecayChange']=np.nan
            self.AcessoryCellTable5.loc[self.CellName,'BurstAHPChange']=np.nan
        
        
        ### Firing Values:
        self.index6 = [self.CellName]
        self.columns6 = ['FiringStim[pA]','FiringType','NumAPsFiring','FiringDuration[ms]','FirstSpikeLatency[ms]',\
                   'MeanFiringFrequency[Hz]','FirstFiringFrequency[Hz]','FastFiringFreqAccomodation','SlowFiringFreqAccomodation','FiringFreqIndex',\
                   'FirstSpikeAmpBase[mV]','APBaseFastAdaption','APBaseSlowAdaption','APBaseAdaptionIndex',\
                   'FirstSpikeAmpThres[mV]','APThresFastAdaption','APThresSlowAdaption','APThresAdaptionIndex']
                   
        self.CellTable6 = pd.DataFrame(index=self.index6, columns=self.columns6)
        if self.FiringValues != 0: 
            self.CellTable6.loc[self.CellName,'FiringStim[pA]']=self.CellInfo.FiringStim
            self.CellTable6.loc[self.CellName,'FiringType']=self.FiringValues.FiringType
            self.CellTable6.loc[self.CellName,'NumAPsFiring']=self.FiringValues.NumAPs.Mean
            
            self.CellTable6.loc[self.CellName,'FiringDuration[ms]']=self.FiringValues.FiringDuration.Mean
            self.CellTable6.loc[self.CellName,'FirstSpikeLatency[ms]']=self.FiringValues.FirstSpikeLatency.Mean
            
            self.CellTable6.loc[self.CellName,'MeanFiringFrequency[Hz]']=self.FiringValues.FiringFrequency.Mean
            self.CellTable6.loc[self.CellName,'FirstFiringFrequency[Hz]']=self.FiringValues.FirstFiringFreq.Mean
            self.CellTable6.loc[self.CellName,'FastFiringFreqAccomodation']=self.FiringValues.FiringFreqFastAcc.Mean
            self.CellTable6.loc[self.CellName,'SlowFiringFreqAccomodation']=self.FiringValues.FiringFreqSlowAcc.Mean
            self.CellTable6.loc[self.CellName,'FiringFreqIndex']=self.FiringValues.FiringFreqIndex.Mean
            
            self.CellTable6.loc[self.CellName,'FirstSpikeAmpBase[mV]']=self.FiringValues.FirstSpikeAmpBase.Mean
            self.CellTable6.loc[self.CellName,'APBaseSlowAdaption']=self.FiringValues.APBaseSlowAdaption.Mean
            self.CellTable6.loc[self.CellName,'APBaseFastAdaption']=self.FiringValues.APBaseFastAdaption.Mean
            self.CellTable6.loc[self.CellName,'APBaseAdaptionIndex']=self.FiringValues.APBaseAdaptionIndex.Mean
            
            self.CellTable6.loc[self.CellName,'FirstSpikeAmpThres[mV]']=self.FiringValues.FirstSpikeAmpThres.Mean
            self.CellTable6.loc[self.CellName,'APThresSlowAdaption']=self.FiringValues.APThresSlowAdaption.Mean
            self.CellTable6.loc[self.CellName,'APThresFastAdaption']=self.FiringValues.APThresFastAdaption.Mean
            self.CellTable6.loc[self.CellName,'APThresAdaptionIndex']=self.FiringValues.APThresAdaptionIndex.Mean
        else:
            self.CellTable6.loc[self.CellName,'FiringStim[pA]']=np.nan
            self.CellTable6.loc[self.CellName,'FiringType']=np.nan
            self.CellTable6.loc[self.CellName,'NumAPs']=np.nan
            
            self.CellTable6.loc[self.CellName,'FiringDuration[ms]']=np.nan
            self.CellTable6.loc[self.CellName,'FirstSpikeLatency[ms]']=np.nan
            
            self.CellTable6.loc[self.CellName,'FiringFrequency[Hz]']=np.nan
            self.CellTable6.loc[self.CellName,'FiringFreqAccomodation']=np.nan
            self.CellTable6.loc[self.CellName,'FiringFreqFastAcc']=np.nan
            self.CellTable6.loc[self.CellName,'FiringFreqIndex']=np.nan
            
            self.CellTable6.loc[self.CellName,'FirstSpikeAmpBase[mV]']=np.nan
            self.CellTable6.loc[self.CellName,'APBaseSlowAdaption']=np.nan
            self.CellTable6.loc[self.CellName,'APBaseFastAdaption']=np.nan
            self.CellTable6.loc[self.CellName,'APBaseAdaptionIndex']=np.nan
            
            self.CellTable6.loc[self.CellName,'FirstSpikeAmpThres[mV]']=np.nan
            self.CellTable6.loc[self.CellName,'APThresSlowAdaption']=np.nan
            self.CellTable6.loc[self.CellName,'APThresFastAdaption']=np.nan
            self.CellTable6.loc[self.CellName,'APThresAdaptionIndex']=np.nan
            
        self.index7 = [self.CellName]
        self.columns7 = ['Comment']
        self.CellTable7 = pd.DataFrame(index=self.index7, columns=self.columns7)
        self.CellTable7.loc[self.CellName,'Comment']=self.CellInfo.Comment
        
        self.CellTable = pd.concat([self.CellTable1,self.CellTable2,self.CellTable3,self.CellTable4,self.CellTable5,self.CellTable6,self.CellTable7],\
                              ignore_index=False, axis=1)
        # Acessory Table:
        self.AcessoryCellTable = pd.concat([self.AcessoryCellTable1,self.AcessoryCellTable3,self.AcessoryCellTable4,self.AcessoryCellTable5],\
                              ignore_index=False, axis=1)
        
class PlotMainFigure:
    def __init__ (self,CellName,CellInfo,Pics,\
                  PassiveValues,PassiveCalcs,IVCalcs,\
                  HypoValues,HypoCalcs,\
                  JustSubValues,JustSubCalcs,\
                  APValues,APCalcs,APToTake,\
                  FiringValues,FiringCalcs,FiringToTake,\
                  PrintOV = 0): 
        ''' Get Values and Length,... '''
        self.CellName = CellName
        self.CellInfo = CellInfo 
        self.Pics = Pics
        self.PrintShow = PrintOV
        self.PassiveValues = PassiveValues
        self.PassiveCalcs = PassiveCalcs
        self.LPassiveTrace = len(self.PassiveCalcs)
        
        # Commen:
        self.FSAxes = 6
        self.FSAnnots  = 6
        self.FSPlotHeaders = 7
        self.Markersize = 5
        self.FSMainEphys = 8
        self.FSQuality = 6
        self.FSCellName = 12
        self.FSCellInfo = 8
        self.ThicknessLines = 0.75
        # Passive Traces:
        self.PassiveWaves = [None] * self.LPassiveTrace
        self.LPassiveWaves = [None] * self.LPassiveTrace
        i = 0
        while i < self.LPassiveTrace:
            self.PassiveWaves[i] = ddPlotting.ExtractPlotting(self.PassiveCalcs[i]) 
            self.LPassiveWaves[i] = len(self.PassiveWaves[i].Waves)
            i +=1

        self.IVCalcs = IVCalcs
        self.IVWaves = ddPlotting.ExtractPlotting(self.IVCalcs) 
        self.LIVWaves = len(self.IVWaves.Waves)
        
        # Hypo Traces:
        self.HypoValues = HypoValues
        self.HypoCalcs = HypoCalcs
        self.LHypoTrace = len(self.HypoCalcs)
        self.HypoTraces = [None] * self.LHypoTrace
        self.LHypoWaves = [None] * self.LHypoTrace
        i = 0
        while i < self.LHypoTrace:
            self.HypoTraces[i] = ddPlotting.ExtractPlotting(self.HypoCalcs[i]) 
            self.LHypoWaves[i] = len(self.HypoTraces[i].Waves)
            i +=1
            
        # Just Sub Traces:
        self.JustSubValues = JustSubValues
        self.JustSubCalcs = JustSubCalcs
        self.LJustSubTrace = len(self.JustSubCalcs)
        self.JustSubTraces = [None] * self.LJustSubTrace
        self.LJustSubWaves = [None] * self.LJustSubTrace
        i = 0
        while i < self.LJustSubTrace:
            self.JustSubTraces[i] = ddPlotting.ExtractPlotting(self.JustSubCalcs[i]) 
            self.LJustSubWaves[i] = len(self.JustSubTraces[i].Waves)
            i +=1
            
        # AP Traces:
        self.APValues = APValues
        self.APCalcs = APCalcs
        self.APToTake = APToTake
        self.LAPTrace = len(self.APCalcs)
        self.APTraces = [None] * self.LAPTrace
        self.LAPWaves = [None] * self.LAPTrace
        i = 0
        while i < self.LAPTrace:
            self.APTraces[i] = ddPlotting.ExtractPlotting(self.APCalcs[i]) 
            self.LAPWaves[i] = len(self.APTraces[i].Waves)
            i +=1 
            
        # FringTraces:         
        self.FiringValues = FiringValues
        self.FiringCalcs = FiringCalcs
        self.FiringToTake = FiringToTake
        self.LFiringTrace = len(self.FiringCalcs)
        self.FiringTraces = [None] * self.LFiringTrace
        self.LFiringWaves = [None] * self.LFiringTrace
        i = 0
        while i < self.LFiringTrace:
            self.FiringTraces[i] = ddPlotting.ExtractPlotting(self.FiringCalcs[i]) 
            self.LFiringWaves[i] = len(self.FiringTraces[i].Waves)
            i +=1 

        ''' Figure '''
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
            self.Figure = plt.figure()
            self.Figure.set_size_inches(11.69, 8.27, forward=True)
        if self.PrintShow == 1:
            self.Figure.set_dpi(600)
        
        ''' Plot Subthreshold: '''
        self.gsSubThres = gridspec.GridSpec(7,3, height_ratios=[2,1,0.25,1,0.5,1,0.05])
        self.gsSubThres.update(left=0.35, bottom= 0.01, top = 0.91, right=0.65, wspace=0.2, hspace=0.2)
        
        ### Plot IV: 
        self.AxIV = plt.subplot(self.gsSubThres[0,:])
        self.IVPlot = [None]*self.LIVWaves
        self.IVYLIMIT = np.zeros((2, 2))
        j = 0
        while j < self.LIVWaves:
            if  self.IVWaves.Waves[j].get_marker() == '.' or  self.IVWaves.Waves[j].get_marker() == 'o':
                    msset = self.Markersize
            else:
                msset = 0.5
            # Printing:    
            self.IVPlot[j] = self.AxIV.plot(self.IVWaves.Waves[j].get_data()[0], self.IVWaves.Waves[j].get_data()[1], self.IVWaves.Waves[j].get_marker(), linestyle = self.IVWaves.Waves[j].get_linestyle(),linewidth = self.ThicknessLines, color = self.IVWaves.Waves[j].get_color(), ms = msset) 
            
            if j == 0: 
                self.IVYLIMIT[0][0] = np.max(self.IVWaves.Waves[j].get_data()[1])
                self.IVYLIMIT[1][0] = np.min(self.IVWaves.Waves[j].get_data()[1])
            if j == 4:
                self.IVYLIMIT[0][1] = np.max(self.IVWaves.Waves[j].get_data()[1])
                self.IVYLIMIT[1][1] = np.min(self.IVWaves.Waves[j].get_data()[1])
            j +=1
        self.AxIV.spines['left'].set_position('zero')
        self.AxIV.spines['bottom'].set_position('zero')
        # Eliminate upper and right axes
        self.AxIV.spines['right'].set_color('none')
        self.AxIV.spines['top'].set_color('none')
        # Show ticks in the left and lower axes only
        self.AxIV.xaxis.set_ticks_position('bottom')
        self.AxIV.yaxis.set_ticks_position('left')    
        # Set Labels and Sizes:
        self.AxIV.set_ylabel('mV',fontsize = self.FSAxes, rotation = 90)
        plt.setp(self.AxIV.get_yticklabels(), fontsize = self.FSAxes)
        self.AxIV.set_xlabel('pA',fontsize = self.FSAxes)
        plt.setp(self.AxIV.get_xticklabels(), fontsize = self.FSAxes) 
        self.AxIV.set_ylim([np.min(self.IVYLIMIT[1])+(0.2*np.min(self.IVYLIMIT[1])),np.max(self.IVYLIMIT[0])+0.2*np.max(self.IVYLIMIT[0])])
        self.AxIV.set_xlim([np.min(self.IVWaves.Waves[0].get_data()[0]),np.max(self.IVWaves.Waves[0].get_data()[0])])
        # Plot Legend: 
        self.AxIV.legend([self.IVWaves.Waves[4],self.IVWaves.Waves[7],self.IVWaves.Waves[0],self.IVWaves.Waves[3]],['Max Res','IR Max','Stable Res','IR Stable'],loc='lower right',fontsize = self.FSAnnots,frameon=False)
        # Plot Texts:
        self.Textstr = 'Vrest = %.2f mV\nIR Max = %.2f MOhm\nTau Max = %.2f ms\nIR Stable = %.2f MOhm\nTau Stable = %.2f ms' % (self.CellInfo.VRest, self.PassiveValues.IRmax, self.PassiveValues.TauMax, self.PassiveValues.IRVStable, self.PassiveValues.TauVStable)
        self.AxIV.text(0.05, 0.95, self.Textstr, transform=self.AxIV.transAxes, fontsize=self.FSAnnots, verticalalignment='top')
        # Plot Title:
        self.AxIV.set_title('IV Curve and Passive Properties:',fontsize = self.FSPlotHeaders,fontweight='bold',loc='left')
  
########Plot Main Title for Ephys and Quality Controls:
        self.QualityTextList = [None]*2
        self.QualityTextList[0]='RecordingQuality: '+self.CellInfo.RecQuality
        self.QualityTextList[1]=', Rs from %2.f to %2.f MOhm' %(self.CellInfo.AccessResistance, self.CellInfo.AccessResEnd)
        self.QualityText = self.QualityTextList[0]+self.QualityTextList[1]
        self.EphysName = 'Electrophysiological Profile of '+self.CellName +':'
        plt.figtext(0.33, 0.97, self.EphysName, transform=self.AxIV.transAxes, fontsize=self.FSMainEphys, ha='left', va = 'top',fontweight='bold')
        plt.figtext(0.33, 0.95, self.QualityText, transform=self.AxIV.transAxes, fontsize=self.FSQuality, ha='left', va = 'top')  

        ### Plot Passive Waves:         
        self.AxPassive = plt.subplot(self.gsSubThres[1,:])
        self.PassivePlot = [None]*self.LPassiveTrace
        i = 0
        while i < self.LPassiveTrace:
            j = 0
            while j < self.LPassiveWaves[i]:
                if  self.PassiveWaves[i].Waves[j].get_marker() == '.' or  self.PassiveWaves[i].Waves[j].get_marker() == 'o':
                        msset = self.Markersize
                else:
                    msset = 0.5
                # Printing:  
                k = 0
                while k < self.LPassiveTrace:
                    self.PassivePlot = self.AxPassive.plot(self.PassiveWaves[k].Waves[j].get_data()[0], self.PassiveWaves[k].Waves[j].get_data()[1], self.PassiveWaves[k].Waves[j].get_marker(), linestyle = self.PassiveWaves[k].Waves[j].get_linestyle(), linewidth = self.ThicknessLines, color = self.PassiveWaves[k].Waves[j].get_color(), ms = msset) 
                    k +=1
                j +=1
            i +=1
        # Eliminate upper and right axes
        self.AxPassive.spines['right'].set_color('none')
        self.AxPassive.spines['top'].set_color('none')
        # Set Labels and Sizes:
        self.AxPassive.set_ylabel('mV',fontsize = self.FSAxes, rotation = 90)
        plt.setp(self.AxPassive.get_yticklabels(), fontsize = self.FSAxes)
        self.AxPassive.set_xticklabels([])
        self.AxPassive.set_xlim(self.PassiveWaves[0].Waves[0].get_data()[0][0],self.PassiveWaves[0].Waves[0].get_data()[0][-1])
        self.AxPassive.set_title('Traces  Passive Properties',fontsize = self.FSPlotHeaders,y = 0.95,fontweight='bold',loc='left')
        
        self.AxPassive.legend([self.PassiveWaves[0].Waves[1],self.PassiveWaves[0].Waves[2],self.PassiveWaves[0].Waves[3],\
                               self.PassiveWaves[0].Waves[4],self.PassiveWaves[0].Waves[5],self.PassiveWaves[0].Waves[6],self.PassiveWaves[0].Waves[7],\
                               self.PassiveWaves[0].Waves[8]],\
                               ['Baseline','VStable','MaxResponse','Rebound','TauVstable','TauMax','SagArea','ReboundArea','ReboundAPs'],\
                               loc=9, bbox_to_anchor=(0.5, -0.1),fontsize = self.FSAnnots, ncol=5,prop={'size':4},frameon=False)
                                #borderpad=0.75,handlelength=0.5,labelspacing=0.5)
        
        ### Plot JustSub:         
        self.AxJustSub = plt.subplot(self.gsSubThres[3,:])
        i = 0
        while i < self.LJustSubTrace:
            j = 0
            while j < self.LJustSubWaves[i]:
                if  self.JustSubTraces[i].Waves[j].get_marker() == '.' or  self.JustSubTraces[i].Waves[j].get_marker() == 'o':
                        msset = self.Markersize
                else:
                    msset = 0.5
                # Printing:
                k = 0
                while k < self.LJustSubTrace:
                    self.AxJustSub.plot(self.JustSubTraces[k].Waves[j].get_data()[0], self.JustSubTraces[k].Waves[j].get_data()[1], self.JustSubTraces[k].Waves[j].get_marker(), linestyle = self.JustSubTraces[k].Waves[j].get_linestyle(), linewidth = self.ThicknessLines, color = self.JustSubTraces[k].Waves[j].get_color(), ms = msset) 
                    k +=1
                j +=1
            i +=1
        # Eliminate upper and right axes
        self.AxJustSub.spines['right'].set_color('none')
        self.AxJustSub.spines['top'].set_color('none')
        # Set Labels and Sizes:
        self.AxJustSub.set_ylabel('mV',fontsize = self.FSAxes, rotation = 90)
        plt.setp(self.AxJustSub.get_yticklabels(), fontsize = self.FSAxes)
        self.AxJustSub.set_xticklabels([])
        self.AxJustSub.set_xlim(self.JustSubTraces[0].Waves[0].get_data()[0][0],self.JustSubTraces[0].Waves[0].get_data()[0][-1])
        self.AxJustSub.set_title('Traces  Just-Subthreshold Properties:',fontsize = self.FSPlotHeaders,y = 0.9,fontweight='bold',loc='left')

        ### Plot Hypo:         
        self.AxHypo = plt.subplot(self.gsSubThres[5,:])
        i = 0
        while i < self.LHypoTrace:
            j = 0
            while j < self.LHypoWaves[i]:
                if  self.HypoTraces[i].Waves[j].get_marker() == '.' or  self.HypoTraces[i].Waves[j].get_marker() == 'o':
                        msset = self.Markersize
                else:
                    msset = 0.5
                # Printing:   
                k = 0
                while k < self.LHypoTrace:
                    self.AxHypo.plot(self.HypoTraces[k].Waves[j].get_data()[0], self.HypoTraces[k].Waves[j].get_data()[1], self.HypoTraces[k].Waves[j].get_marker(), linestyle = self.HypoTraces[k].Waves[j].get_linestyle(), linewidth = self.ThicknessLines, color = self.HypoTraces[k].Waves[j].get_color(), ms = msset) 
                    k +=1
                j +=1
            i +=1
        # Eliminate upper and right axes
        self.AxHypo.spines['right'].set_color('none')
        self.AxHypo.spines['top'].set_color('none')
        # Set Labels and Sizes:
        self.AxHypo.set_ylabel('mV',fontsize = self.FSAxes, rotation = 90)
        plt.setp(self.AxHypo.get_yticklabels(), fontsize = self.FSAxes)
        self.AxHypo.set_xlabel('ms',fontsize = self.FSAxes,labelpad=0.5)
        plt.setp(self.AxHypo.get_xticklabels(), fontsize = self.FSAxes) 
        self.AxHypo.set_xlim(self.HypoTraces[0].Waves[0].get_data()[0][0],self.HypoTraces[0].Waves[0].get_data()[0][-1])
        self.AxHypo.set_title('Traces  Hypoerpolarisation Properties:',fontsize = self.FSPlotHeaders, y = 0.95,fontweight='bold',loc='left')

        # Plot Data
        self.axTextHypo = plt.subplot(self.gsSubThres[4,0])
        self.axTextHypo.axis('off')
        self.axTextHypo.set_title('Hyperpolarisation',fontsize = self.FSPlotHeaders,y=0.75,fontweight='bold')
        self.TextstrHypo = 'SagIndex = %.2f %%\nSagArea = %.0f mV$^2$/ms \nFastRectification = %.2f' \
        % (self.HypoValues.SagIndex.Mean, self.HypoValues.SagArea.Mean, self.PassiveValues.FastRecHypo)
        self.axTextHypo.text(0.05, 0.80, self.TextstrHypo, transform=self.axTextHypo.transAxes, fontsize=self.FSAnnots, verticalalignment='top')
        
        self.axTextDepo = plt.subplot(self.gsSubThres[4,1])
        self.axTextDepo.axis('off')
        self.axTextDepo.set_title('Depolarisation',fontsize = self.FSPlotHeaders,y=0.75,fontweight='bold')
        self.TextstrDepo = 'SagIndex = %.2f %%\nSagArea = %.0f mV$^2$/ms \nFastRectification = %.2f' \
        % (self.JustSubValues.SagIndex.Mean, self.JustSubValues.SagArea.Mean, self.PassiveValues.FastRecDepo)
        self.axTextDepo.text(0.05, 0.80, self.TextstrDepo, transform=self.axTextDepo.transAxes, fontsize=self.FSAnnots, verticalalignment='top')

        self.axTextRebound = plt.subplot(self.gsSubThres[4,2])
        self.axTextRebound.axis('off')
        self.axTextRebound.set_title('Rebound',fontsize = self.FSPlotHeaders,y=0.75,fontweight='bold')
        self.TextstrRebound = 'ReboundIndex = %.2f %%\nReboundAPs = %.0f #\nDepo Rebound = %.2f %%\nDepo RebAmp = %.2f mV' \
        % (self.HypoValues.ReboundIndex.Mean, self.HypoValues.ReboundAPs.Mean,  self.JustSubValues.ReboundIndex.Mean, self.JustSubValues.ReboundAmplitude.Mean)
        self.axTextRebound.text(0.05, 0.80, self.TextstrRebound, transform=self.axTextRebound.transAxes, fontsize=self.FSAnnots, verticalalignment='top')

        ''' Plot APs and Firing: '''
        self.gsSupThres = gridspec.GridSpec(12,3, height_ratios=[1,1,1,1,1,0.75,1,1,1,1,1,1]) #, width_ratios=[2,1,1])
        self.gsSupThres.update(left=0.69, bottom= 0.04, top = 0.91, right=0.99, wspace=0.35, hspace=0.2)
        
        ####### Plot APs
        # Set Subplots:
        if len(self.APToTake) >=6:
            self.LAP = 6
        else:
            self.LAP = len(self.APToTake) 
        self.axAPPlots = [None]*self.LAP
        self.axAPPlots[0] = plt.subplot(self.gsSupThres[0:3,0:2])
        i = 1
        while i <self.LAP:
            self.axAPPlots[i] = plt.subplot(self.gsSupThres[i-1,2])
            i +=1  
        # Plotting: 
        i = 0
        while i < self.LAP:
            j = 0
            while j < self.LAPWaves[self.APToTake[i]]:
                if  self.APTraces[self.APToTake[i]].Waves[j].get_marker() == '.' or  self.APTraces[self.APToTake[i]].Waves[j].get_marker() == 'o':
                        msset = self.Markersize
                else:
                    msset = 0.5
                # Printing:   
                self.axAPPlots[i].plot(self.APTraces[self.APToTake[i]].Waves[j].get_data()[0], self.APTraces[self.APToTake[i]].Waves[j].get_data()[1], self.APTraces[self.APToTake[i]].Waves[j].get_marker(), linestyle = self.APTraces[self.APToTake[i]].Waves[j].get_linestyle(), linewidth = self.ThicknessLines, color = self.APTraces[self.APToTake[i]].Waves[j].get_color(), ms = msset) 
                j +=1
            
        # Axes:
            if i == self.LAP-1:
                self.axAPPlots[i].set_xlim(self.APTraces[self.APToTake[i]].Waves[0].get_data()[0][0],self.APTraces[self.APToTake[i]].Waves[0].get_data()[0][-1])
                start, end = self.axAPPlots[i].get_xlim()
                self.axAPPlots[i].xaxis.set_ticks(np.arange(start, end, 200))
                self.axAPPlots[i].set_xlabel('ms',fontsize = self.FSAxes,labelpad=0.5)
                plt.setp(self.axAPPlots[i].get_xticklabels(), fontsize = self.FSAxes)            
                self.axAPPlots[i].set_ylabel('mV',fontsize = self.FSAxes, rotation = 90)
                plt.setp(self.axAPPlots[i].get_yticklabels(),rotation = 'vertical', fontsize = self.FSAxes)            
                self.axAPPlots[i].spines["top"].set_visible(False)
                self.axAPPlots[i].spines["right"].set_visible(False)
            else:
                self.axAPPlots[i].set_xlim(self.APTraces[self.APToTake[i]].Waves[0].get_data()[0][0],self.APTraces[self.APToTake[i]].Waves[0].get_data()[0][-1])
                start, end = self.axAPPlots[i].get_xlim()
                self.axAPPlots[i].xaxis.set_ticks(np.arange(start, end, 200))
                self.axAPPlots[i].set_xticklabels([])
                self.axAPPlots[i].set_ylabel('mV',fontsize = self.FSAxes, rotation = 90)
                plt.setp(self.axAPPlots[i].get_yticklabels(),rotation = 'vertical', fontsize = self.FSAxes)            
                self.axAPPlots[i].spines["top"].set_visible(False)
                self.axAPPlots[i].spines["right"].set_visible(False)
                    
            i +=1 
            
        # Annotate first One:
        # Vec for Annotations:
        self.APPeakValue = self.APValues.APAmplitudeBaseline.Mean + APValues.BaselineMean.Mean
        self.AnnotWave = [None] * 6
        self.AnnotWave[0] = ("Base: %.1f mV" % self.APValues.BaselineMean.Mean)
        self.AnnotWave[1] = ("Peak: %.1f mV" % self.APPeakValue)
        self.AnnotWave[2] = ("Peak-Baseline Amp: %.1f mV" % self.APValues.APAmplitudeBaseline.Mean)
        self.AnnotWave[3] = ("AHP Area: %.0f mV$^{2}$/ms" % self.APValues.AHPArea.Mean)
        self.AnnotWave[4] = ("Latency: %.1f ms" % self.APValues.Latency.Mean)
        self.AnnotWave[5] = ("sAHP Amp: %.1f mV" % self.APValues.sAHPAmp.Mean)
        if self.APValues.APType == 'Burst':
            self.AnnotWave = [None] * 7
            self.AnnotWave[0] = ("Base: %.1f mV" % self.APValues.BaselineMean.Mean)
            self.AnnotWave[1] = ("Peak: %.1f mV" % self.APPeakValue)
            self.AnnotWave[2] = ("Peak-Baseline Amp: %.1f mV" % self.APValues.APAmplitudeBaseline.Mean)
            self.AnnotWave[3] = ("Change Amp: %.1f mV" % self.APValues.APAmpBaseChange.Mean)
            self.AnnotWave[4] = ("AHP Area: %.0f mV$^{2}$/ms" % self.APValues.AHPArea.Mean)
            self.AnnotWave[5] = ("Latency: %.1f ms" % self.APValues.Latency.Mean)
            self.AnnotWave[6] = ("sAHP Amp: %.1f mV" % self.APValues.sAHPAmp.Mean)

        i = 0
        k = 0
        while k < len(self.AnnotWave):
            self.axAPPlots[0].annotate(self.AnnotWave[k], self.APTraces[self.APToTake[i]].Annot[k].xy,self.APTraces[self.APToTake[i]].Annot[k].xyann, self.APTraces[self.APToTake[i]].Annot[k].xycoords, self.APTraces[self.APToTake[i]].Annot[k]._textcoords, fontsize = self.FSAnnots,color = self.APTraces[self.APToTake[i]].Annot[k].get_color())
            k +=1
                    
        # Title of the whole Plot: 
        self.TextRheo = ' at Rheobase of: %.1f pA' % self.CellInfo.RheoLPR
        self.TitleInBetween = ' with '
        self.TitleAP = self.APValues.APType +self.TitleInBetween + self.APValues.AHPType + self.TextRheo
        self.axAPPlots[0].set_title(self.TitleAP,fontsize = self.FSPlotHeaders,fontweight='bold',loc='left')
        
        # Plot Single Best AP 
        self.axApPlot = plt.subplot(self.gsSupThres[3:5,0])
        
        i = self.APToTake[0]
        j = 0
        while j < len(self.APTraces[i].WavesAP):
            if  self.APTraces[i].WavesAP[j].get_marker() == '.' or  self.APTraces[i].WavesAP[j].get_marker() == 'o':
                msset = self.Markersize
            else:
                msset = 0.5
            self.axApPlot.plot(self.APTraces[i].WavesAP[j].get_data()[0], self.APTraces[i].WavesAP[j].get_data()[1], self.APTraces[i].WavesAP[j].get_marker(), linestyle = self.APTraces[i].WavesAP[j].get_linestyle(), linewidth = self.ThicknessLines, color = self.APTraces[i].WavesAP[j].get_color(), ms = msset) 
            j +=1
        # Axes:
        self.axApPlot.set_xlim(self.APTraces[i].WavesAP[0].get_data()[0][0],self.APTraces[i].WavesAP[0].get_data()[0][-1])
        self.axApPlot.set_xlabel('ms',fontsize = self.FSAxes,labelpad=0.5)
        plt.setp(self.axApPlot.get_xticklabels(), fontsize = self.FSAxes)            
        self.axApPlot.set_ylabel('mV',fontsize = self.FSAxes, rotation = 90)
        plt.setp(self.axApPlot.get_yticklabels(),rotation = 'vertical', fontsize = self.FSAxes)            
        self.axApPlot.spines["top"].set_visible(False)
        self.axApPlot.spines["right"].set_visible(False)
        # Annotate AP: 
        self.AnnotAP = [None] * 4
        self.AnnotAP[0] = ("Peak: %.1f mV" % self.APPeakValue)
        self.AnnotAP[1] = ("Peak-Thres Amp: %.1f mV" % self.APValues.APAmplitudeThreshold.Mean)
        self.AnnotAP[2] = ("Half Width: %.1f ms" % self.APValues.HalfWidth.Mean)
        self.AnnotAP[3] = ("Threshold: %.1f ms" % self.APValues.Threshold.Mean)
        if self.APValues.APType == 'Burst':
            self.AnnotAP = [None] * 10
            self.AnnotAP[0] = ("Peak: %.1f mV" % self.APPeakValue)
            self.AnnotAP[1] = ("Peak-Thres Amp: %.1f mV" % self.APValues.APAmplitudeThreshold.Mean)
            self.AnnotAP[2] = ("Change Amp: %.1f mV" % self.APValues.APAmpThesChange.Mean)            
            self.AnnotAP[3] = ("Half Width: %.1f ms" % self.APValues.HalfWidth.Mean)
            self.AnnotAP[4] = ("Change HW: %.1f ms" % self.APValues.HalfWidthChange.Mean)
            self.AnnotAP[5] = ("Threshold: %.1f mV" % self.APValues.Threshold.Mean)
            self.AnnotAP[6] = ("Thres Change: %.1f mV" % self.APValues.ThresChange.Mean)
            self.AnnotAP[7] = ("BurstAHP: %.1f mV" % self.APValues.BurstAHPVm.Mean)
            self.AnnotAP[8] = ("AHP Change: %.1f mV" % self.APValues.BurstAHPVmChange.Mean)
            self.AnnotAP[9] = ("Burst Area: %.0f mV$^{2}$/ms" % self.APValues.BurstArea.Mean)           
 
        i = 0
        k = 0
        while k < len(self.AnnotAP):
            self.axApPlot.annotate(self.AnnotAP[k], self.APTraces[self.APToTake[i]].AnnotAP[k].xy,self.APTraces[self.APToTake[i]].AnnotAP[k].xyann, self.APTraces[self.APToTake[i]].AnnotAP[k].xycoords, self.APTraces[self.APToTake[i]].AnnotAP[k]._textcoords, fontsize = self.FSAnnots ,color = self.APTraces[self.APToTake[i]].AnnotAP[k].get_color())
            k +=1
            
        # Plot Single Best AHP: 
        self.axAHPPlot = plt.subplot(self.gsSupThres[3:5,1])
        j = 0
        while j < len(self.APTraces[i].WavesAHP):
            if  self.APTraces[i].WavesAHP[j].get_marker() == '.' or  self.APTraces[i].WavesAHP[j].get_marker() == 'o':
                msset = self.Markersize
            else:
                msset = 0.5
            self.axAHPPlot.plot(self.APTraces[i].WavesAHP[j].get_data()[0], self.APTraces[i].WavesAHP[j].get_data()[1], self.APTraces[i].WavesAHP[j].get_marker(), linestyle = self.APTraces[i].WavesAHP[j].get_linestyle(), linewidth = self.ThicknessLines, color = self.APTraces[i].WavesAHP[j].get_color(), ms = msset) 
            j +=1
        #Axes:
        self.axAHPPlot.set_xlim(self.APTraces[i].WavesAHP[0].get_data()[0][0]-2,self.APTraces[i].WavesAHP[0].get_data()[0][-1])
        self.axAHPPlot.set_xlabel('ms',fontsize = self.FSAxes,labelpad=0.5)
        plt.setp(self.axAHPPlot.get_xticklabels(), fontsize = self.FSAxes)            
        self.axAHPPlot.set_ylabel('mV',fontsize = self.FSAxes, rotation = 90)
        plt.setp(self.axAHPPlot.get_yticklabels(),rotation = 'vertical', fontsize = self.FSAxes)            
        self.axAHPPlot.spines["top"].set_visible(False)
        self.axAHPPlot.spines["right"].set_visible(False)
        # Annotate AP: 
        if self.APValues.AHPAll[self.APToTake[0]] == 'fMAHP' or self.APValues.AHPAll[self.APToTake[0]] == 'FmAHP':
            self.AnnotAHP = [None] * 6
            self.AnnotAHP[0] = ("fAHP: %.1f mV" % self.APValues.fAHPVm.Mean)
            self.AnnotAHP[1] = ("ADP: %.1f mV" % self.APValues.ADPVm.Mean)
            self.AnnotAHP[2] = ("mAHP: %.1f mV" % self.APValues.mAHPVm.Mean)
            self.AnnotAHP[3] = ("fAHPTtP: %.1f ms" % self.APValues.fAHPTtP.Mean)
            self.AnnotAHP[4] = ("ADPTtP: %.1f ms" % self.APValues.ADPTtP.Mean)
            self.AnnotAHP[5] = ("mAHPTtP: %.1f ms" % self.APValues.mAHPTtP.Mean)
        elif self.APValues.AHPAll[self.APToTake[0]] == 'fAHP':
            self.AnnotAHP = [None] * 2
            self.AnnotAHP[0] = ("fAHP: %.1f mV" % self.APValues.fAHPVm.Mean)
            self.AnnotAHP[1] = ("fAHPTtP: %.1f ms" % self.APValues.fAHPTtP.Mean)    
        elif self.APValues.AHPAll[self.APToTake[0]] == 'ADP':
            self.AnnotAHP = [None] * 2
            self.AnnotAHP[0] = ("ADP: %.1f mV" % self.APValues.ADPVm.Mean)
            self.AnnotAHP[1] = ("ADPTtP: %.1f ms" % self.APValues.ADPTtP.Mean)    
        elif self.APValues.AHPAll[self.APToTake[0]] == 'mAHP':
            self.AnnotAHP = [None] * 2
            self.AnnotAHP[0] = ("mAHP: %.1f mV" % self.APValues.mAHPVm.Mean)
            self.AnnotAHP[1] = ("mAHPTtP: %.1f ms" % self.APValues.mAHPTtP.Mean)  
        
        i = 0 
        k = 0
        while k < len(self.AnnotAHP):
            self.axAHPPlot.annotate(self.AnnotAHP[k], self.APTraces[self.APToTake[i]].AnnotAHP[k].xy,self.APTraces[self.APToTake[i]].AnnotAHP[k].xyann, self.APTraces[self.APToTake[i]].AnnotAHP[k].xycoords, self.APTraces[self.APToTake[i]].AnnotAHP[k]._textcoords, fontsize = self.FSAnnots ,color = self.APTraces[self.APToTake[i]].AnnotAHP[k].get_color())
            k +=1
            
        ####### Plot Firing 
#        print(self.iringValues.FiringType)
#        print(self.FiringValues.FiringType != 'Onset Cell')
        if self.FiringValues.FiringType != 'Onset Cell ' and self.FiringValues.FiringType != 'Onset Cell':
            # Plot Traces:        
            if len(self.FiringToTake) >=6:
                self.LFiring = 6
            else:
                self.LFiring = len(self.FiringToTake) 
            self.axFiringPlots = [None]*self.LFiring
            i = 0
            while i <self.LFiring:
                self.axFiringPlots[i] = plt.subplot(self.gsSupThres[6+i,2])
                i +=1  
                
            # Plotting: 
            i = 0
            while i < self.LFiring:
                j = 0
                while j < self.LFiringWaves[self.FiringToTake[i]]:
                    if  self.FiringTraces[self.FiringToTake[i]].Waves[j].get_marker() == '.' or  self.FiringTraces[self.FiringToTake[i]].Waves[j].get_marker() == 'o':
                            msset = self.Markersize
                    else:
                        msset = 0.5
                    # Printing:   
                    self.axFiringPlots[i].plot(self.FiringTraces[self.FiringToTake[i]].Waves[j].get_data()[0], self.FiringTraces[self.FiringToTake[i]].Waves[j].get_data()[1], self.FiringTraces[self.FiringToTake[i]].Waves[j].get_marker(), linestyle = self.FiringTraces[self.FiringToTake[i]].Waves[j].get_linestyle(), linewidth = self.ThicknessLines, color = self.FiringTraces[self.FiringToTake[i]].Waves[j].get_color(), ms = msset) 
                    j +=1
            # Axes
                if i == self.LFiring-1:
                    self.axFiringPlots[i].set_xlim(self.FiringTraces[self.FiringToTake[i]].Waves[0].get_data()[0][0],self.FiringTraces[self.FiringToTake[i]].Waves[0].get_data()[0][-1])
                    start, end = self.axFiringPlots[i].get_xlim()
                    self.axFiringPlots[i].xaxis.set_ticks(np.arange(start, end, 200))
                    self.axFiringPlots[i].set_xlabel('ms',fontsize = self.FSAxes,labelpad=0.5)
                    plt.setp(self.axFiringPlots[i].get_xticklabels(), fontsize = self.FSAxes)            
                    self.axFiringPlots[i].set_ylabel('mV',fontsize = self.FSAxes, rotation = 90)
                    plt.setp(self.axFiringPlots[i].get_yticklabels(),rotation = 'vertical', fontsize = self.FSAxes)            
                    self.axFiringPlots[i].spines["top"].set_visible(False)
                    self.axFiringPlots[i].spines["right"].set_visible(False)
                else:
                    self.axFiringPlots[i].set_xlim(self.FiringTraces[self.FiringToTake[i]].Waves[0].get_data()[0][0],self.FiringTraces[self.FiringToTake[i]].Waves[0].get_data()[0][-1])
                    start, end = self.axFiringPlots[i].get_xlim()
                    self.axFiringPlots[i].xaxis.set_ticks(np.arange(start, end, 200))
                    self.axFiringPlots[i].set_xticklabels([])
                    self.axFiringPlots[i].set_ylabel('mV',fontsize = self.FSAxes, rotation = 90)
                    plt.setp(self.axFiringPlots[i].get_yticklabels(),rotation = 'vertical', fontsize = self.FSAxes)            
                    self.axFiringPlots[i].spines["top"].set_visible(False)
                    self.axFiringPlots[i].spines["right"].set_visible(False)    
                i +=1

            # Plot FiringFreq:
            self.axFiringFreq = plt.subplot(self.gsSupThres[6:9,0:2])
            self.LenFiringFreq = [None]* self.LFiringTrace
            self.xlengthFiringFreqStart = [None]* self.LFiringTrace
            self.xlengthFiringFreqEnd = [None]* self.LFiringTrace
            i = 0
            while i < self.LFiringTrace:
                if hasattr(self.FiringTraces[i], 'WavesFreq'):
                    self.LenFiringFreq[i] = len(self.FiringTraces[i].WavesFreq)
                else:
                    self.LenFiringFreq[i] = 0    
                j = 0
                while j < self.LenFiringFreq[i]:
                    if  self.FiringTraces[i].WavesFreq[j][0].get_marker() == '.' or  self.FiringTraces[i].WavesFreq[j][0].get_marker() == 'o':
                            msset = self.Markersize
                    else:
                        msset = 0.5
                    # Printing:   
                    k = 0
                    while k < self.LFiringTrace:
                        if hasattr(self.FiringTraces[k], 'WavesFreq'): 
                            self.axFiringFreq.plot(self.FiringTraces[k].WavesFreq[j][0].get_data()[0], self.FiringTraces[k].WavesFreq[j][0].get_data()[1], self.FiringTraces[k].WavesFreq[j][0].get_marker(), linestyle = self.FiringTraces[k].WavesFreq[j][0].get_linestyle(), linewidth = self.ThicknessLines, color = self.FiringTraces[k].WavesFreq[j][0].get_color(), ms = msset) 
                            self.xlengthFiringFreqStart[k] = self.FiringTraces[k].WavesFreq[0][0].get_data()[0][0]
                            self.xlengthFiringFreqEnd[k] = self.FiringTraces[k].WavesFreq[0][0].get_data()[0][-1]
                        else:
                            self.xlengthFiringFreqStart[k] = np.nan
                            self.xlengthFiringFreqEnd[k] = np.nan     
                        k +=1
                    j +=1
                i +=1
            # Axes:
            self.axFiringFreq.set_xlim(np.nanmin(self.xlengthFiringFreqStart),np.nanmax(self.xlengthFiringFreqEnd))
            start, end = self.axFiringFreq.get_xlim()
            self.axFiringFreq.xaxis.set_ticks(np.arange(start, end, 100))
            self.axFiringFreq.set_xlabel('ms',fontsize = self.FSAxes,labelpad=0.5)
            plt.setp(self.axFiringFreq.get_xticklabels(), fontsize = self.FSAxes)            
            self.axFiringFreq.set_ylabel('Hz',fontsize = self.FSAxes, rotation = 90)
            plt.setp(self.axFiringFreq.get_yticklabels(),rotation = 'vertical', fontsize = self.FSAxes)            
            self.axFiringFreq.spines["top"].set_visible(False)
            self.axFiringFreq.spines["right"].set_visible(False)

            # Legend
            self.FiringLegend = [None]*2
            self.FiringLegend[0] = 'Fast Accomodation: %.3f' % self.FiringValues.FiringFreqFastAcc.Mean
            self.FiringLegend[1] = 'Slow Accomodation Fit: %.3f\nAccomodation Index: %.3f' % (self.FiringValues.FiringFreqSlowAcc.Mean,self.FiringValues.FiringFreqIndex.Mean)
            
            if hasattr(self.FiringTraces[0], 'WavesFreq'):
                a = 0
            else:
                a = 1

            self.axFiringFreq.legend([self.FiringTraces[a].WavesFreq[1][0],self.FiringTraces[a].WavesFreq[2][0]],\
                                   [self.FiringLegend[0],self.FiringLegend[1]],\
                                   loc=0, fontsize = self.FSAnnots, frameon=False)

                
            # Title for Firing:
            self.TextFiring = [None]*3
            self.TextFiring[0]= ': %.1f ms' % (self.FiringValues.FiringDuration.Mean)
            self.TextFiring[1]= ' ,%.2f Hz' % (self.FiringValues.FiringFrequency.Mean)
            self.TextFiring[2]= ' %.2f mV' % (self.FiringValues.FirstSpikeAmpBase.Mean)
            self.TitleFiring = self.FiringValues.FiringType + self.TextFiring[0] + self.TextFiring[1] +self.TextFiring[2]
            self.axFiringFreq.set_title(self.TitleFiring,fontsize = self.FSPlotHeaders,fontweight='bold',loc='left')
            
            # Plot FiringAmplitudes:
            self.axFiringAmp = plt.subplot(self.gsSupThres[9:12,0:2])
            self.LenFiringAmp = [None]* self.LFiringTrace
            self.xlengthFiringAmpStart = [None]* self.LFiringTrace
            self.xlengthFiringAmpEnd = [None]* self.LFiringTrace
            i = 0
            while i < self.LFiringTrace:
                if hasattr(self.FiringTraces[i], 'WavesAmp'): 
                    self.LenFiringAmp[i] = len(self.FiringTraces[i].WavesAmp)
                else:
                    self.LenFiringAmp[i] = 0
                j = 0
                while j < self.LenFiringAmp[i]:
                    if  self.FiringTraces[i].WavesAmp[j][0].get_marker() == '.' or  self.FiringTraces[i].WavesAmp[j][0].get_marker() == 'o':
                            msset = self.Markersize
                    else:
                        msset = 0.5
                    # Printing:   
                    k = 0
                    while k < self.LFiringTrace:
                        if hasattr(self.FiringTraces[k], 'WavesAmp'): 
                            self.axFiringAmp.plot(self.FiringTraces[k].WavesAmp[j][0].get_data()[0], self.FiringTraces[k].WavesAmp[j][0].get_data()[1], self.FiringTraces[k].WavesAmp[j][0].get_marker(), linestyle = self.FiringTraces[k].WavesAmp[j][0].get_linestyle(), linewidth = self.ThicknessLines, color = self.FiringTraces[k].WavesAmp[j][0].get_color(), ms = msset) 
                            self.xlengthFiringAmpStart[k] = self.FiringTraces[k].WavesAmp[0][0].get_data()[0][0]
                            self.xlengthFiringAmpEnd[k] = self.FiringTraces[k].WavesAmp[0][0].get_data()[0][-1]
                        else:
                            self.xlengthFiringAmpStart[k] = np.nan
                            self.xlengthFiringAmpEnd[k] = np.nan
                        
                        k +=1
                    j +=1
                i +=1
            # Axes:
            self.axFiringAmp.set_xlim(np.nanmin(self.xlengthFiringAmpStart),np.nanmax(self.xlengthFiringAmpEnd))
            start, end = self.axFiringAmp.get_xlim()
            self.axFiringAmp.xaxis.set_ticks(np.arange(start, end, 100))
            self.axFiringAmp.set_xlabel('ms',fontsize = self.FSAxes,labelpad=0.5)
            plt.setp(self.axFiringAmp.get_xticklabels(), fontsize = self.FSAxes)            
            self.axFiringAmp.set_ylim(bottom=0)
            self.axFiringAmp.set_ylabel('mV',fontsize = self.FSAxes, rotation = 90)
            plt.setp(self.axFiringAmp.get_yticklabels(),rotation = 'vertical', fontsize = self.FSAxes)            
            self.axFiringAmp.spines["top"].set_visible(False)
            self.axFiringAmp.spines["right"].set_visible(False)
            # Legend
            self.FiringAmpLegend = [None]*4
            self.FiringAmpLegend[0] = 'Base-Peak Fast Adaption: %.3f' % self.FiringValues.APBaseFastAdaption.Mean
            self.FiringAmpLegend[1] = 'Base-Peak Slow Adaption Fit: %.3f\nBase-Peak Adaption Index: %.3f' % (self.FiringValues.APBaseSlowAdaption.Mean,self.FiringValues.APBaseAdaptionIndex.Mean)
            
            self.FiringAmpLegend[2] = 'Thres-Peak Fast Adaption: %.3f' % self.FiringValues.APThresFastAdaption.Mean
            self.FiringAmpLegend[3] = 'Thres-Peak Slow Adaption Fit: %.3f\nThres-Peak Adaption Index: %.3f' % (self.FiringValues.APThresSlowAdaption.Mean,self.FiringValues.APThresAdaptionIndex.Mean)
            
            if hasattr(self.FiringTraces[0], 'WavesAmp'):
                a = 0
            else:
                a = 1
            
            self.axFiringAmp.legend([self.FiringTraces[a].WavesAmp[1][0],self.FiringTraces[a].WavesAmp[2][0],self.FiringTraces[a].WavesAmp[4][0],self.FiringTraces[a].WavesAmp[5][0]],\
                                   [self.FiringAmpLegend[0],self.FiringAmpLegend[1],self.FiringAmpLegend[2],self.FiringAmpLegend[3]],\
                                   loc=0, fontsize = self.FSAnnots, frameon=False)
        else:
            
            self.axFiringPlots = [None]*self.LFiringTrace
            i = 0
            while i <self.LFiringTrace:
                self.axFiringPlots[i] = plt.subplot(self.gsSupThres[18+i])
                i +=1  
                
            # Plotting: 
            i = 0
            while i < self.LFiringTrace:
                j = 0
                while j < self.LFiringWaves[i]:
                    if  self.FiringTraces[i].Waves[j].get_marker() == '.' or  self.FiringTraces[i].Waves[j].get_marker() == 'o':
                            msset = self.Markersize
                    else:
                        msset = 0.5
                    # Printing:   
                    self.axFiringPlots[i].plot(self.FiringTraces[i].Waves[j].get_data()[0], self.FiringTraces[i].Waves[j].get_data()[1], self.FiringTraces[i].Waves[j].get_marker(), linestyle = self.FiringTraces[i].Waves[j].get_linestyle(), linewidth = self.ThicknessLines, color = self.FiringTraces[i].Waves[j].get_color(), ms = msset) 
                    j +=1
                #Axes
                self.axFiringPlots[i].set_xlim(self.FiringTraces[i].Waves[0].get_data()[0][0],self.FiringTraces[i].Waves[0].get_data()[0][-1])
                start, end = self.axFiringPlots[i].get_xlim()
                self.axFiringPlots[i].xaxis.set_ticks(np.arange(start, end, 200))
                self.axFiringPlots[i].set_xlabel('ms',fontsize = self.FSAxes,labelpad=0.5)
                plt.setp(self.axFiringPlots[i].get_xticklabels(), fontsize = self.FSAxes)            
                self.axFiringPlots[i].set_ylabel('mV',fontsize = self.FSAxes, rotation = 90)
                plt.setp(self.axFiringPlots[i].get_yticklabels(),rotation = 'vertical', fontsize = self.FSAxes)            
                self.axFiringPlots[i].spines["top"].set_visible(False)
                self.axFiringPlots[i].spines["right"].set_visible(False)
                i +=1
            
            self.axFiringPlots[0].set_title('Onset Cell',fontsize = self.FSPlotHeaders,fontweight='bold',loc='left')
        
        
        ''' CellCharacteristics and Figure and Additional Stuff: '''
        self.gsGeneral = gridspec.GridSpec(10,3) #, width_ratios=[2,1,1])
        self.gsGeneral.update(left=0.01, bottom= 0.04, top = 0.91, right=0.31, wspace=0.04, hspace=0.04)
        
        # Print General Information:
        self.AxCellInfo = plt.subplot(self.gsGeneral[0,0:2]) #> Takes gsGeneral[0:2,0:2]
        self.AxCellInfo.axis('off')
        # Table of Info:
        self.CellInfoArray = [None]*15
        self.CellInfoArray[0] = 'Recording Date: '+ self.CellInfo.RecDate.strftime("%d %m %Y")+'\n'
        self.CellInfoArray[1] = 'Animal Number: %0.f' % self.CellInfo.AnimalNumber+'\n'
        self.CellInfoArray[2] = 'Genotype: '+self.CellInfo.Genotype+'\n'
        self.CellInfoArray[3] = 'Age: %2.f Weeks' % self.CellInfo.AgeWeeks+'\n'
        self.CellInfoArray[4] = 'Injection: '+ self.CellInfo.Injection+'\n'
        self.CellInfoArray[5] = 'Expression: %2.f Days' % self.CellInfo.ExpressionDays+'\n'
        self.CellInfoArray[6] = 'Tissue: ' + self.CellInfo.Tissue+'\n'
        self.CellInfoArray[7] = '\n'
        self.CellInfoArray[8] = 'Region: ' + self.CellInfo.Region+'\n'
        self.CellInfoArray[9] = 'Subregion: ' +self.CellInfo.SubRegion+'\n'
        self.CellInfoArray[10] = 'Celltype: ' + self.CellInfo.Celltype+'\n'
        self.CellInfoArray[11] = 'Morphology: ' + self.CellInfo.Morphology+'\n'
        self.CellInfoArray[12] = 'ImmunoStaining: ' + self.CellInfo.Immuno+'\n'
        self.CellInfoArray[13] = '\n'
        self.CellInfoArray[14] = 'Comment: ' + self.CellInfo.Comment+'\n'
        #Printing:
        self.CellInfoText = ''.join(self.CellInfoArray)
        self.AxCellInfo.text(0.001, 1.8, self.CellName, transform=self.AxCellInfo.transAxes, fontsize=self.FSCellName, verticalalignment='top',fontweight='bold')       
        self.AxCellInfo.text(0.001, 1.45, self.CellInfoText, transform=self.AxCellInfo.transAxes, fontsize=self.FSCellInfo, verticalalignment='top')
        
        # Pictures
        if len(self.Pics) > 0:
            if len(self.Pics) < 4:
                self.PicPlotNum = len(self.Pics)
            else:
                self.PicPlotNum = 4
            self.AxPics =[None]*self.PicPlotNum
            self.AxPics[0] = plt.subplot(self.gsGeneral[6:9,:])
            i = 1
            while i < self.PicPlotNum:
                self.AxPics[i] = plt.subplot(self.gsGeneral[9,i-1])
                i +=1
            i = 0
            while i < self.PicPlotNum:
                self.AxPics[i].imshow(self.Pics[i-1])
                self.AxPics[i].axis('off')
                i +=1
                
        ''' Printing Figure '''
        if self.PrintShow == 1:
            self.SavingName = self.CellName+'_CellCharacterisation'
            ddPlotting.save(self.SavingName, ext="png", close=False, verbose=True)
            #ddPlotting.save(self.SavingName, ext="svg", close=True, verbose=True)
#            plt.close('All')
            plt.ion()    



''' Start of the Script: '''
class AnalysisCellCharacteristics:
    def __init__ (self,Cellname,Conditions,OVTableName):
        self.CellName = Cellname
        self.Conditions = Conditions # Dictionary with All conditions:
        self.OVTableName = OVTableName
        
        ''' Unravel Conditions: '''
        self.AmpApThres = ddhelp.SearchKeyDic(self.Conditions,'AmpApThres')
        self.PassiveRange = ddhelp.SearchKeyDic(self.Conditions,'PassiveRange')
        self.CurrentAmpsBasic = ddhelp.SearchKeyDic(self.Conditions,'CurrentAmpsBasic')
        self.AnalyseOnlyIR = ddhelp.SearchKeyDic(self.Conditions,'AnalyseOnlyIR')
        
        # LPR Time for Firing
        self.LPRXTime = ddhelp.SearchKeyDic(self.Conditions,'LPRXTime')
        
        # Print/Show Conditions:
        self.PrintShowAll = ddhelp.SearchKeyDic(self.Conditions,'PrintShowAll')
        self.PrintShowPassive = ddhelp.SearchKeyDic(self.Conditions,'PrintShowPassive')
        self.PrintShowHypo = ddhelp.SearchKeyDic(self.Conditions,'PrintShowHypo')
        self.PrintShowJustSub = ddhelp.SearchKeyDic(self.Conditions,'PrintShowJustSub')
        self.PrintShowAP = ddhelp.SearchKeyDic(self.Conditions,'PrintShowAP')
        self.PrintShowFiring = ddhelp.SearchKeyDic(self.Conditions,'PrintShowFiring')
        self.PrintOV = ddhelp.SearchKeyDic(self.Conditions,'PrintOV')
        
        # Calculate Conditions: 
        self.ToCalcAll = ddhelp.SearchKeyDic(self.Conditions,'ToCalcAll')
        self.ToCalcPassive = ddhelp.SearchKeyDic(self.Conditions,'ToCalcPassive')
        self.ToCalcHypo = ddhelp.SearchKeyDic(self.Conditions,'ToCalcHypo')
        self.ToCalcJustSub = ddhelp.SearchKeyDic(self.Conditions,'ToCalcJustSub')
        self.ToCalcAP = ddhelp.SearchKeyDic(self.Conditions,'ToCalcAP')
        self.ToCalcFiring = ddhelp.SearchKeyDic(self.Conditions,'ToCalcFiring')
        
        # Convert when All:
        if self.PrintShowAll == 1:
            self.PrintShowPassive = 1
            self.PrintShowHypo = 1
            self.PrintShowJustSub = 1
            self.PrintShowAP =1
            self.PrintShowFiring = 1
            self.PrintOV = 1
        
        ''' Import Table '''
        self.CellInfo = ImportCellCheat('XFile.xlsx')
        
        ''' Import Pics '''
        self.AllFolders = [name for name in os.listdir(os.getcwd())\
                           if os.path.isdir(os.path.join(os.getcwd(), name))]
        if 'Pics' in self.AllFolders:
            self.Pics = DDImport.ImportPicsFolder('Pics')   
        else:
            self.Pics = []
        
        ''' Import Traces '''
        self.BasicTraces = ImportAndSortIR('Basic')
        self.HypoTraces = ImportOther('Hypo50',self.BasicTraces,self.AnalyseOnlyIR)
        self.JustSubTraces = ImportOther('JustSub',self.BasicTraces,self.AnalyseOnlyIR)
        self.APTraces = ImportOther('LPR',self.BasicTraces,self.AnalyseOnlyIR)
        self.FiringTraces = ImportOther(self.LPRXTime,self.BasicTraces,self.AnalyseOnlyIR)
        
        ''' Get from which Current Injections the Analysis is Based On: ''' 
        self.Stim = {'HypoStim':self.HypoTraces.AnalysisOn}
        self.Stim['JustSubStim'] = self.JustSubTraces.AnalysisOn
        self.Stim['APStim'] = self.APTraces.AnalysisOn
        self.Stim['FiringStim'] = self.FiringTraces.AnalysisOn
        if self.CellInfo.HypoStim == np.nan or self.Stim['HypoStim'] == 'IRStimulation':
            self.CellInfo.HypoStim = self.Conditions['CurrentAmpsBasic'][self.BasicTraces.BasicHypo]
        if self.CellInfo.JustSubStim == np.nan or self.Stim['JustSubStim'] == 'IRStimulation':
            self.CellInfo.JustSubStim = self.Conditions['CurrentAmpsBasic'][self.BasicTraces.BasicJustSub]
        if self.CellInfo.RheoLPR == np.nan or self.Stim['APStim'] == 'IRStimulation':
            self.CellInfo.RheoLPR = self.Conditions['CurrentAmpsBasic'][self.BasicTraces.BasicAP]
        if self.Stim['FiringStim'] == 'IRStimulation':
            self.CellInfo.FiringStim = self.Conditions['CurrentAmpsBasic'][self.BasicTraces.BasicFiring]
        else:
            self.CellInfo.FiringStim = self.CellInfo.RheoLPR * int(''.join(filter(str.isdigit, self.LPRXTime)))
        # Number of Traces: 
        self.CellInfo.NumHypoTraces = self.HypoTraces.NumWaves
        self.CellInfo.NumJustSubTraces = self.JustSubTraces.NumWaves
        self.CellInfo.NumAPTraces = self.APTraces.NumWaves
        self.CellInfo.NumFiringTraces = self.FiringTraces.NumWaves
        
        ''' Go To CellCharact Folder: '''
        if not os.path.isdir(os.getcwd()+'/Calcs_CellCharaterisation'):
            os.makedirs(os.getcwd()+'/Calcs_CellCharaterisation') 
        ''' > Single Values to Excel Sheets: '''
        os.chdir(os.getcwd()+'/Calcs_CellCharaterisation')
        
        ''' Calculations '''
        #### Passive Properties: 
        if self.ToCalcPassive == 1 or self.ToCalcAll == 1:
            self.Passive = AnalysisPassive(self.BasicTraces,self.CurrentAmpsBasic,self.PassiveRange)
            # Saving Single Table:
            ddhelp.SaveAsExcelFile(self.Passive.GroundValues.Table, self.CellName+'_PassiveValues')
        else:
            self.Passive = type('test', (), {})()
            self.Passive.Values = 0
        #### Hypo Properties:
        if self.ToCalcHypo == 1 or self.ToCalcAll == 1:
            self.Hypo = Analysis(self.HypoTraces,'SubthresholdProperties','Main','SubActiveValues')
            # Saving Single Table:
            ddhelp.SaveAsExcelFile(self.Hypo.Values.Table, self.CellName+'_HypoValues')
        else:
            self.Hypo = type('test', (), {})()
            self.Hypo.Values = 0
        #### JustSub Properties:
        if self.ToCalcJustSub == 1 or self.ToCalcAll == 1:
            self.JustSub = Analysis(self.JustSubTraces,'SubthresholdProperties','Main','SubActiveValues')
            # Saving Single Table:
            ddhelp.SaveAsExcelFile(self.JustSub.Values.Table, self.CellName+'_JustSubValues')
        else:
            self.JustSub = type('test', (), {})()
            self.JustSub.Values = 0
        #### AP Properties:
        if self.ToCalcAP == 1 or self.ToCalcAll == 1:
            self.AP = Analysis(self.APTraces,'SuprathresholdProperties','MainAP','APValues')
            self.APCount = Counter(ddhelp.ListofSubAttributes(self.AP.Values,'TraceToTake'))
            self.APToTake = ddhelp.SortBySD(self.AP.Values.Table)

            # Saving Single Table:
            ddhelp.SaveAsExcelFile(self.AP.Values.Table, self.CellName+'_APValues')
        else:
            self.AP = type('test', (), {})()
            self.AP.Values = 0
        #### Firing Properties:
        if self.ToCalcFiring == 1 or self.ToCalcAll == 1:
            self.Firing = Analysis(self.FiringTraces,'SuprathresholdProperties','MainFiring','FiringValues')
            self.FiringCount = Counter(ddhelp.ListofSubAttributes(self.Firing.Values,'TraceToTake'))
            self.FiringToTake = ddhelp.SortBySD(self.Firing.Values.Table)
            # Saving Single Table:
            ddhelp.SaveAsExcelFile(self.Firing.Values.Table, self.CellName+'_FiringValues')
        else:
            self.Firing = type('test', (), {})()
            self.Firing.Values = 0
        
        ''' Create Main CellTable '''
        self.CellTable = CellTable(self.CellName,self.CellInfo,self.Passive.Values,self.Hypo.Values,self.JustSub.Values,self.AP.Values,self.Firing.Values)         
        ### Compare with other!..... Replace or add Cell! 
        
        ''' Single Figures '''
        if self.PrintShowAll == 1 or self.PrintShowPassive == 1 or self.PrintShowAll == 2 or self.PrintShowPassive == 2:
            self.PassivFig = SubthresholdProperties.PlotPassive(self.Passive.SubThresValues,self.Passive.IVCalc,self.Passive.TauCalc,self.Passive.Values,self.PrintShowPassive,self.CellName)
        if self.PrintShowAll == 1 or self.PrintShowHypo == 1 or self.PrintShowAll == 2 or self.PrintShowHypo == 2:
            
            self.JustSubHypoFig = SubthresholdProperties.SubActivePlot(self.Hypo.Calcs,self.Hypo.Values,self.PrintShowHypo,self.CellName)
        if self.PrintShowAll == 1 or self.PrintShowJustSub == 1 or self.PrintShowAll == 2 or self.PrintShowJustSub == 2:
            self.JustSubDepoFig = SubthresholdProperties.SubActivePlot(self.JustSub.Calcs,self.JustSub.Values,self.PrintShowJustSub,self.CellName)
        if self.PrintShowAll == 1 or self.PrintShowAP == 1 or self.PrintShowAll == 2 or self.PrintShowAP == 2:
            self.APFig = SuprathresholdProperties.PlotAPs(self.AP.Calcs,self.AP.Values,self.PrintShowAP,self.CellName)
        if self.PrintShowAll == 1 or self.PrintShowFiring == 1 or self.PrintShowAll == 2 or self.PrintShowFiring == 2:
            if self.Firing.Values.NumAPs.Mean > 2:
                self.FiringFig = SuprathresholdProperties.PlotFiring(self.Firing.Calcs,self.Firing.Values,self.PrintShowFiring,self.CellName)
        
        ''' Go Back To mainFolder '''
        os.chdir("..")
        
        ''' Save Main ExcelSheet in CellFodler: '''
        ddhelp.SaveAsExcelFile(self.CellTable.CellTable, self.CellName+'_OV_CellCharacterisation')
        
        ''' Main Figure '''
        if self.PrintShowAll == 1 or self.PrintOV == 1:
            self.MainFig = PlotMainFigure(self.CellName,self.CellInfo,self.Pics,\
                              self.Passive.Values,self.Passive.SubThresValues,self.Passive.IVCalc,\
                              self.Hypo.Values,self.Hypo.Calcs,\
                              self.JustSub.Values,self.JustSub.Calcs,\
                              self.AP.Values,self.AP.Calcs,self.APToTake,\
                              self.Firing.Values,self.Firing.Calcs,self.FiringToTake,\
                              self.PrintOV)
        
        ''' Print All Cells Sheet '''
        os.chdir("..")        
        # Import OVTable:
        self.OVTable,X = DDImport.ImportExcel(self.OVTableName)

        # Print or UpDate All_Characterisation:
        self.NewEntry = self.CellTable.CellTable
        if not hasattr(self, 'OVTable') or self.OVTable is None:
            self.NewTable = self.NewEntry
        else:   
            self.C = ddhelp.UpdateAllTable(self.NewEntry,self.OVTable)
            self.NewTable = self.C.NewTable.sort_index()
            
        self.A = ddhelp.SaveAsExcelFile(self.NewTable,self.OVTableName)   
        
        # Get Accesory Table into Folder:
        if not os.path.isdir(os.getcwd()+'/'+self.OVTableName):
            os.makedirs(os.getcwd()+'/'+self.OVTableName)        
        os.chdir(os.getcwd()+'/'+self.OVTableName)
        self.AccessoryOVTableName = self.OVTableName+'_Accessory'
        self.AccessoryOVTable,X = DDImport.ImportExcel(self.AccessoryOVTableName)
        self.AccessoryNewEntry = self.CellTable.AcessoryCellTable
        if not hasattr(self, 'AccessoryOVTable') or self.AccessoryOVTable is None:
            self.AccessoryNewTable = self.AccessoryNewEntry
        else:   
            self.AccessoryC = ddhelp.UpdateAllTable(self.AccessoryNewEntry,self.AccessoryOVTable)
            self.AccessoryNewTable = self.AccessoryC.NewTable.sort_index()    
        self.A = ddhelp.SaveAsExcelFile(self.AccessoryNewTable,self.AccessoryOVTableName)   
        os.chdir("..")
        
        # Plot MainFigure in All Cells Folder:
        if self.PrintOV == 1:
            if not os.path.isdir(os.getcwd()+'/'+self.OVTableName):
                os.makedirs(os.getcwd()+'/'+self.OVTableName)        
            os.chdir(os.getcwd()+'/'+self.OVTableName)
#            ddPlotting.save(self.CellName, ext="png", close=True, verbose=True)
            self.MainFig.Figure.savefig(self.CellName+'.png',dpi=600) # Save Fig
            plt.close('All')
            os.chdir("..")

        os.chdir(self.CellName)
        print(self.CellName+'.....FINISHED')
#        print(self.APToTake)


''' Call Script: '''
if Testing == 1:
    A = AnalysisCellCharacteristics(CellName,Conditions,OVTableName)
#    B = A.APToTake
#    print(B)
#    B = A.CellTable.AcessoryCellTable
#    a1 = A.Hypo.Calcs[1].MaxResponseValues.Devi.t
#    B = A.Hypo.Calcs[1].MaxResponseValues.Devi.A.Wave
#    B1 = A.Hypo.Calcs[1].MaxResponseValues.Devi.A.WaveDiff
#    b4 = np.argmax(B1)
#    b0 = A.Hypo.Calcs[1].MaxResponseValues.Devi.A.PotMax
#    b01 = A.Hypo.Calcs[1].MaxResponseValues.Devi.PotMaxDiff
#    b = A.Hypo.Calcs[1].MaxResponseValues.Devi.MaxTAllmost
#    b1 = np.argmax(A.Hypo.Calcs[1].MaxResponseValues.Devi.Wave)
#    b2 = A.Hypo.Calcs[1].MaxResponseValues.Devi.FindToTakeInMaxTAllmost
#    b3 = np.argmin(b2)
#    b5 = A.Hypo.Calcs[1].MaxResponseValues.Devi.MaxT
#    plt.ion()
#    c =  plt.figure()
#    cAx =  plt.subplot(1,1,1)
#    cAx.plot(B1)
#    

#''' Test Cell Folder:'''
#import DDFolderLevel
#FolderConditions = {'AnalyseOneCell':0} 
#FolderConditions['AnalyseAllCells']=0 
#FolderConditions['AnalyseNewCells']=0 
#FolderConditions['AnalyseOldCells']=1
#AnalysisPath = '/Users/DennisDa/Desktop/DataAnalaysis/DZNE_CellCharacterisation/Test 2'
#os.chdir(AnalysisPath) 
#FolderListName = 'Test_Characterisation'
#Identifier = 'DD'
#A = DDFolderLevel.WhatToAnalyse(AnalysisPath,FolderConditions,FolderListName,Identifier)
#print(A.CellsToAnalyse)
#print(A.NumCells)




