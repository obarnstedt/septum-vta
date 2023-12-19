#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:33:35 2017

@author: DennisDa
"""
import os
import DDImport
import ddhelp

''' To Test: '''
''' To Test: '''
Testing = 0
if Testing ==1:
    PathToData = '/Users/DennisDa/Desktop/DataAnalaysis/DZNE_CellCharacterisation/Test'
#    PathToData = '/Volumes/NO NAME/New Subiculum Recordings'
    os.chdir(PathToData)
    CellsfolderPath = os.getcwd()
    
    #PathToData = '/Users/DennisDa/Desktop/DataAnalaysis/Test Sorting'
    #CellsfolderPath = PathToData
    #os.chdir(PathToData)
    ConditionsFolder={'AnalyseOneCell':0}
    ConditionsFolder['AnalyseAllCells']  = 0
    ConditionsFolder['AnalyseNewCells']  = 0
    ConditionsFolder['AnalyseOldCells']  = 1


class WhatToAnalyse:
    def __init__ (self,AnalysisPath,ConditionsFolder,FolderListName,Identifier):
        self.Folderpath = AnalysisPath
        self.Conditions = ConditionsFolder
        self.FolderListName = FolderListName
        self.Identifier = Identifier
        
        # Unravel Conditions:
        self.AnalyseOneCell = ddhelp.SearchKeyDic(self.Conditions,'AnalyseOneCell')
        self.AnalyseAllCells = ddhelp.SearchKeyDic(self.Conditions,'AnalyseAllCells')
        self.AnalyseNewCells = ddhelp.SearchKeyDic(self.Conditions,'AnalyseNewCells')
        self.AnalyseOldCells = ddhelp.SearchKeyDic(self.Conditions,'AnalyseOldCells')
        # Files In Folder:
        self.files = [f for f in os.listdir(self.Folderpath) if ( f.startswith(self.Identifier))]
        # Files in List: 
        self.ExcelListAll,self.OK = DDImport.ImportExcel(self.FolderListName)
        if self.OK == 1: 
            self.ExcelList = self.ExcelListAll.index.tolist()
            self.OldCells = list(set(self.files) & set(self.ExcelList))
            self.NumOldCells = len(self.OldCells)
            
            self.NewCells = [x for x in self.files if x not in self.ExcelList]
            
            self.NumNewCells = len(self.NewCells)
            self.AllCells = self.files
            self.NumAllCells = len(self.AllCells)
            
            # OutPut:
            if self.AnalyseAllCells == 1:
                self.CellsToAnalyse = self.AllCells
                self.NumCells = self.NumAllCells
            if self.AnalyseNewCells == 1:
                self.CellsToAnalyse = self.NewCells
                self.NumCells = self.NumNewCells
            if self.AnalyseOldCells == 1:
                self.CellsToAnalyse = self.OldCells
                self.NumCells = self.NumOldCells
            if self.AnalyseOneCell == 1:
                self.Path = os.getcwd()
                a = self.Path.split("/")
                self.CellsToAnalyse = a[-1]
                self.NumCells = 1
                os.chdir("..")
        else:
            self.CellsToAnalyse = self.files
            self.NumCells = len(self.files) 
            
        if self.AnalyseOneCell == 1:
            self.Path = os.getcwd()
            a = self.Path.split("/")
            self.CellsToAnalyse = a[-1]
            self.NumCells = 1
            os.chdir("..")
        
        

''' TO TEST: '''
''' TO TEST: '''
if Testing == 1:
    A = WhatToAnalyse(CellsfolderPath,ConditionsFolder,'AllCells_Characterisation','DD')
#    A = WhatToAnalyse(CellsfolderPath,ConditionsFolder,'SortingOV','DD')
    a = A.files
    b = A.ExcelList 
    c = A.OldCells
    d = A.NewCells
    e = A.CellsToAnalyse

        
        