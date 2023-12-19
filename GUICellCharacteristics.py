# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUICellCharacteristics2.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!
''' Import Scripts: '''
import os
import numpy as np
#import matplotlib as mpl 
#if mpl.get_backend != 'agg': 
#    mpl.use('agg')
#print(mpl.get_backend)
import ddhelp
import DDFolderLevel
import DDCellLevel
import DDImport
import pandas as pd
#import matplotlib as mpl 
#if mpl.get_backend != 'agg': 
#    mpl.use('agg')
#print(mpl.get_backend)
#from matplotlib import pyplot as plt

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer

''' GUI: '''
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 614)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalFrame = QtWidgets.QFrame(self.centralwidget)
        self.verticalFrame.setGeometry(QtCore.QRect(0, 0, 591, 591))
        self.verticalFrame.setObjectName("verticalFrame")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalFrame)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.SetWorkingDirectory = QtWidgets.QPushButton(self.verticalFrame)
        self.SetWorkingDirectory.setObjectName("SetWorkingDirectory")
        self.horizontalLayout.addWidget(self.SetWorkingDirectory)
        self.ShowWorkingDirectory = QtWidgets.QTextEdit(self.verticalFrame)
        self.ShowWorkingDirectory.setMaximumSize(QtCore.QSize(16777215, 40))
        self.ShowWorkingDirectory.setObjectName("ShowWorkingDirectory")
        self.horizontalLayout.addWidget(self.ShowWorkingDirectory)
        self.verticalLayout_5.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.AnalyseOneCell = QtWidgets.QCheckBox(self.verticalFrame)
        self.AnalyseOneCell.setChecked(False)
        self.AnalyseOneCell.setObjectName("AnalyseOneCell")
        self.horizontalLayout_2.addWidget(self.AnalyseOneCell)
        self.AnalyseAllCells = QtWidgets.QCheckBox(self.verticalFrame)
        self.AnalyseAllCells.setChecked(True)
        self.AnalyseAllCells.setObjectName("AnalyseAllCells")
        self.horizontalLayout_2.addWidget(self.AnalyseAllCells)
        self.AnalyseNewCells = QtWidgets.QCheckBox(self.verticalFrame)
        self.AnalyseNewCells.setObjectName("AnalyseNewCells")
        self.horizontalLayout_2.addWidget(self.AnalyseNewCells)
        self.AnalyseOldCells = QtWidgets.QCheckBox(self.verticalFrame)
        self.AnalyseOldCells.setObjectName("AnalyseOldCells")
        self.horizontalLayout_2.addWidget(self.AnalyseOldCells)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.CellstoAnalyse = QtWidgets.QLabel(self.verticalFrame)
        self.CellstoAnalyse.setMaximumSize(QtCore.QSize(300, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.CellstoAnalyse.setFont(font)
        self.CellstoAnalyse.setObjectName("CellstoAnalyse")
        self.horizontalLayout_10.addWidget(self.CellstoAnalyse)
        self.Identifier = QtWidgets.QLabel(self.verticalFrame)
        self.Identifier.setObjectName("Identifier")
        self.horizontalLayout_10.addWidget(self.Identifier)
        self.IVCalcfrom = QtWidgets.QTextEdit(self.verticalFrame)
        self.IVCalcfrom.setMaximumSize(QtCore.QSize(44, 20))
        self.IVCalcfrom.setObjectName("IVCalcfrom")
        self.horizontalLayout_10.addWidget(self.IVCalcfrom)
        self.horizontalLayout_12.addLayout(self.horizontalLayout_10)
        self.OVTableIdentifier = QtWidgets.QLabel(self.verticalFrame)
        self.OVTableIdentifier.setObjectName("OVTableIdentifier")
        self.horizontalLayout_12.addWidget(self.OVTableIdentifier)
        self.ResultsTableName = QtWidgets.QTextEdit(self.verticalFrame)
        self.ResultsTableName.setMaximumSize(QtCore.QSize(16777215, 20))
        self.ResultsTableName.setObjectName("ResultsTableName")
        self.horizontalLayout_12.addWidget(self.ResultsTableName)
        self.horizontalLayout_14.addLayout(self.horizontalLayout_12)
        self.verticalLayout_5.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.SelectCellToAnalyse = QtWidgets.QPushButton(self.verticalFrame)
        self.SelectCellToAnalyse.setObjectName("SelectCellToAnalyse")
        self.horizontalLayout_3.addWidget(self.SelectCellToAnalyse)
        self.ShowCellToAnalyse = QtWidgets.QTextEdit(self.verticalFrame)
        self.ShowCellToAnalyse.setMaximumSize(QtCore.QSize(16777215, 20))
        self.ShowCellToAnalyse.setObjectName("ShowCellToAnalyse")
        self.horizontalLayout_3.addWidget(self.ShowCellToAnalyse)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.EphysProfile = QtWidgets.QCheckBox(self.verticalFrame)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.EphysProfile.setFont(font)
        self.EphysProfile.setChecked(True)
        self.EphysProfile.setObjectName("EphysProfile")
        self.horizontalLayout_4.addWidget(self.EphysProfile)
        self.AnalyseAll = QtWidgets.QCheckBox(self.verticalFrame)
        self.AnalyseAll.setChecked(True)
        self.AnalyseAll.setObjectName("AnalyseAll")
        self.horizontalLayout_4.addWidget(self.AnalyseAll)
        self.AnalyseOnlyIR = QtWidgets.QCheckBox(self.verticalFrame)
        self.AnalyseOnlyIR.setObjectName("AnalyseOnlyIR")
        self.horizontalLayout_4.addWidget(self.AnalyseOnlyIR)
        self.verticalLayout_5.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.CurrentInputLabel = QtWidgets.QLabel(self.verticalFrame)
        self.CurrentInputLabel.setObjectName("CurrentInputLabel")
        self.horizontalLayout_9.addWidget(self.CurrentInputLabel)
        self.CurrentInputVecIR = QtWidgets.QTextEdit(self.verticalFrame)
        self.CurrentInputVecIR.setMaximumSize(QtCore.QSize(16777215, 20))
        self.CurrentInputVecIR.setObjectName("CurrentInputVecIR")
        self.horizontalLayout_9.addWidget(self.CurrentInputVecIR)
        self.verticalLayout_5.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.AnalysePassive = QtWidgets.QCheckBox(self.verticalFrame)
        self.AnalysePassive.setChecked(True)
        self.AnalysePassive.setObjectName("AnalysePassive")
        self.verticalLayout_2.addWidget(self.AnalysePassive)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.LAbelIV = QtWidgets.QLabel(self.verticalFrame)
        self.LAbelIV.setMaximumSize(QtCore.QSize(47, 70))
        self.LAbelIV.setObjectName("LAbelIV")
        self.horizontalLayout_13.addWidget(self.LAbelIV)
        self.RangeIV = QtWidgets.QTextEdit(self.verticalFrame)
        self.RangeIV.setMaximumSize(QtCore.QSize(63, 70))
        self.RangeIV.setObjectName("RangeIV")
        self.horizontalLayout_13.addWidget(self.RangeIV)
        self.verticalLayout_2.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.AnalyseHpo = QtWidgets.QCheckBox(self.verticalFrame)
        self.AnalyseHpo.setChecked(True)
        self.AnalyseHpo.setObjectName("AnalyseHpo")
        self.verticalLayout_6.addWidget(self.AnalyseHpo)
        self.AnalyseJustSub = QtWidgets.QCheckBox(self.verticalFrame)
        self.AnalyseJustSub.setChecked(True)
        self.AnalyseJustSub.setObjectName("AnalyseJustSub")
        self.verticalLayout_6.addWidget(self.AnalyseJustSub)
        self.horizontalLayout_5.addLayout(self.verticalLayout_6)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.AnalyseAP = QtWidgets.QCheckBox(self.verticalFrame)
        self.AnalyseAP.setChecked(True)
        self.AnalyseAP.setObjectName("AnalyseAP")
        self.verticalLayout.addWidget(self.AnalyseAP)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.APThreshold = QtWidgets.QSpinBox(self.verticalFrame)
        self.APThreshold.setMinimum(-60)
        self.APThreshold.setMaximum(100)
        self.APThreshold.setSingleStep(10)
        self.APThreshold.setObjectName("APThreshold")
        self.horizontalLayout_11.addWidget(self.APThreshold)
        self.DetectAPThreshold = QtWidgets.QLabel(self.verticalFrame)
        self.DetectAPThreshold.setMaximumSize(QtCore.QSize(65, 40))
        self.DetectAPThreshold.setObjectName("DetectAPThreshold")
        self.horizontalLayout_11.addWidget(self.DetectAPThreshold)
        self.verticalLayout.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_5.addLayout(self.verticalLayout)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.AnalyseFiring = QtWidgets.QCheckBox(self.verticalFrame)
        self.AnalyseFiring.setChecked(True)
        self.AnalyseFiring.setObjectName("AnalyseFiring")
        self.verticalLayout_3.addWidget(self.AnalyseFiring)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.XTimesLPR = QtWidgets.QSpinBox(self.verticalFrame)
        self.XTimesLPR.setMaximum(10)
        self.XTimesLPR.setProperty("value", 3)
        self.XTimesLPR.setObjectName("XTimesLPR")
        self.horizontalLayout_6.addWidget(self.XTimesLPR)
        self.XTimesLPRLabel = QtWidgets.QLabel(self.verticalFrame)
        self.XTimesLPRLabel.setMaximumSize(QtCore.QSize(65, 40))
        self.XTimesLPRLabel.setObjectName("XTimesLPRLabel")
        self.horizontalLayout_6.addWidget(self.XTimesLPRLabel)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_5.addLayout(self.verticalLayout_3)
        self.verticalLayout_5.addLayout(self.horizontalLayout_5)
        self.Print = QtWidgets.QLabel(self.verticalFrame)
        self.Print.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.Print.setFont(font)
        self.Print.setObjectName("Print")
        self.verticalLayout_5.addWidget(self.Print)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.PrintAll = QtWidgets.QCheckBox(self.verticalFrame)
        self.PrintAll.setChecked(True)
        self.PrintAll.setObjectName("PrintAll")
        self.horizontalLayout_7.addWidget(self.PrintAll)
        self.PrintOV = QtWidgets.QCheckBox(self.verticalFrame)
        self.PrintOV.setChecked(True)
        self.PrintOV.setObjectName("PrintOV")
        self.horizontalLayout_7.addWidget(self.PrintOV)
        self.PrintPassive = QtWidgets.QCheckBox(self.verticalFrame)
        self.PrintPassive.setChecked(True)
        self.PrintPassive.setObjectName("PrintPassive")
        self.horizontalLayout_7.addWidget(self.PrintPassive)
        self.PrintHypo = QtWidgets.QCheckBox(self.verticalFrame)
        self.PrintHypo.setChecked(True)
        self.PrintHypo.setObjectName("PrintHypo")
        self.horizontalLayout_7.addWidget(self.PrintHypo)
        self.PrintJustSub = QtWidgets.QCheckBox(self.verticalFrame)
        self.PrintJustSub.setChecked(True)
        self.PrintJustSub.setObjectName("PrintJustSub")
        self.horizontalLayout_7.addWidget(self.PrintJustSub)
        self.PrintAP = QtWidgets.QCheckBox(self.verticalFrame)
        self.PrintAP.setChecked(True)
        self.PrintAP.setObjectName("PrintAP")
        self.horizontalLayout_7.addWidget(self.PrintAP)
        self.PrintFiring = QtWidgets.QCheckBox(self.verticalFrame)
        self.PrintFiring.setChecked(True)
        self.PrintFiring.setObjectName("PrintFiring")
        self.horizontalLayout_7.addWidget(self.PrintFiring)
        self.verticalLayout_5.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.StartAnalysis = QtWidgets.QPushButton(self.verticalFrame)
        self.StartAnalysis.setObjectName("StartAnalysis")
        self.horizontalLayout_8.addWidget(self.StartAnalysis)
        self.ShowCurrentCell = QtWidgets.QTextEdit(self.verticalFrame)
        self.ShowCurrentCell.setMaximumSize(QtCore.QSize(16777215, 40))
        self.ShowCurrentCell.setObjectName("ShowCurrentCell")
        self.horizontalLayout_8.addWidget(self.ShowCurrentCell)
        self.textEdit_2 = QtWidgets.QTextEdit(self.verticalFrame)
        self.textEdit_2.setMaximumSize(QtCore.QSize(16777215, 40))
        self.textEdit_2.setObjectName("textEdit_2")
        self.horizontalLayout_8.addWidget(self.textEdit_2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_8)
        self.progressBar = QtWidgets.QProgressBar(self.verticalFrame)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_5.addWidget(self.progressBar)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 22))
        self.menubar.setObjectName("menubar")
        self.menuDennis_Analysis = QtWidgets.QMenu(self.menubar)
        self.menuDennis_Analysis.setObjectName("menuDennis_Analysis")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuDennis_Analysis.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def startLoop(self):
        while True:
    #                time.sleep(0.05)
            value = self.progressBar.value() + 0.0001
            self.progressBar.setValue(value)
            app.processEvents()
#            if (not self._active or
#                value >= self.progressbar.maximum()):
#                break
    #            self.button.setText('Start')
#        self._active = False

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.SetWorkingDirectory.setText(_translate("MainWindow", "Set working directory:"))
        self.AnalyseOneCell.setText(_translate("MainWindow", "One cell"))
        self.AnalyseAllCells.setText(_translate("MainWindow", "All Cells"))
        self.AnalyseNewCells.setText(_translate("MainWindow", "New Cells"))
        self.AnalyseOldCells.setText(_translate("MainWindow", "Old Cells"))
        self.CellstoAnalyse.setText(_translate("MainWindow", "Cells to Analyse:"))
        self.Identifier.setText(_translate("MainWindow", "Identifier:"))
        self.IVCalcfrom.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">DD</p></body></html>"))
        self.OVTableIdentifier.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">ResultsTable</p><p align=\"center\">Name:</p></body></html>"))
        self.ResultsTableName.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">AllCells_Characterisation</p></body></html>"))
        self.SelectCellToAnalyse.setText(_translate("MainWindow", "Select Cell:"))
        self.EphysProfile.setText(_translate("MainWindow", "Ephys Profile"))
        self.AnalyseAll.setText(_translate("MainWindow", "Whole Analysis"))
        self.AnalyseOnlyIR.setText(_translate("MainWindow", "Only IR protocol"))
        self.CurrentInputLabel.setText(_translate("MainWindow", "IR Current Inputs (pA):"))
        self.CurrentInputVecIR.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">-200, -100, -50, -30, -20, -10, 10, 20, 30, 50, 100, 200, 300, 400, 500</p></body></html>"))
        self.AnalysePassive.setText(_translate("MainWindow", "Passive Properties"))
        self.LAbelIV.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">IV Calc </p><p align=\"center\">from:</p></body></html>"))
        self.RangeIV.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">-30, -20, -10, 10, 20, 30</p></body></html>"))
        self.AnalyseHpo.setText(_translate("MainWindow", "Hypo"))
        self.AnalyseJustSub.setText(_translate("MainWindow", "Just Sub"))
        self.AnalyseAP.setText(_translate("MainWindow", "AP Properties"))
        self.DetectAPThreshold.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Detect </p><p align=\"center\">AP </p><p align=\"center\">Threshold</p></body></html>"))
        self.AnalyseFiring.setText(_translate("MainWindow", "Firing Properties"))
        self.XTimesLPRLabel.setText(_translate("MainWindow", "Times LPR"))
        self.Print.setText(_translate("MainWindow", "Print:"))
        self.PrintAll.setText(_translate("MainWindow", "PrintAll"))
        self.PrintOV.setText(_translate("MainWindow", "Overview"))
        self.PrintPassive.setText(_translate("MainWindow", "Passive"))
        self.PrintHypo.setText(_translate("MainWindow", "Hypo"))
        self.PrintJustSub.setText(_translate("MainWindow", "JustSub"))
        self.PrintAP.setText(_translate("MainWindow", "AP"))
        self.PrintFiring.setText(_translate("MainWindow", "Firing"))
        self.StartAnalysis.setText(_translate("MainWindow", "Start Analysis"))
        self.menuDennis_Analysis.setTitle(_translate("MainWindow", "Analysis"))

        ''' Connect to Functions: '''
        # Set and Show: Working directory:
        def pick_new():
            dialog = QFileDialog()
            folder_path = dialog.getExistingDirectory(None, "Select Folder")
            self.ShowWorkingDirectory.setText(folder_path)
            os.chdir(folder_path)
            return folder_path
        self.SetWorkingDirectory.clicked.connect(pick_new)
        
        # Set and Show Single Cell: 
        def pick_Cell():
            dialog = QFileDialog()
            Singlefolder_path = dialog.getExistingDirectory(None, "Select Folder")
            os.chdir(Singlefolder_path)
            a = Singlefolder_path.split("/")
            self.ShowCellToAnalyse.setText(a[-1])
            return Singlefolder_path
        
        self.SelectCellToAnalyse.clicked.connect(pick_Cell)
        
        
        ''' Start The Analysis: '''
        def Start_Analysis():
########### First: Run Over Check-Boxes and Take Values:
        # A) To Calculate:
            if self.AnalyseAll.isChecked():
                Conditions = {'ToCalcAll':1} 
            else:
                Conditions = {'ToCalcAll':0}
            if self.AnalysePassive.isChecked():
                Conditions['ToCalcPassive']=1
            else:
                Conditions['ToCalcPassive']=0
            if self.AnalyseHpo.isChecked():
                Conditions['ToCalcHypo']=1
            else:
                Conditions['ToCalcHypo']=0
            if self.AnalyseJustSub.isChecked():
                Conditions['ToCalcJustSub']=1
            else:
                Conditions['ToCalcJustSub']=0
            if self.AnalyseAP.isChecked():
                Conditions['ToCalcAP']=1
            else:
                Conditions['ToCalcAP']=0
            if self.AnalyseFiring.isChecked():
                Conditions['ToCalcFiring']=1
            else:
                Conditions['ToCalcFiring']=0
                
        # B) To Print:
            if self.PrintAll.isChecked():
                Conditions['PrintShowAll']=1
            else:
                Conditions['PrintShowAll']=0
            if self.PrintPassive.isChecked():
                Conditions['PrintShowPassive']=1 
            else:
                Conditions['PrintShowPassive']=0
            if self.PrintHypo.isChecked():
                Conditions['PrintShowHypo']=1
            else:
                Conditions['PrintShowHypo']=0
            if self.PrintJustSub.isChecked():
                Conditions['PrintShowJustSub']=1
            else:
                Conditions['PrintShowJustSub']=0
            if self.PrintAP.isChecked():
                Conditions['PrintShowAP']=1
            else:
                Conditions['PrintShowAP']=0
            if self.PrintFiring.isChecked():
                Conditions['PrintShowFiring']= 1 
            else:
                Conditions['PrintShowFiring']= 0
            if self.PrintOV.isChecked():
                Conditions['PrintOV']=1
            else:
                Conditions['PrintOV']=1
                
        # C) Get Folder Conditions:
            if self.AnalyseOneCell.isChecked():
                FolderConditions = {'AnalyseOneCell':1} 
            else:
                FolderConditions = {'AnalyseOneCell':0}
            if self.AnalyseAllCells.isChecked():
                FolderConditions['AnalyseAllCells']=1 
            else:
                FolderConditions['AnalyseAllCells']=0   
            if self.AnalyseNewCells.isChecked():
                FolderConditions['AnalyseNewCells']=1 
            else:
                FolderConditions['AnalyseNewCells']=0
            if self.AnalyseOldCells.isChecked():
                FolderConditions['AnalyseOldCells']=1 
            else:
                FolderConditions['AnalyseOldCells']=0
        # D) Get Texts:
            Identifier = self.IVCalcfrom.toPlainText()
            ResultsTableName = self.ResultsTableName.toPlainText() 
        # F) Texts/Numbers for Conditions:
            Conditions['AmpApThres'] = self.APThreshold.value() 
            XTimesLPR = self.XTimesLPR.value() 
            Conditions['LPRXTime'] = 'LPR'+str(XTimesLPR)
            CurrentAmpsBasic1 = self.CurrentInputVecIR.toPlainText() 
            Conditions['CurrentAmpsBasic'] = np.fromstring(CurrentAmpsBasic1, dtype=int, sep=", ")
            IVRange = self.RangeIV.toPlainText() 
            Conditions['PassiveRange'] = np.fromstring(IVRange,dtype=np.float, sep=", ")
            if self.AnalyseOnlyIR.isChecked():
                FolderConditions['AnalyseOnlyIR']=1 
            else:
                FolderConditions['AnalyseOnlyIR']=0
        # G) Get folder_path:
            folder_path = os.getcwd()
            print(folder_path)
            print(Conditions)
######### Select Cells to Analyse:    
            A = DDFolderLevel.WhatToAnalyse(os.getcwd(),FolderConditions,ResultsTableName,Identifier) 
            CellsToAnalyse = A.CellsToAnalyse
            NumCellsToAnalyse = A.NumCells
            print(NumCellsToAnalyse)
            print(CellsToAnalyse)
            i = 0
#            AnalysedCells = [None]*NumCellsToAnalyse
            
######### Set ProgressBar:
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(NumCellsToAnalyse)
            
            ''' Start Anlaysis: '''
            NotAnalysed =[]
            while i < NumCellsToAnalyse:
                self.progressBar.setValue(i+1)
#                plt.close('all') # closes all figures > Save Memory!!
                
#                print(CellsToAnalyse[i])
                
#                if os.path.isfile(ResultsTableName+'.xlsx'):
#                    OVTable,X = DDImport.ImportExcel(ResultsTableName) 
#                else:
#                    OVTable = None

                # Go to CellFolder:
                if NumCellsToAnalyse > 1:  
                    if os.getcwd() != folder_path: # return to folder_path, when not already there!
                        os.chdir(folder_path)
                    os.chdir(CellsToAnalyse[i])
                    self.ShowCurrentCell.setText(CellsToAnalyse[i])
                    CellName = CellsToAnalyse[i]
                else:
                    if type(CellsToAnalyse) == list:
                        CellsToAnalyse = CellsToAnalyse[0]
                    
                    os.chdir(CellsToAnalyse)
                    self.ShowCurrentCell.setText(CellsToAnalyse) 
                    CellName = CellsToAnalyse
                    
                TextNumCells = '=> %.0f of %.0f cells' % ((i+1), NumCellsToAnalyse)
                self.textEdit_2.setText(TextNumCells)
                app.processEvents()
                
######### RUN MAIN SCRIPT:
                try:
                    print(CellName)
#                    AnalysedCells[i] = DDCellLevel.AnalysisCellCharacteristics(CellName,Conditions,ResultsTableName)
                    DDCellLevel.AnalysisCellCharacteristics(CellName,Conditions,ResultsTableName)                    
                    os.chdir("..")
                    i += 1
                except:
                    NotAnalysed.append(CellsToAnalyse[i])
                    os.chdir("..")
                    i += 1
                    continue 
            
          
######### Print Names, that are not Analysed:
            if NotAnalysed:
                os.chdir("..")
                df = pd.DataFrame({'Not_Analysed':NotAnalysed})
                ddhelp.SaveAsExcelFile(df,(ResultsTableName+'_NOT_Analysed'))
    
        # Do it:
        self.StartAnalysis.clicked.connect(Start_Analysis)
        
#        def fileQuit(self):
#            self.close()
#
#        def closeEvent(self, ce):
#            self.fileQuit()



''' Run GUI: '''
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

