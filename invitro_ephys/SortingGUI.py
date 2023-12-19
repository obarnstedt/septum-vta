# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SortingGUI.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

''' Import Needed Scripts: '''
import os
import DDFolderLevel
import DDSortingTracesEphysProfile as SortingProgramm
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer


''' Main Window: '''
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(484, 355)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 1000))
        self.verticalFrame_2 = QtWidgets.QWidget(MainWindow)
        self.verticalFrame_2.setMaximumSize(QtCore.QSize(667, 675))
        self.verticalFrame_2.setObjectName("verticalFrame_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalFrame_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(0, -1, -1, -1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.SetWorkingDirectory = QtWidgets.QPushButton(self.verticalFrame_2)
        self.SetWorkingDirectory.setMinimumSize(QtCore.QSize(150, 25))
        self.SetWorkingDirectory.setMaximumSize(QtCore.QSize(150, 25))
        self.SetWorkingDirectory.setObjectName("SetWorkingDirectory")
        self.horizontalLayout_3.addWidget(self.SetWorkingDirectory)
        self.ShowWorkingDirectory = QtWidgets.QTextEdit(self.verticalFrame_2)
        self.ShowWorkingDirectory.setMinimumSize(QtCore.QSize(250, 50))
        self.ShowWorkingDirectory.setMaximumSize(QtCore.QSize(250, 50))
        self.ShowWorkingDirectory.setObjectName("ShowWorkingDirectory")
        self.horizontalLayout_3.addWidget(self.ShowWorkingDirectory)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.SelectFolder = QtWidgets.QPushButton(self.verticalFrame_2)
        self.SelectFolder.setMinimumSize(QtCore.QSize(150, 25))
        self.SelectFolder.setMaximumSize(QtCore.QSize(150, 25))
        self.SelectFolder.setObjectName("SelectFolder")
        self.horizontalLayout.addWidget(self.SelectFolder)
        self.ShowSelectedFolder = QtWidgets.QTextEdit(self.verticalFrame_2)
        self.ShowSelectedFolder.setMinimumSize(QtCore.QSize(250, 25))
        self.ShowSelectedFolder.setMaximumSize(QtCore.QSize(250, 25))
        self.ShowSelectedFolder.setObjectName("ShowSelectedFolder")
        self.horizontalLayout.addWidget(self.ShowSelectedFolder)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.verticalFrame_2)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.InputTable = QtWidgets.QTextEdit(self.verticalFrame_2)
        self.InputTable.setMinimumSize(QtCore.QSize(150, 25))
        self.InputTable.setMaximumSize(QtCore.QSize(150, 25))
        self.InputTable.setObjectName("InputTable")
        self.horizontalLayout_2.addWidget(self.InputTable)
        self.label_2 = QtWidgets.QLabel(self.verticalFrame_2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.NumeberIRTraces = QtWidgets.QSpinBox(self.verticalFrame_2)
        self.NumeberIRTraces.setMinimumSize(QtCore.QSize(50, 25))
        self.NumeberIRTraces.setMaximumSize(QtCore.QSize(50, 25))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.NumeberIRTraces.setFont(font)
        self.NumeberIRTraces.setMaximum(50)
        self.NumeberIRTraces.setProperty("value", 15)
        self.NumeberIRTraces.setObjectName("NumeberIRTraces")
        self.horizontalLayout_2.addWidget(self.NumeberIRTraces)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.CompareWithTable = QtWidgets.QCheckBox(self.verticalFrame_2)
        self.CompareWithTable.setMinimumSize(QtCore.QSize(180, 25))
        self.CompareWithTable.setMaximumSize(QtCore.QSize(180, 25))
        self.CompareWithTable.setObjectName("CompareWithTable")
        self.horizontalLayout_4.addWidget(self.CompareWithTable)
        self.TableToCompareWith = QtWidgets.QTextEdit(self.verticalFrame_2)
        self.TableToCompareWith.setMinimumSize(QtCore.QSize(150, 25))
        self.TableToCompareWith.setMaximumSize(QtCore.QSize(150, 25))
        self.TableToCompareWith.setObjectName("TableToCompareWith")
        self.horizontalLayout_4.addWidget(self.TableToCompareWith)
        self.label_6 = QtWidgets.QLabel(self.verticalFrame_2)
        self.label_6.setMinimumSize(QtCore.QSize(60, 25))
        self.label_6.setMaximumSize(QtCore.QSize(60, 25))
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_4.addWidget(self.label_6)
        self.IdentifierDisplay = QtWidgets.QTextEdit(self.verticalFrame_2)
        self.IdentifierDisplay.setMinimumSize(QtCore.QSize(50, 25))
        self.IdentifierDisplay.setMaximumSize(QtCore.QSize(50, 25))
        self.IdentifierDisplay.setObjectName("IdentifierDisplay")
        self.horizontalLayout_4.addWidget(self.IdentifierDisplay)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.StartSorting = QtWidgets.QPushButton(self.verticalFrame_2)
        self.StartSorting.setMinimumSize(QtCore.QSize(200, 100))
        self.StartSorting.setMaximumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.StartSorting.setFont(font)
        self.StartSorting.setMouseTracking(False)
        self.StartSorting.setObjectName("StartSorting")
        self.horizontalLayout_5.addWidget(self.StartSorting)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_5 = QtWidgets.QLabel(self.verticalFrame_2)
        self.label_5.setMinimumSize(QtCore.QSize(70, 25))
        self.label_5.setMaximumSize(QtCore.QSize(70, 25))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_7.addWidget(self.label_5)
        self.DisplayActuallCellName = QtWidgets.QTextEdit(self.verticalFrame_2)
        self.DisplayActuallCellName.setMinimumSize(QtCore.QSize(150, 25))
        self.DisplayActuallCellName.setMaximumSize(QtCore.QSize(150, 25))
        self.DisplayActuallCellName.setObjectName("DisplayActuallCellName")
        self.horizontalLayout_7.addWidget(self.DisplayActuallCellName)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.ActualCell = QtWidgets.QTextEdit(self.verticalFrame_2)
        self.ActualCell.setMinimumSize(QtCore.QSize(50, 25))
        self.ActualCell.setMaximumSize(QtCore.QSize(50, 25))
        self.ActualCell.setObjectName("ActualCell")
        self.horizontalLayout_6.addWidget(self.ActualCell)
        self.label_3 = QtWidgets.QLabel(self.verticalFrame_2)
        self.label_3.setMinimumSize(QtCore.QSize(25, 30))
        self.label_3.setMaximumSize(QtCore.QSize(30, 25))
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_6.addWidget(self.label_3)
        self.AllCells = QtWidgets.QTextEdit(self.verticalFrame_2)
        self.AllCells.setMinimumSize(QtCore.QSize(50, 25))
        self.AllCells.setMaximumSize(QtCore.QSize(50, 25))
        self.AllCells.setObjectName("AllCells")
        self.horizontalLayout_6.addWidget(self.AllCells)
        self.label_4 = QtWidgets.QLabel(self.verticalFrame_2)
        self.label_4.setMinimumSize(QtCore.QSize(40, 25))
        self.label_4.setMaximumSize(QtCore.QSize(40, 25))
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_6.addWidget(self.label_4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.progressBar = QtWidgets.QProgressBar(self.verticalFrame_2)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_2.addWidget(self.progressBar)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        MainWindow.setCentralWidget(self.verticalFrame_2)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 484, 22))
        self.menubar.setObjectName("menubar")
        self.menuDennis_Analysis = QtWidgets.QMenu(self.menubar)
        self.menuDennis_Analysis.setObjectName("menuDennis_Analysis")
        self.menuSorting_Cells = QtWidgets.QMenu(self.menubar)
        self.menuSorting_Cells.setObjectName("menuSorting_Cells")
        MainWindow.setMenuBar(self.menubar)
        self.menubar.addAction(self.menuDennis_Analysis.menuAction())
        self.menubar.addAction(self.menuSorting_Cells.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.SetWorkingDirectory.setText(_translate("MainWindow", "Select Path:"))
        self.SelectFolder.setText(_translate("MainWindow", "Select Specific Cell"))
        self.label.setText(_translate("MainWindow", "Input Table:"))
        self.InputTable.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">SortingOV</p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Traces for IR:"))
        self.CompareWithTable.setText(_translate("MainWindow", "Take only Cell from this list:"))
        self.label_6.setText(_translate("MainWindow", "Identifier: "))
        self.IdentifierDisplay.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">DD</p></body></html>"))
        self.StartSorting.setText(_translate("MainWindow", "Start Sorting"))
        self.label_5.setText(_translate("MainWindow", "Progress:"))
        self.label_3.setText(_translate("MainWindow", "from"))
        self.label_4.setText(_translate("MainWindow", "Cells"))
        self.menuDennis_Analysis.setTitle(_translate("MainWindow", "Plotting"))
        self.menuSorting_Cells.setTitle(_translate("MainWindow", "Sorting Cells"))





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
#            os.chdir(Singlefolder_path)
            a = Singlefolder_path.split("/")
            self.ShowSelectedFolder.setText(a[-1])
            return Singlefolder_path
        self.SelectFolder.clicked.connect(pick_Cell)
        
        ''' Start The Analysis: '''
        def Sorting():
            Identifier = self.IdentifierDisplay.toPlainText()
            # Get Values:
            # IR
            NumIR = int(self.NumeberIRTraces.value())
            Conditions = {'NumIRTraces':NumIR}     
            # TableToSort:
            TableToSort = self.InputTable.toPlainText()
            TableToSortOriginal = TableToSort+'.xlsx'
            if TableToSort is not '':
                Conditions ['OVSortingTableName'] = TableToSortOriginal
            else:
                Conditions ['OVSortingTableName'] = None 
                
            # What to Sort
            SpecificCell = self.ShowSelectedFolder.toPlainText()
            
            if SpecificCell:
                CellsToAnalyse = [SpecificCell]
                
            elif self.CompareWithTable.isChecked():
                FolderConditions = {'AnalyseOneCell':0}
                FolderConditions['AnalyseAllCells']=0 
                FolderConditions['AnalyseNewCells']=0
                FolderConditions['AnalyseOldCells']=1
                CellTable = self.TableToCompareWith.toPlainText()
                A = DDFolderLevel.WhatToAnalyse(os.getcwd(),FolderConditions,CellTable,Identifier) 
                CellsToAnalyse = A.CellsToAnalyse  
            else:
                CellsToAnalyse1 =  [name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name))]    
                CellsToAnalyse = [name for name in CellsToAnalyse1 if name.startswith(Identifier)]

            NumCellsToAnalyse = len(CellsToAnalyse)
            folder_path = os.getcwd()
            print(folder_path)
            print(Conditions)
            print(CellsToAnalyse)
            # Set ProgressBar:
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(NumCellsToAnalyse)
            ''' Start Sorting: '''
            # While-Loop:
            i = 0
            while i < NumCellsToAnalyse:
                self.progressBar.setValue(i+1)
                self.AllCells.setText(str(NumCellsToAnalyse))
                
                if NumCellsToAnalyse > 1:  
                    if os.getcwd() != folder_path: # return to folder_path, when not already there!
                        os.chdir(folder_path)
                    os.chdir(CellsToAnalyse[i])
                    CellName = CellsToAnalyse[i]
                else:
                    if type(CellsToAnalyse) == list:
                        CellsToAnalyse = CellsToAnalyse[0]
                    os.chdir(CellsToAnalyse)
                    CellName = CellsToAnalyse
                    
                # Displaying Stuff:
                self.DisplayActuallCellName.setText(CellName)
                self.ActualCell.setText(str(int(i+1)))
                app.processEvents()
                    
                try:
                    print(CellName)
                    print(os.getcwd())
                    # MainScript:
                    SortingProgramm.SortingCellCharacteristics(CellName,Conditions)
                    
                    os.chdir("..")
                    i += 1
                except:
                    print('Error')
                    os.chdir("..")
                    i += 1
                    continue
                
        # Do it:
        self.StartSorting.clicked.connect(Sorting)
       







if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

