# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PlottingGUI.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!
import os
import DDImport
import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt 
from matplotlib import gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import math

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
 


''' Classes: ''' 

class ImportFolder():
    def __init__ (self,Foldername):
         self.files, self.Waves,self.TimeVec,self.SampFreq,self.RecTime = DDImport.ImportFolder(Foldername)
         self.NumPlots = len(self.Waves)
         
class PlotFolderClass(FigureCanvas):    
    def __init__(self, Times, Waves,filesName, parent=None, width=5, height=4, dpi=300):
        NumSubplt = math.ceil(math.sqrt(len(Times)))
        plt.ioff()
        gs = gridspec.GridSpec(NumSubplt, NumSubplt)
        gs.update(left=0.1, bottom= 0.1, top = 0.9, right=0.9, wspace=0.1)
        fig,axes = plt.subplots(NumSubplt,NumSubplt)
        fig.set_size_inches(width,height)
        fig.set_dpi(dpi)
        Matrix = np.zeros((int(NumSubplt),int(NumSubplt)))
        AxesMatrix = np.where(Matrix==0)
        
        # YLimits:
        yMinLimits = [None]*len(Times)
        yMaxLimits = [None]*len(Times)
        i = 0
        while i < len(Times):
            yMinLimits[i] = np.min(Waves[i])
            yMaxLimits[i] = np.max(Waves[i])
            i +=1
        # Plotting:
        i = 0
        while i < len(Times):
            axes[AxesMatrix[0][i],AxesMatrix[1][i]]=plt.subplot(gs[AxesMatrix[0][i],AxesMatrix[1][i]])
            axes[AxesMatrix[0][i],AxesMatrix[1][i]].plot(Times[i],Waves[i],lw = 0.5, color = 'k')
            axes[AxesMatrix[0][i],AxesMatrix[1][i]].set_title(filesName[i], y = 0.95,fontsize = 4, fontweight='bold')
            # Limits:
            axes[AxesMatrix[0][i],AxesMatrix[1][i]].set_xlim([np.min(Times[i]),np.max(Times[i])])
            axes[AxesMatrix[0][i],AxesMatrix[1][i]].set_ylim([np.min(yMinLimits),np.max(yMaxLimits)])
            # Axes Appearance:
            axes[AxesMatrix[0][i],AxesMatrix[1][i]].spines["top"].set_visible(False)
            axes[AxesMatrix[0][i],AxesMatrix[1][i]].spines["right"].set_visible(False)
            if np.max(Times[i]) < 1000:
                axes[AxesMatrix[0][i],AxesMatrix[1][i]].xaxis.set_ticks(np.arange(0,np.max(Times[i]),100))
            else:
                axes[AxesMatrix[0][i],AxesMatrix[1][i]].xaxis.set_ticks(np.arange(0,np.max(Times[i]),np.max(Times[i])/10))
                
            if i < (len(Times)-NumSubplt):
                axes[AxesMatrix[0][i],AxesMatrix[1][i]].get_xaxis().set_ticklabels([])
            else:
                axes[AxesMatrix[0][i],AxesMatrix[1][i]].set_xlabel('ms',fontsize = 4)
                axes[AxesMatrix[0][i],AxesMatrix[1][i]].xaxis.set_label_coords(0.5, -0.05)
                plt.setp(axes[AxesMatrix[0][i],AxesMatrix[1][i]].get_xticklabels(),rotation = 45, fontsize = 4)
            if AxesMatrix[1][i] != 0:
                axes[AxesMatrix[0][i],AxesMatrix[1][i]].get_yaxis().set_ticklabels([])
            else:
                axes[AxesMatrix[0][i],AxesMatrix[1][i]].set_ylabel('mV',fontsize = 4, rotation = 90)
                axes[AxesMatrix[0][i],AxesMatrix[1][i]].yaxis.set_label_coords(-0.05, 0.5)
                plt.setp(axes[AxesMatrix[0][i],AxesMatrix[1][i]].get_yticklabels(),rotation = 'vertical', fontsize = 4)
            i += 1
        # Delete Subplots not used: 
        while i < NumSubplt*NumSubplt:
            axes[AxesMatrix[0][i],AxesMatrix[1][i]]=plt.subplot(gs[AxesMatrix[0][i],AxesMatrix[1][i]])
            axes[AxesMatrix[0][i],AxesMatrix[1][i]].plot(0,0,lw = 1, color = 'w')
            axes[AxesMatrix[0][i],AxesMatrix[1][i]].axis('off')
            i += 1
            
        
        # Set FigureCanvas: 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        #SetToolBox
#        self.toolbar = NavigationToolbar(parent,FigureCanvas)

class PlotSingleClass(FigureCanvas):    
    def __init__(self, Times, Waves,filesName, parent=None, width=5, height=4, dpi=210):
        
        fig = Figure(figsize=(width, height), dpi=dpi)
        axes = fig.add_subplot(111)
        
        # Plotting:
        axes.plot(Times,Waves,lw = 0.5, color = 'k')
        axes.set_title(filesName, y = 1,fontsize = 4, fontweight='bold')
        # Axis Appearance:
        axes.set_xlim([np.min(Times),np.max(Times)]) 
        axes.set_xlabel('ms',fontsize = 4)
        axes.xaxis.set_label_coords(0.5, -0.1)
        axes.tick_params(axis='both', which='major', labelsize=4)
        axes.set_ylabel('mV',fontsize = 4, rotation = 90)
        axes.yaxis.set_label_coords(-0.1, 0.5)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        
        # Set FigureCanvas: 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

def CustumisedPlots(self,Wave,SampFreq):
    Boarders = {'PointsFrom': self.TimeWindowFrom.value()*SampFreq}
    Boarders['PointsTo'] = self.TimeWindowTo.value()*SampFreq 
    Boarders['TimeTo'] = self.TimeWindowTo.value()
    Boarders['TimeFrom'] = self.TimeWindowFrom.value()
    TimeDeltaText = self.TimeWindowDelta.toPlainText()
    TimeDelta1 = np.fromstring(TimeDeltaText, dtype=int, sep=", ")
    if np.size(TimeDelta1) != 0:
        TimeDelta = TimeDelta1[0]*SampFreq
        if TimeDelta > 0:
            Boarders['PointsTo'] = Boarders['PointsFrom'] + TimeDelta
            self.TimeWindowTo.setProperty("value", int(Boarders['PointsTo']/SampFreq))
            Boarders['TimeTo'] = self.TimeWindowTo.value()
            Boarders['TimeFrom'] = self.TimeWindowFrom.value()
        else:
            TimeDelta = (Boarders['PointsTo']- Boarders['PointsFrom'])/SampFreq
            Boarders['TimeTo'] = self.TimeWindowTo.value()
            Boarders['TimeFrom'] = self.TimeWindowFrom.value()
            self.TimeWindowDelta.setText(str(TimeDelta))    
        
    if self.checkBox.isChecked():
        BaselineFrom = self.CalcBaselineFrom.value()
        BaselineTo = self.CalcBaselineTo.value()
        BaselineDeltaText = self.CalcBaselineDelta.toPlainText()
        BaselineDelta1 = np.fromstring(BaselineDeltaText, dtype=int, sep=", ")
        if np.size(BaselineDelta1) != 0 and BaselineDelta1[0] > 0:
            BaselineDelta = BaselineDelta1[0]
            BaselineTo = BaselineFrom + BaselineDelta 
            self.CalcBaselineTo.setProperty("value", int(BaselineTo))
            self.CalcBaselineDelta.setText(str(BaselineDelta)) 
            
        self.CalcBaselineDelta.setText(str(int(BaselineTo-BaselineFrom))) 
        Baseline = np.mean(Wave[int(BaselineFrom*SampFreq):int(BaselineTo*SampFreq)])
        AmpFrom = self.BaselineMinus.value()
        AmpTo = self.BaselinePlus.value()
        Boarders['VmFrom'] = Baseline+AmpFrom  
        Boarders['VmTo'] = Baseline+AmpTo
        VmBaselineDeltaText = self.textEdit.toPlainText()
        VmDelta1 = np.fromstring(VmBaselineDeltaText, dtype=int, sep=", ")
        if np.size(VmDelta1) != 0:
            VmDelta = VmDelta1[0]
            if VmDelta > 0:
                Boarders['VmTo'] = Boarders['VmFrom'] + VmDelta 
                self.BaselinePlus.setProperty("value", int(Boarders['VmTo']-Baseline))
            else:
                VmDelta = (Boarders['VmTo']- Boarders['VmFrom'])
                self.textEdit.setText(str(VmDelta))     
        
    elif self.checkBox_2.isChecked():
        Boarders['VmFrom'] = self.mVFrom.value()
        Boarders['VmTo'] = self.mVTo.value() 
        VmDeltaText = self.mVDelta.toPlainText()
        VmDelta1 = np.fromstring(VmDeltaText, dtype=int, sep=", ")
        if np.size(VmDelta1) != 0:
            VmDelta = VmDelta1[0]
            if VmDelta > 0:
                Boarders['VmTo'] = Boarders['VmFrom'] + VmDelta
                self.mVTo.setProperty("value", int(Boarders['VmTo']))
            else:
                VmDelta = (Boarders['VmTo']- Boarders['VmFrom'])
                self.mVDelta.setText(str(VmDelta))     
    return Boarders
    
class PlotSingleClassWithBoarders (FigureCanvas):    
    def __init__(self, Times, Waves, SampFreq,filesName, Boarders, ToPrint,CellNamePrint,\
                 ScaleMs=np.nan,ScalemV=np.nan,\
                 parent=None, width=5, height=4, dpi=210):
        
        fig = Figure(figsize=(width, height), dpi=dpi)
        axes = fig.add_subplot(111)
        
        #Scale:
        if ScaleMs is not np.nan and ScalemV is not np.nan:
            ScaleBarTimeOn = int(Boarders['PointsTo']-ScaleMs)/SampFreq 
            ScaleBarTimeTo = Boarders['PointsTo']/SampFreq
            ScaleBarVmOn = Boarders['VmTo']-ScalemV
            ScaleBarVmTo = Boarders['VmTo']
            axes.plot([ScaleBarTimeOn,ScaleBarTimeTo],[ScaleBarVmTo,ScaleBarVmTo],'-r',lw = 1)
            axes.plot([ScaleBarTimeTo,ScaleBarTimeTo],[ScaleBarVmOn,ScaleBarVmTo],'-r',lw = 1)
        
        # Plotting:
        axes.plot(Times[int(Boarders['PointsFrom']):int(Boarders['PointsTo'])],Waves[int(Boarders['PointsFrom']):int(Boarders['PointsTo'])],lw = 0.5, color = 'k')
        axes.set_title(filesName, y = 1,fontsize = 4, fontweight='bold')
        # Axis Appearance:
        axes.set_xlim([Boarders['TimeFrom'],Boarders['TimeTo']]) 
        axes.set_ylim([Boarders['VmFrom'],Boarders['VmTo']]) 
        axes.set_xlabel('ms',fontsize = 4)
        axes.xaxis.set_label_coords(0.5, -0.1)
        axes.tick_params(axis='both', which='major', labelsize=4)
        axes.set_ylabel('mV',fontsize = 4, rotation = 90)
        axes.yaxis.set_label_coords(-0.1, 0.5)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        
        # Printing:
        if ToPrint ==1:
           CanvasPersonal = FigureCanvas(fig)
           CanvasPersonal.print_figure(CellNamePrint+".svg",format='svg',transparent=True, dpi=1200)
#            plt.savefig(CellNamePrint+".jpg",format='jpg', dpi=1200)
#            plt.ion()
        # Set FigureCanvas: 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(621, 685)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(600, 675))
        MainWindow.setMaximumSize(QtCore.QSize(648, 685))
        self.verticalFrame_2 = QtWidgets.QWidget(MainWindow)
        self.verticalFrame_2.setMaximumSize(QtCore.QSize(667, 675))
        self.verticalFrame_2.setObjectName("verticalFrame_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalFrame_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.ShowingTabe = QtWidgets.QTabWidget(self.verticalFrame_2)
        self.ShowingTabe.setMinimumSize(QtCore.QSize(600, 700))
        self.ShowingTabe.setMaximumSize(QtCore.QSize(600, 700))
        self.ShowingTabe.setObjectName("ShowingTabe")
        self.ShowingTraces = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ShowingTraces.sizePolicy().hasHeightForWidth())
        self.ShowingTraces.setSizePolicy(sizePolicy)
        self.ShowingTraces.setObjectName("ShowingTraces")
        self.verticalFrame = QtWidgets.QFrame(self.ShowingTraces)
        self.verticalFrame.setGeometry(QtCore.QRect(0, 0, 600, 600))
        self.verticalFrame.setMinimumSize(QtCore.QSize(600, 600))
        self.verticalFrame.setMaximumSize(QtCore.QSize(600, 600))
        self.verticalFrame.setObjectName("verticalFrame")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalFrame)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(0, -1, -1, -1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.SetWorkingDirectory = QtWidgets.QPushButton(self.verticalFrame)
        self.SetWorkingDirectory.setMinimumSize(QtCore.QSize(128, 0))
        self.SetWorkingDirectory.setMaximumSize(QtCore.QSize(120, 16777215))
        self.SetWorkingDirectory.setObjectName("SetWorkingDirectory")
        self.horizontalLayout_3.addWidget(self.SetWorkingDirectory)
        self.ShowWorkingDirectory = QtWidgets.QTextEdit(self.verticalFrame)
        self.ShowWorkingDirectory.setMinimumSize(QtCore.QSize(150, 0))
        self.ShowWorkingDirectory.setMaximumSize(QtCore.QSize(200, 25))
        self.ShowWorkingDirectory.setObjectName("ShowWorkingDirectory")
        self.horizontalLayout_3.addWidget(self.ShowWorkingDirectory)
        self.SelectFolder = QtWidgets.QPushButton(self.verticalFrame)
        self.SelectFolder.setMaximumSize(QtCore.QSize(120, 16777215))
        self.SelectFolder.setObjectName("SelectFolder")
        self.horizontalLayout_3.addWidget(self.SelectFolder)
        self.ShowSelectedFolder = QtWidgets.QTextEdit(self.verticalFrame)
        self.ShowSelectedFolder.setMinimumSize(QtCore.QSize(150, 0))
        self.ShowSelectedFolder.setMaximumSize(QtCore.QSize(150, 25))
        self.ShowSelectedFolder.setObjectName("ShowSelectedFolder")
        self.horizontalLayout_3.addWidget(self.ShowSelectedFolder)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.PlotFolder = QtWidgets.QPushButton(self.verticalFrame)
        self.PlotFolder.setObjectName("PlotFolder")
        self.horizontalLayout_2.addWidget(self.PlotFolder)
        self.PlotSingle = QtWidgets.QPushButton(self.verticalFrame)
        self.PlotSingle.setObjectName("PlotSingle")
        self.horizontalLayout_2.addWidget(self.PlotSingle)
        self.PlotNumber = QtWidgets.QTextEdit(self.verticalFrame)
        self.PlotNumber.setMinimumSize(QtCore.QSize(0, 0))
        self.PlotNumber.setMaximumSize(QtCore.QSize(40, 25))
        self.PlotNumber.setObjectName("PlotNumber")
        self.horizontalLayout_2.addWidget(self.PlotNumber)
        self.Of = QtWidgets.QLabel(self.verticalFrame)
        self.Of.setMaximumSize(QtCore.QSize(16, 20))
        self.Of.setObjectName("Of")
        self.horizontalLayout_2.addWidget(self.Of)
        self.NumAllTraces = QtWidgets.QTextEdit(self.verticalFrame)
        self.NumAllTraces.setMaximumSize(QtCore.QSize(40, 25))
        self.NumAllTraces.setObjectName("NumAllTraces")
        self.horizontalLayout_2.addWidget(self.NumAllTraces)
        self.Traces = QtWidgets.QLabel(self.verticalFrame)
        self.Traces.setMaximumSize(QtCore.QSize(44, 20))
        self.Traces.setObjectName("Traces")
        self.horizontalLayout_2.addWidget(self.Traces)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(-1, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.PreviousPlot = QtWidgets.QPushButton(self.verticalFrame)
        self.PreviousPlot.setMaximumSize(QtCore.QSize(120, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.PreviousPlot.setFont(font)
        self.PreviousPlot.setIconSize(QtCore.QSize(16, 16))
        self.PreviousPlot.setObjectName("PreviousPlot")
        self.verticalLayout_2.addWidget(self.PreviousPlot)
        self.NextPlot = QtWidgets.QPushButton(self.verticalFrame)
        self.NextPlot.setMaximumSize(QtCore.QSize(120, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.NextPlot.setFont(font)
        self.NextPlot.setObjectName("NextPlot")
        self.verticalLayout_2.addWidget(self.NextPlot)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.PlottingWindow = QtWidgets.QGraphicsView(self.verticalFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PlottingWindow.sizePolicy().hasHeightForWidth())
        self.PlottingWindow.setSizePolicy(sizePolicy)
        self.PlottingWindow.setMinimumSize(QtCore.QSize(575, 575))
        self.PlottingWindow.setMaximumSize(QtCore.QSize(575, 575))
        self.PlottingWindow.setObjectName("PlottingWindow")
        self.verticalLayout_5.addWidget(self.PlottingWindow)
        self.PlottingWindow.raise_()
        self.ShowingTabe.addTab(self.ShowingTraces, "")
        self.PrintingTraces = QtWidgets.QWidget()
        self.PrintingTraces.setObjectName("PrintingTraces")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.PrintingTraces)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 589, 37))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.SetWorkingDirectory2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.SetWorkingDirectory2.setObjectName("SetWorkingDirectory2")
        self.horizontalLayout.addWidget(self.SetWorkingDirectory2)
        self.ShowWorkingDirectory2 = QtWidgets.QTextBrowser(self.horizontalLayoutWidget)
        self.ShowWorkingDirectory2.setMinimumSize(QtCore.QSize(150, 0))
        self.ShowWorkingDirectory2.setMaximumSize(QtCore.QSize(200, 25))
        self.ShowWorkingDirectory2.setObjectName("ShowWorkingDirectory2")
        self.horizontalLayout.addWidget(self.ShowWorkingDirectory2)
        self.SelectFolder2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.SelectFolder2.setObjectName("SelectFolder2")
        self.horizontalLayout.addWidget(self.SelectFolder2)
        self.ShowSelectedFolder2 = QtWidgets.QTextEdit(self.horizontalLayoutWidget)
        self.ShowSelectedFolder2.setMaximumSize(QtCore.QSize(16777215, 25))
        self.ShowSelectedFolder2.setObjectName("ShowSelectedFolder2")
        self.horizontalLayout.addWidget(self.ShowSelectedFolder2)
        self.layoutWidget = QtWidgets.QWidget(self.PrintingTraces)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 40, 591, 46))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.PlotFolder_2 = QtWidgets.QPushButton(self.layoutWidget)
        self.PlotFolder_2.setObjectName("PlotFolder_2")
        self.horizontalLayout_8.addWidget(self.PlotFolder_2)
        self.PlotSingle_2 = QtWidgets.QPushButton(self.layoutWidget)
        self.PlotSingle_2.setObjectName("PlotSingle_2")
        self.horizontalLayout_8.addWidget(self.PlotSingle_2)
        self.PlotNumber_2 = QtWidgets.QTextEdit(self.layoutWidget)
        self.PlotNumber_2.setMinimumSize(QtCore.QSize(0, 0))
        self.PlotNumber_2.setMaximumSize(QtCore.QSize(40, 25))
        self.PlotNumber_2.setObjectName("PlotNumber_2")
        self.horizontalLayout_8.addWidget(self.PlotNumber_2)
        self.Of_2 = QtWidgets.QLabel(self.layoutWidget)
        self.Of_2.setMaximumSize(QtCore.QSize(16, 20))
        self.Of_2.setObjectName("Of_2")
        self.horizontalLayout_8.addWidget(self.Of_2)
        self.NumAllTraces_2 = QtWidgets.QTextEdit(self.layoutWidget)
        self.NumAllTraces_2.setMaximumSize(QtCore.QSize(40, 25))
        self.NumAllTraces_2.setObjectName("NumAllTraces_2")
        self.horizontalLayout_8.addWidget(self.NumAllTraces_2)
        self.Traces_2 = QtWidgets.QLabel(self.layoutWidget)
        self.Traces_2.setMaximumSize(QtCore.QSize(44, 20))
        self.Traces_2.setObjectName("Traces_2")
        self.horizontalLayout_8.addWidget(self.Traces_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setContentsMargins(-1, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.PreviousPlot_2 = QtWidgets.QPushButton(self.layoutWidget)
        self.PreviousPlot_2.setMaximumSize(QtCore.QSize(120, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.PreviousPlot_2.setFont(font)
        self.PreviousPlot_2.setIconSize(QtCore.QSize(16, 16))
        self.PreviousPlot_2.setObjectName("PreviousPlot_2")
        self.verticalLayout_3.addWidget(self.PreviousPlot_2)
        self.NextPlot_2 = QtWidgets.QPushButton(self.layoutWidget)
        self.NextPlot_2.setMaximumSize(QtCore.QSize(120, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.NextPlot_2.setFont(font)
        self.NextPlot_2.setObjectName("NextPlot_2")
        self.verticalLayout_3.addWidget(self.NextPlot_2)
        self.horizontalLayout_8.addLayout(self.verticalLayout_3)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.PrintingTraces)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(0, 260, 694, 407))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_10.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout_10.setContentsMargins(0, 0, 103, 50)
        self.horizontalLayout_10.setSpacing(5)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout_7.setContentsMargins(-1, 0, -1, -1)
        self.verticalLayout_7.setSpacing(2)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.SetSavingPath = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.SetSavingPath.setMinimumSize(QtCore.QSize(150, 0))
        self.SetSavingPath.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.SetSavingPath.setObjectName("SetSavingPath")
        self.verticalLayout_7.addWidget(self.SetSavingPath)
        self.NameSavingPath = QtWidgets.QTextEdit(self.horizontalLayoutWidget_3)
        self.NameSavingPath.setMinimumSize(QtCore.QSize(150, 0))
        self.NameSavingPath.setMaximumSize(QtCore.QSize(125, 50))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.NameSavingPath.setFont(font)
        self.NameSavingPath.setObjectName("NameSavingPath")
        self.verticalLayout_7.addWidget(self.NameSavingPath)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_7.addWidget(self.pushButton)
        self.NameSavingFolder_2 = QtWidgets.QTextEdit(self.horizontalLayoutWidget_3)
        self.NameSavingFolder_2.setMinimumSize(QtCore.QSize(150, 0))
        self.NameSavingFolder_2.setMaximumSize(QtCore.QSize(125, 50))
        self.NameSavingFolder_2.setObjectName("NameSavingFolder_2")
        self.verticalLayout_7.addWidget(self.NameSavingFolder_2)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_7.addWidget(self.pushButton_2)
        self.AppendFilename = QtWidgets.QTextEdit(self.horizontalLayoutWidget_3)
        self.AppendFilename.setMinimumSize(QtCore.QSize(150, 0))
        self.AppendFilename.setMaximumSize(QtCore.QSize(125, 50))
        self.AppendFilename.setObjectName("AppendFilename")
        self.verticalLayout_7.addWidget(self.AppendFilename)
        spacerItem1 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout_7.addItem(spacerItem1)
        self.PrintButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.PrintButton.setMinimumSize(QtCore.QSize(150, 0))
        self.PrintButton.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.PrintButton.setAutoFillBackground(False)
        self.PrintButton.setCheckable(False)
        self.PrintButton.setObjectName("PrintButton")
        self.verticalLayout_7.addWidget(self.PrintButton)
        spacerItem2 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_7.addItem(spacerItem2)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSpacing(10)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.checkBox_3 = QtWidgets.QCheckBox(self.horizontalLayoutWidget_3)
        self.checkBox_3.setMaximumSize(QtCore.QSize(16777215, 15))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.checkBox_3.setFont(font)
        self.checkBox_3.setObjectName("checkBox_3")
        self.verticalLayout_4.addWidget(self.checkBox_3)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.ScaleMV = QtWidgets.QSpinBox(self.horizontalLayoutWidget_3)
        self.ScaleMV.setMaximumSize(QtCore.QSize(16777215, 15))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.ScaleMV.setFont(font)
        self.ScaleMV.setObjectName("ScaleMV")
        self.ScaleMV.setMinimum(0)
        self.ScaleMV.setMaximum(30000000)
        self.ScaleMV.setProperty("value", 10)
        self.horizontalLayout_12.addWidget(self.ScaleMV)
        self.label_17 = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_12.addWidget(self.label_17)
        self.verticalLayout_4.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.ScaleMs = QtWidgets.QSpinBox(self.horizontalLayoutWidget_3)
        self.ScaleMs.setMaximumSize(QtCore.QSize(16777215, 15))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.ScaleMs.setFont(font)
        self.ScaleMs.setObjectName("ScaleMs")
        self.ScaleMs.setMinimum(0)
        self.ScaleMs.setMaximum(30000000)
        self.ScaleMs.setProperty("value", 100)
        self.horizontalLayout_11.addWidget(self.ScaleMs)
        self.label_8 = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_11.addWidget(self.label_8)
        self.verticalLayout_4.addLayout(self.horizontalLayout_11)
        self.verticalLayout_7.addLayout(self.verticalLayout_4)
        self.horizontalLayout_10.addLayout(self.verticalLayout_7)
        self.graphicsViewPrinting = QtWidgets.QGraphicsView(self.horizontalLayoutWidget_3)
        self.graphicsViewPrinting.setEnabled(True)
        self.graphicsViewPrinting.setMinimumSize(QtCore.QSize(423, 345))
        self.graphicsViewPrinting.setMaximumSize(QtCore.QSize(396, 316))
        self.graphicsViewPrinting.setSizeIncrement(QtCore.QSize(0, 0))
        self.graphicsViewPrinting.setBaseSize(QtCore.QSize(0, 0))
        self.graphicsViewPrinting.setObjectName("graphicsViewPrinting")
        self.horizontalLayout_10.addWidget(self.graphicsViewPrinting)
        self.layoutWidget1 = QtWidgets.QWidget(self.PrintingTraces)
        self.layoutWidget1.setGeometry(QtCore.QRect(0, 90, 589, 169))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_5 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_5.setMaximumSize(QtCore.QSize(130, 16777215))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_4.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_6.setMaximumSize(QtCore.QSize(71, 16777215))
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_4.addWidget(self.label_6)
        self.TimeWindowFrom = QtWidgets.QSpinBox(self.layoutWidget1)
        self.TimeWindowFrom.setMaximumSize(QtCore.QSize(60, 16777215))
        self.TimeWindowFrom.setMinimum(-100000)
        self.TimeWindowFrom.setMaximum(30000000)
        self.TimeWindowFrom.setSingleStep(10)
        self.TimeWindowFrom.setObjectName("TimeWindowFrom")
        self.horizontalLayout_4.addWidget(self.TimeWindowFrom)
        self.label_7 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_7.setMaximumSize(QtCore.QSize(15, 16777215))
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_4.addWidget(self.label_7)
        self.TimeWindowTo = QtWidgets.QSpinBox(self.layoutWidget1)
        self.TimeWindowTo.setMaximumSize(QtCore.QSize(60, 16777215))
        self.TimeWindowTo.setMinimum(-10000)
        self.TimeWindowTo.setMaximum(10000000)
        self.TimeWindowTo.setSingleStep(10)
        self.TimeWindowTo.setProperty("value", 800)
        self.TimeWindowTo.setObjectName("TimeWindowTo")
        self.horizontalLayout_4.addWidget(self.TimeWindowTo)
        self.label_19 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_19.setMaximumSize(QtCore.QSize(84, 16777215))
        self.label_19.setObjectName("label_19")
        self.horizontalLayout_4.addWidget(self.label_19)
        self.TimeWindowDelta = QtWidgets.QTextEdit(self.layoutWidget1)
        self.TimeWindowDelta.setMaximumSize(QtCore.QSize(100, 25))
        self.TimeWindowDelta.setObjectName("TimeWindowDelta")
        self.horizontalLayout_4.addWidget(self.TimeWindowDelta)
        self.verticalLayout_8.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_9 = QtWidgets.QLabel(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setMaximumSize(QtCore.QSize(85, 16777215))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_9.addWidget(self.label_9)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setContentsMargins(0, -1, 0, 0)
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.checkBox_2 = QtWidgets.QCheckBox(self.layoutWidget1)
        self.checkBox_2.setMaximumSize(QtCore.QSize(20, 16777215))
        self.checkBox_2.setText("")
        self.checkBox_2.setObjectName("checkBox_2")
        self.horizontalLayout_7.addWidget(self.checkBox_2)
        spacerItem3 = QtWidgets.QSpacerItem(11, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem3)
        self.label_13 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_13.setMaximumSize(QtCore.QSize(71, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_7.addWidget(self.label_13)
        self.mVFrom = QtWidgets.QSpinBox(self.layoutWidget1)
        self.mVFrom.setMaximumSize(QtCore.QSize(60, 16777215))
        self.mVFrom.setMinimum(-100000000)
        self.mVFrom.setMaximum(1000000000)
        self.mVFrom.setSingleStep(10)
        self.mVFrom.setProperty("value", -70)
        self.mVFrom.setObjectName("mVFrom")
        self.horizontalLayout_7.addWidget(self.mVFrom)
        self.label_14 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_14.setMaximumSize(QtCore.QSize(15, 16777215))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_7.addWidget(self.label_14)
        self.mVTo = QtWidgets.QSpinBox(self.layoutWidget1)
        self.mVTo.setMaximumSize(QtCore.QSize(60, 16777215))
        self.mVTo.setMinimum(-1000000000)
        self.mVTo.setMaximum(1000000000)
        self.mVTo.setSingleStep(10)
        self.mVTo.setProperty("value", 50)
        self.mVTo.setObjectName("mVTo")
        self.horizontalLayout_7.addWidget(self.mVTo)
        self.label_15 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_15.setMaximumSize(QtCore.QSize(84, 16777215))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_7.addWidget(self.label_15)
        self.mVDelta = QtWidgets.QTextEdit(self.layoutWidget1)
        self.mVDelta.setMaximumSize(QtCore.QSize(100, 25))
        self.mVDelta.setObjectName("mVDelta")
        self.horizontalLayout_7.addWidget(self.mVDelta)
        self.verticalLayout_6.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setContentsMargins(0, -1, 0, 0)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.checkBox = QtWidgets.QCheckBox(self.layoutWidget1)
        self.checkBox.setMaximumSize(QtCore.QSize(20, 16777215))
        self.checkBox.setText("")
        self.checkBox.setChecked(True)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_6.addWidget(self.checkBox)
        spacerItem4 = QtWidgets.QSpacerItem(12, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem4)
        self.label_10 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_10.setMaximumSize(QtCore.QSize(56, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_6.addWidget(self.label_10)
        self.label_20 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_20.setMaximumSize(QtCore.QSize(11, 16777215))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.horizontalLayout_6.addWidget(self.label_20)
        self.BaselineMinus = QtWidgets.QSpinBox(self.layoutWidget1)
        self.BaselineMinus.setMaximumSize(QtCore.QSize(60, 16777215))
        self.BaselineMinus.setMinimum(-1000000000)
        self.BaselineMinus.setMaximum(999999999)
        self.BaselineMinus.setSingleStep(10)
        self.BaselineMinus.setProperty("value", -10)
        self.BaselineMinus.setObjectName("BaselineMinus")
        self.horizontalLayout_6.addWidget(self.BaselineMinus)
        self.label_11 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_11.setMaximumSize(QtCore.QSize(15, 16777215))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_6.addWidget(self.label_11)
        self.BaselinePlus = QtWidgets.QSpinBox(self.layoutWidget1)
        self.BaselinePlus.setMaximumSize(QtCore.QSize(60, 16777215))
        self.BaselinePlus.setMinimum(-1000000000)
        self.BaselinePlus.setMaximum(1000000000)
        self.BaselinePlus.setSingleStep(10)
        self.BaselinePlus.setProperty("value", 110)
        self.BaselinePlus.setObjectName("BaselinePlus")
        self.horizontalLayout_6.addWidget(self.BaselinePlus)
        self.label_12 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_12.setMaximumSize(QtCore.QSize(89, 16777215))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_6.addWidget(self.label_12)
        self.textEdit = QtWidgets.QTextEdit(self.layoutWidget1)
        self.textEdit.setMaximumSize(QtCore.QSize(100, 25))
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout_6.addWidget(self.textEdit)
        self.verticalLayout_6.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_9.addLayout(self.verticalLayout_6)
        self.verticalLayout_8.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label = QtWidgets.QLabel(self.layoutWidget1)
        self.label.setMaximumSize(QtCore.QSize(130, 16777215))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_5.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_2.setMaximumSize(QtCore.QSize(71, 16777215))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_5.addWidget(self.label_2)
        self.CalcBaselineFrom = QtWidgets.QSpinBox(self.layoutWidget1)
        self.CalcBaselineFrom.setMaximumSize(QtCore.QSize(60, 16777215))
        self.CalcBaselineFrom.setMinimum(-1000)
        self.CalcBaselineFrom.setMaximum(300000)
        self.CalcBaselineFrom.setObjectName("CalcBaselineFrom")
        self.horizontalLayout_5.addWidget(self.CalcBaselineFrom)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_3.setMaximumSize(QtCore.QSize(15, 16777215))
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.CalcBaselineTo = QtWidgets.QSpinBox(self.layoutWidget1)
        self.CalcBaselineTo.setMaximumSize(QtCore.QSize(60, 16777215))
        self.CalcBaselineTo.setMinimum(-1000)
        self.CalcBaselineTo.setMaximum(1000000000)
        self.CalcBaselineTo.setProperty("value", 100)
        self.CalcBaselineTo.setObjectName("CalcBaselineTo")
        self.horizontalLayout_5.addWidget(self.CalcBaselineTo)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_4.setMaximumSize(QtCore.QSize(84, 16777215))
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        self.CalcBaselineDelta = QtWidgets.QTextEdit(self.layoutWidget1)
        self.CalcBaselineDelta.setMaximumSize(QtCore.QSize(100, 25))
        self.CalcBaselineDelta.setObjectName("CalcBaselineDelta")
        self.horizontalLayout_5.addWidget(self.CalcBaselineDelta)
        self.verticalLayout_8.addLayout(self.horizontalLayout_5)
        self.ResizeActualFigure = QtWidgets.QPushButton(self.layoutWidget1)
        self.ResizeActualFigure.setMinimumSize(QtCore.QSize(150, 0))
        self.ResizeActualFigure.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.ResizeActualFigure.setObjectName("ResizeActualFigure")
        self.verticalLayout_8.addWidget(self.ResizeActualFigure)
        self.ShowingTabe.addTab(self.PrintingTraces, "")
        self.verticalLayout.addWidget(self.ShowingTabe)
        MainWindow.setCentralWidget(self.verticalFrame_2)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 621, 22))
        self.menubar.setObjectName("menubar")
        self.menuDennis_Analysis = QtWidgets.QMenu(self.menubar)
        self.menuDennis_Analysis.setObjectName("menuDennis_Analysis")
        MainWindow.setMenuBar(self.menubar)
        self.menubar.addAction(self.menuDennis_Analysis.menuAction())

        self.retranslateUi(MainWindow)
        self.ShowingTabe.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ShowingTabe.setToolTip(_translate("MainWindow", "<html><head/><body><p>Showing Traces</p></body></html>"))
        self.SetWorkingDirectory.setText(_translate("MainWindow", "Select Cell:"))
        self.SelectFolder.setText(_translate("MainWindow", "Select Folder:"))
        self.PlotFolder.setText(_translate("MainWindow", "Show Folder"))
        self.PlotSingle.setText(_translate("MainWindow", "Show Single Traces:"))
        self.Of.setText(_translate("MainWindow", "of"))
        self.Traces.setText(_translate("MainWindow", "Traces"))
        self.PreviousPlot.setText(_translate("MainWindow", "Previous Trace"))
        self.NextPlot.setText(_translate("MainWindow", "Next Trace"))
        self.ShowingTabe.setTabText(self.ShowingTabe.indexOf(self.ShowingTraces), _translate("MainWindow", "Show Traces"))
        self.SetWorkingDirectory2.setText(_translate("MainWindow", "Select Cell:"))
        self.SelectFolder2.setText(_translate("MainWindow", "Select Folder:"))
        self.PlotFolder_2.setText(_translate("MainWindow", "Show Folder"))
        self.PlotSingle_2.setText(_translate("MainWindow", "Show Single Traces:"))
        self.Of_2.setText(_translate("MainWindow", "of"))
        self.Traces_2.setText(_translate("MainWindow", "Traces"))
        self.PreviousPlot_2.setText(_translate("MainWindow", "Previous Trace"))
        self.NextPlot_2.setText(_translate("MainWindow", "Next Trace"))
        self.SetSavingPath.setText(_translate("MainWindow", "Saving Path:"))
        self.pushButton.setText(_translate("MainWindow", "Saving Folder:"))
        self.NameSavingFolder_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Examples</p></body></html>"))
        self.pushButton_2.setText(_translate("MainWindow", "Append Filename:"))
        self.AppendFilename.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.PrintButton.setText(_translate("MainWindow", "Print"))
        self.checkBox_3.setText(_translate("MainWindow", "ScaleBar"))
        self.label_17.setText(_translate("MainWindow", "mV"))
        self.label_8.setText(_translate("MainWindow", "ms"))
        self.label_5.setText(_translate("MainWindow", "Set Time Window:"))
        self.label_6.setText(_translate("MainWindow", "From:"))
        self.label_7.setText(_translate("MainWindow", "to"))
        self.label_19.setText(_translate("MainWindow", "ms with Δ of:"))
        self.TimeWindowDelta.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>"))
        self.label_9.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Set Mili-Volt </p><p align=\"center\">Range:</p></body></html>"))
        self.label_13.setText(_translate("MainWindow", "From: "))
        self.label_14.setText(_translate("MainWindow", "to"))
        self.label_15.setText(_translate("MainWindow", "mV with Δ of:"))
        self.mVDelta.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>"))
        self.label_10.setText(_translate("MainWindow", "Baseline:"))
        self.label_20.setText(_translate("MainWindow", "-"))
        self.label_11.setText(_translate("MainWindow", "/ +"))
        self.label_12.setText(_translate("MainWindow", "mV with Δ of:"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>"))
        self.label.setText(_translate("MainWindow", "Calculate Baseline:"))
        self.label_2.setText(_translate("MainWindow", "From:"))
        self.label_3.setText(_translate("MainWindow", "to"))
        self.label_4.setText(_translate("MainWindow", "ms with Δ of:"))
        self.CalcBaselineDelta.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.Helvetica Neue DeskInterface\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>"))
        self.ResizeActualFigure.setText(_translate("MainWindow", "Resize Actual Image"))
        self.ShowingTabe.setTabText(self.ShowingTabe.indexOf(self.PrintingTraces), _translate("MainWindow", "Print Traces"))
        self.menuDennis_Analysis.setTitle(_translate("MainWindow", "Plotting"))

        ''' Own Stuff: '''
        # Picking Stuff: 
        def pick_new():
            dialog = QFileDialog()
            folder_path = dialog.getExistingDirectory(None, "Select Folder")
            CellMatrix = folder_path.split("/")
            Cell = CellMatrix[-1]
            self.ShowWorkingDirectory.setText(Cell)
            self.ShowWorkingDirectory2.setText(Cell)
            os.chdir(folder_path)
            return folder_path
        self.SetWorkingDirectory.clicked.connect(pick_new)
        self.SetWorkingDirectory2.clicked.connect(pick_new)
        
        # Set and Show Single Cell: 
        def pick_Cell():
            dialog = QFileDialog()
            Singlefolder_path = dialog.getExistingDirectory(None, "Select Folder")
            os.chdir(Singlefolder_path)
            a = Singlefolder_path.split("/")
            Folder = a[-1]
            self.ShowSelectedFolder.setText(a[-1])
            self.ShowSelectedFolder2.setText(a[-1])
            return Singlefolder_path, Folder
        self.SelectFolder.clicked.connect(pick_Cell)
        self.SelectFolder2.clicked.connect(pick_Cell)
        
        def pick_SavingPath():
            dialog = QFileDialog()
            folder_path = dialog.getExistingDirectory(None, "Select Folder")
            self.NameSavingPath.setText(folder_path)
        self.SetSavingPath.clicked.connect(pick_SavingPath)
        
        def pick_SavingFolder():
            dialog = QFileDialog()
            Singlefolder_path = dialog.getExistingDirectory(None, "Select Folder")
            a = Singlefolder_path.split("/")
            Folder = a[-1]
            self.NameSavingFolder_2.setText(Folder)
        self.pushButton.clicked.connect(pick_SavingFolder)
        
        def importStuffandPrint():
            Folder = self.ShowSelectedFolder.toPlainText() 
            A = ImportFolder(Folder)
            dr = PlotFolderClass(A.TimeVec,A.Waves,A.files) 
            graphicscene = QtWidgets.QGraphicsScene() 
            graphicscene.addWidget(dr)  
            self.PlottingWindow.setScene(graphicscene)
            img_aspect_ratio =  float(dr.size().width()) / dr.size().height() 
            widthImageViewer = self.PlottingWindow.size().width()
            self.PlottingWindow.setFixedHeight( widthImageViewer / img_aspect_ratio )
            self.PlottingWindow.fitInView(self.PlottingWindow.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.PlottingWindow.show()
        self.PlotFolder.clicked.connect(importStuffandPrint)
        
        def importStuffandPrintSingle():
            x = 0
            Folder = self.ShowSelectedFolder.toPlainText() 
            A = ImportFolder(Folder)
            dr = PlotSingleClass(A.TimeVec[x],A.Waves[x],A.files[x]) 
            graphicscene = QtWidgets.QGraphicsScene() 
            graphicscene.addWidget(dr)  
            self.PlottingWindow.setScene(graphicscene)
            img_aspect_ratio =  float(dr.size().width()) / dr.size().height() 
            widthImageViewer = self.PlottingWindow.size().width()
            self.PlottingWindow.setFixedHeight( widthImageViewer / img_aspect_ratio )
            self.PlottingWindow.fitInView(self.PlottingWindow.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.PlottingWindow.show()
            XToShow = x+1
            self.PlotNumber.setText(str(XToShow))
            AllTraces = len(A.TimeVec)
            self.NumAllTraces.setText(str(AllTraces))
        self.PlotSingle.clicked.connect(importStuffandPrintSingle)
        
        def NextStuff():
            Folder = self.ShowSelectedFolder.toPlainText() 
            A = ImportFolder(Folder)
            TraceNumber = self.PlotNumber.toPlainText()
            AllTraces = len(A.TimeVec)
            x1 = np.fromstring(TraceNumber, dtype=int, sep=", ")
            x = int(x1[0])
            if x > AllTraces-1:
                x = 0
            dr = PlotSingleClass(A.TimeVec[x],A.Waves[x],A.files[x]) 
            graphicscene = QtWidgets.QGraphicsScene() 
            graphicscene.addWidget(dr)  
            self.PlottingWindow.setScene(graphicscene)
            img_aspect_ratio =  float(dr.size().width()) / dr.size().height() 
            widthImageViewer = self.PlottingWindow.size().width()
            self.PlottingWindow.setFixedHeight( widthImageViewer / img_aspect_ratio )
            self.PlottingWindow.fitInView(self.PlottingWindow.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.PlottingWindow.show()
            XToShow = x+1
            self.PlotNumber.setText(str(XToShow))
            self.NumAllTraces.setText(str(AllTraces))
        self.NextPlot.clicked.connect(NextStuff)
        
        def PreviousStuff():
            Folder = self.ShowSelectedFolder.toPlainText() 
            A = ImportFolder(Folder)
            TraceNumber = self.PlotNumber.toPlainText()
            AllTraces = len(A.TimeVec)
            x1 = np.fromstring(TraceNumber, dtype=int, sep=", ")
            x = int(x1[0])-2
            if x < 0:
                x = AllTraces - np.absolute(x)
            dr = PlotSingleClass(A.TimeVec[x],A.Waves[x],A.files[x]) 
            graphicscene = QtWidgets.QGraphicsScene() 
            graphicscene.addWidget(dr)  
            self.PlottingWindow.setScene(graphicscene)
            img_aspect_ratio =  float(dr.size().width()) / dr.size().height() 
            widthImageViewer = self.PlottingWindow.size().width()
            self.PlottingWindow.setFixedHeight( widthImageViewer / img_aspect_ratio )
            self.PlottingWindow.fitInView(self.PlottingWindow.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.PlottingWindow.show()
            XToShow = x+1
            self.PlotNumber.setText(str(XToShow))
            self.NumAllTraces.setText(str(AllTraces))
        self.PreviousPlot.clicked.connect(PreviousStuff)

        ''' Printing Suff '''        
        def PreToPrinting_Folder():
            Folder = self.ShowSelectedFolder2.toPlainText() 
            A = ImportFolder(Folder)
            dr = PlotFolderClass(A.TimeVec,A.Waves,A.files) 
            graphicscene = QtWidgets.QGraphicsScene() 
            graphicscene.addWidget(dr)  
            self.graphicsViewPrinting.setScene(graphicscene)
            img_aspect_ratio =  float(dr.size().width()) / dr.size().height() 
            widthImageViewer = self.graphicsViewPrinting.size().width()
            self.graphicsViewPrinting.setFixedHeight( widthImageViewer / img_aspect_ratio )
            self.graphicsViewPrinting.fitInView(self.graphicsViewPrinting.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.graphicsViewPrinting.show()
#            self.toolbar = NavigationToolbar(dr, self.graphicsViewPrinting)
        self.PlotFolder_2.clicked.connect(PreToPrinting_Folder)
        
        
        def PlotCustomised():
            x = 0
            Folder = self.ShowSelectedFolder2.toPlainText() 
            A = ImportFolder(Folder)
            # ScaleBar: 
            if self.checkBox_3.isChecked():
                ScaleBarMs = self.ScaleMs.value()*A.SampFreq[x]
                ScaleBarVm = self.ScaleMV.value()
            else:
                ScaleBarMs = np.nan
                ScaleBarVm = np.nan   
            # Plotting Boarders:    
            PlottingBoarders = CustumisedPlots(self,A.Waves[x],A.SampFreq[x])
            # Plotting:
            dr = PlotSingleClassWithBoarders(A.TimeVec[x],A.Waves[x],A.SampFreq[x],A.files[x],PlottingBoarders,0,'NAN',ScaleBarMs,ScaleBarVm) 
            # To Graphic-View with Ajdustment:
            graphicscene2 = QtWidgets.QGraphicsScene() 
            graphicscene2.addWidget(dr)  
            self.graphicsViewPrinting.setScene(graphicscene2)
            img_aspect_ratio =  float(dr.size().width()) / dr.size().height() 
            widthImageViewer = self.graphicsViewPrinting.size().width()
            self.graphicsViewPrinting.setFixedHeight( widthImageViewer / img_aspect_ratio )
            self.graphicsViewPrinting.fitInView(self.graphicsViewPrinting.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.graphicsViewPrinting.show() 
            # PlottingNumber: n
            XToShow = x+1
            self.PlotNumber_2.setText(str(XToShow))
            AllTraces = len(A.TimeVec)
            self.NumAllTraces_2.setText(str(AllTraces)) 
        self.PlotSingle_2.clicked.connect(PlotCustomised)
        
        def NextStuff_Print():
            Folder = self.ShowSelectedFolder2.toPlainText() 
            A = ImportFolder(Folder)
            # Getting Trace Number
            TraceNumber = self.PlotNumber_2.toPlainText()
            AllTraces = len(A.TimeVec)
            x1 = np.fromstring(TraceNumber, dtype=int, sep=", ")
            x = int(x1[0])
            if x > AllTraces-1:
                x = 0
            # Getting Boarders
            PlottingBoarders = CustumisedPlots(self,A.Waves[x],A.SampFreq[x])
            # ScaleBar: 
            if self.checkBox_3.isChecked():
                ScaleBarMs = self.ScaleMs.value()*A.SampFreq[x]
                ScaleBarVm = self.ScaleMV.value()
            else:
                ScaleBarMs = np.nan
                ScaleBarVm = np.nan 
            # Plotting
            dr = PlotSingleClassWithBoarders(A.TimeVec[x],A.Waves[x],A.SampFreq[x],A.files[x],PlottingBoarders,0,'NAN',ScaleBarMs,ScaleBarVm) 
            # Set Graphic View:
            graphicscene2 = QtWidgets.QGraphicsScene() 
            graphicscene2.addWidget(dr)  
            self.graphicsViewPrinting.setScene(graphicscene2)
            img_aspect_ratio =  float(dr.size().width()) / dr.size().height() 
            widthImageViewer = self.graphicsViewPrinting.size().width()
            self.graphicsViewPrinting.setFixedHeight( widthImageViewer / img_aspect_ratio )
            self.graphicsViewPrinting.fitInView(self.graphicsViewPrinting.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.graphicsViewPrinting.show()  
            # Set New TraceNumber:
            XToShow = x+1
            self.PlotNumber_2.setText(str(XToShow))
            AllTraces = len(A.TimeVec)
            self.NumAllTraces_2.setText(str(AllTraces)) 
        self.NextPlot_2.clicked.connect(NextStuff_Print)
        
        def PreviousStuff_Print():
            Folder = self.ShowSelectedFolder2.toPlainText() 
            A = ImportFolder(Folder)
            # Getting Trace Number:
            TraceNumber = self.PlotNumber_2.toPlainText()
            AllTraces = len(A.TimeVec)
            x1 = np.fromstring(TraceNumber, dtype=int, sep=", ")
            x = int(x1[0])-2
            if x < 0:
                x = AllTraces - np.absolute(x)
            # Getting Boarders:
            PlottingBoarders = CustumisedPlots(self,A.Waves[x],A.SampFreq[x])
            # ScaleBar: 
            if self.checkBox_3.isChecked():
                ScaleBarMs = self.ScaleMs.value()*A.SampFreq[x]
                ScaleBarVm = self.ScaleMV.value()
            else:
                ScaleBarMs = np.nan
                ScaleBarVm = np.nan 
            # Plotting
            dr = PlotSingleClassWithBoarders(A.TimeVec[x],A.Waves[x],A.SampFreq[x],A.files[x],PlottingBoarders,0,'NAN',ScaleBarMs,ScaleBarVm) 
            # Setting Graphic-View:
            graphicscene2 = QtWidgets.QGraphicsScene() 
            graphicscene2.addWidget(dr)  
            self.graphicsViewPrinting.setScene(graphicscene2)
            img_aspect_ratio =  float(dr.size().width()) / dr.size().height() 
            widthImageViewer = self.graphicsViewPrinting.size().width()
            self.graphicsViewPrinting.setFixedHeight( widthImageViewer / img_aspect_ratio )
            self.graphicsViewPrinting.fitInView(self.graphicsViewPrinting.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.graphicsViewPrinting.show()  
            # Set New TraceNumber:
            XToShow = x+1
            self.PlotNumber_2.setText(str(XToShow))
            AllTraces = len(A.TimeVec)
            self.NumAllTraces_2.setText(str(AllTraces))
        self.PreviousPlot_2.clicked.connect(PreviousStuff_Print)
        
        def ResizeActualFigure_Print():
            Folder = self.ShowSelectedFolder2.toPlainText() 
            A = ImportFolder(Folder)
            # Getting TraceNumber:
            TraceNumber = self.PlotNumber_2.toPlainText()
            AllTraces = len(A.TimeVec)
            x1 = np.fromstring(TraceNumber, dtype=int, sep=", ")
            x = int(x1[0])-1
            # Getting Boarders:
            PlottingBoarders = CustumisedPlots(self,A.Waves[x],A.SampFreq[x])
            # ScaleBar: 
            if self.checkBox_3.isChecked():
                ScaleBarMs = self.ScaleMs.value()*A.SampFreq[x]
                ScaleBarVm = self.ScaleMV.value()
            else:
                ScaleBarMs = np.nan
                ScaleBarVm = np.nan 
            # Plotting
            dr = PlotSingleClassWithBoarders(A.TimeVec[x],A.Waves[x],A.SampFreq[x],A.files[x],PlottingBoarders,0,'NAN',ScaleBarMs,ScaleBarVm) 
            # Setting Graphic View:
            graphicscene2 = QtWidgets.QGraphicsScene() 
            graphicscene2.addWidget(dr)  
            self.graphicsViewPrinting.setScene(graphicscene2)
            img_aspect_ratio =  float(dr.size().width()) / dr.size().height() 
            widthImageViewer = self.graphicsViewPrinting.size().width()
            self.graphicsViewPrinting.setFixedHeight( widthImageViewer / img_aspect_ratio )
            self.graphicsViewPrinting.fitInView(self.graphicsViewPrinting.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.graphicsViewPrinting.show()  
            # Setting TraceNumber:
            XToShow = x+1
            self.PlotNumber_2.setText(str(XToShow))
            AllTraces = len(A.TimeVec)
            self.NumAllTraces_2.setText(str(AllTraces))
        self.ResizeActualFigure.clicked.connect(ResizeActualFigure_Print)
        
        def Printing():
            # FolderStuff and Import Traces:
            OriginalPath = os.getcwd()
            CellName = self.ShowWorkingDirectory2.toPlainText() 
            Folder = self.ShowSelectedFolder2.toPlainText() 
            A = ImportFolder(Folder)
            TraceNumber = self.PlotNumber_2.toPlainText()
            AllTraces = len(A.TimeVec)
            x1 = np.fromstring(TraceNumber, dtype=int, sep=", ")
            x = int(x1[0])-1
            
            # Plotting Boarders:
            PlottingBoarders = CustumisedPlots(self,A.Waves[x],A.SampFreq[x])
            # ScaleBar: 
            if self.checkBox_3.isChecked():
                ScaleBarMs = self.ScaleMs.value()*A.SampFreq[x]
                ScaleBarVm = self.ScaleMV.value()
            else:
                ScaleBarMs = np.nan
                ScaleBarVm = np.nan 
                
            # Go To Specificed Folder:
            NewPath = self.NameSavingPath.toPlainText()
#            print(NewPath)
            os.chdir(NewPath)
            NewFolder = self.NameSavingFolder_2.toPlainText()
            if NewFolder is not None:
                if not os.path.isdir(os.getcwd()+'/'+ NewFolder):
                    os.makedirs(os.getcwd()+'/'+ NewFolder)        
                os.chdir(os.getcwd()+'/'+ NewFolder)
            
            # Printing:
            CellNameAppend = self.AppendFilename.toPlainText()
            CellNamePrint = CellName + '_'+ A.files[x] + '_' + CellNameAppend # FileName! 
            dr = PlotSingleClassWithBoarders(A.TimeVec[x],A.Waves[x],A.SampFreq[x],A.files[x],PlottingBoarders,1,CellNamePrint,ScaleBarMs,ScaleBarVm) 
            # Going to Original Folder:
            os.chdir(OriginalPath)
            # Setting Graphic View:
            graphicscene2 = QtWidgets.QGraphicsScene() 
            graphicscene2.addWidget(dr)  
            self.graphicsViewPrinting.setScene(graphicscene2)
            img_aspect_ratio =  float(dr.size().width()) / dr.size().height() 
            widthImageViewer = self.graphicsViewPrinting.size().width()
            self.graphicsViewPrinting.setFixedHeight( widthImageViewer / img_aspect_ratio )
            self.graphicsViewPrinting.fitInView(self.graphicsViewPrinting.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.graphicsViewPrinting.show()  
            # Setting TraceNumber:
            XToShow = x+1
            self.PlotNumber_2.setText(str(XToShow))
            AllTraces = len(A.TimeVec)
            self.NumAllTraces_2.setText(str(AllTraces))
        self.PrintButton.clicked.connect(Printing)    
                
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

