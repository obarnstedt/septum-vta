#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:44:38 2017
@author: DennisDa

Simple, Two and Multiple Group Testing
> Inclusive Testing of Dependencies of different Tests (Normlaity, Equal Variances, Skewness)
> Includes Figures to show distributions/ Tests,.... (0 = Nothing, 1 = Print JPG, 2 = Print SVG, 3 = Show)

> Input For Classes: 
    Values(as whatever), *kavas, ParameterName, GroupNames, PrintShow)
    
> Table of Content:
    0) Helping Definitions and Printing Definitions and TextBox Definitions
    A) Convert everything to Numpy Array
    B) Single Distributions: Normality, Skewness
    C) Compare Distributions: Variances
    D) Compare Distributions: Significance Test
    E) Compare Distributions: Correlation/Regression
    F) Compare MultiGroups: Significance Test
    
> Classes to Call: 
    DistributionStatSingle (X, Parametername='TBA',PrintShow)
    DistributionStatGroups (X, PramaeterGroup,PrintShow)
    Variance2Groups (self, X, Y, Xnorm, Xskew, Ynorm, Yskew, ParameterName,GroupNames, PrintShow)
    Test2ContinGroups (X, Y,YisGroupVec,Sided, Paired, EqualVariance, Xnorm, Xskew, Ynorm, Yskew, ParameterName,GroupNames, PrintShow)
    LinearCorrelation (X,Y,YisGroupVec,EqualVariance, Xnorm, Xskew, Ynorm, Yskew,ParameterNames, PrintShow):
    TestContinMultiGroup (...,)
    
> To Do:    > Repeated Measurement ANOVA and Friedmann's Test, 
            > Everything for Binary Data:    
    
"""

''' A: Import Scripts: '''
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng
from itertools import combinations
import dunn 

#from lmfit.models import SkewedGaussianModel
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import ddPlotting



''' Helping Definitions:'''
def Convert(X):
    # 1) Convert from pd.DataFrames:
    if type(X) == pd.core.frame.DataFrame:
        X = pd.DataFrame.as_matrix(X)  
    if type(X) == pd.core.series.Series:
        X = pd.DataFrame.as_matrix(X) 
    # 2) Convert from Row To Column    
    a = X.size
    if a == 1:
        A = np.zeros(shape=[a,1])
        A[:,0] = X[0,:]
    else:
        A = np.asarray(X, dtype=np.float64)
    return A

# From GroupingVector to two or more Tables:
def GroupVecToDist(X,GroupVec):
    Groups = np.unique(GroupVec)   
    ArrayIdx = [None]*len(Groups)
    Array = [None]*len(Groups)
    i = 0
    while i < len(Groups):
        ArrayIdx[i] = np.argwhere(GroupVec[0] == Groups[i])
        Array[i] = np.ndarray.flatten(X[ArrayIdx[i]])
        i += 1
    if len(Array) == 2:
        X = np.zeros(shape=(len(Array[0]),1))
        Y = np.zeros(shape=(len(Array[1]),1))
        X = Array[0]
        Y = Array[1]
    else:
        X = np.nan
        Y = np.nan
    return X, Y, Array

''' B: Single Distribution Properties: '''
class DistPropertiesCalc:
    def __init__(self, Data):
        self.NumValues = len(Data)
        self.mean = np.mean(Data)
        self.std = np.std(Data)
        self.median = np.median(Data)
        self.perc95 = np.percentile(Data,95)
        self.perc75 = np.percentile(Data,75)
        self.perc25 = np.percentile(Data,25)
        self.perc5 = np.percentile(Data,5)

def NormalityCalc(X,PrintShow):
    # Normality Test based on SampleSize:
    A = X
    if A.size < 20:
        # Wilk-Saphiro test for small sample size > 8ung: besser direkt nicht parametrisch! 
        Result = stats.shapiro(X)
        pValue = Result[1]     
    else:
        # D’Agostino and Pearson’s Test
        Result = stats.normaltest(X,nan_policy='propagate')
        pValue = Result[1]
    # TestResults    
    if pValue < 0.05:
        Normality = 0
    else:
        Normality = 1 
    return Normality, pValue

def SkewnessCalc(X, PrintShow):
    A = X
    if A.size > 8:
        ResultSkew = stats.skew(X)
        ResultsSkewTest = stats.skewtest(X)
        pValue = ResultsSkewTest[1] 
        if pValue > 0.05:
            Skweness = 0
        else:
            Skweness = 1     
    else:
        Skweness = 0
        pValue = 0.01
        ResultSkew = np.nan
    return Skweness, pValue, ResultSkew

''' Plotting Definitions '''
# Colors:
ColorsGoup =[[250/255,169/255,0/255],\
             [6/255,162/255,203/255],\
             [33/255,133/255,89/255],\
             [91/255,37/255,131/255],\
             [234/255,207/255,0/255],\
             [139/255,92/255,159/255]]

def PlotNormDist(X,Binsize): 
    if Binsize == 0:
        Binsize ='auto' # Based on Autobins of matplotlib
    hmean = np.mean(X)
    hstd = np.std(X)
    Figure1 = plt.figure()
    Count,bins,_ = plt.hist(X,bins=Binsize,normed=0)
    plt.close(Figure1)
    binwidth = (bins[1]-bins[0])
    # NormalDist
    X.sort()
    NormPdf = stats.norm.pdf(X, hmean, hstd)
    NormPdf = NormPdf * (binwidth*len(X))
    XValues = X
    YValues=NormPdf
    return XValues,YValues

def PlotNormSkwedDist(X,bins,binwidth): 
    hmean = np.mean(X)
    hstd = np.std(X)
    Figure1 = plt.figure()
    Count,bins,_ = plt.hist(X,bins='auto',normed=0)
    plt.close(Figure1)
    # NormalDist:
    X.sort()
    NormPdf = stats.norm.pdf(X, hmean, hstd)
    NormPdf = NormPdf * (binwidth*len(X))
    # Skewed Dist:
    lnspc = np.linspace(np.min(bins), np.max(bins), len(X))
    ag,bg,cg = stats.gamma.fit(X)  
    pdf_gamma = stats.gamma.pdf(lnspc, ag, bg,cg)  
    pdf_gamma = pdf_gamma * (binwidth*len(X))
    XValues = lnspc
    YValues=pdf_gamma
    if np.max(pdf_gamma) > 4*np.max(Count): 
        XValues = np.nan
        YValues = np.nan
    return XValues,YValues

def HistGausPlot(X,Y,Color1,Color2):
    both = np.concatenate((X,Y),axis=0)
    bins = np.linspace(np.min(both),np.max(both),int(len(both)/2))
    ax = plt.hist(X,bins=bins,normed=0,label='Population1',histtype = 'bar',edgecolor='k',fc=Color1, alpha = 0.25, linewidth=0.5)
    ax = plt.hist(Y,bins=bins,normed=0,label='Population2',histtype = 'bar',edgecolor='k',fc=Color2, alpha = 0.25,linewidth=0.5)
    binssize = np.absolute(bins[1]-bins[0])
#    print(binssize)
    X1SkewDistPop, Y1SkewDistPop = PlotNormSkwedDist(X,bins,binssize)
    ax = plt.plot(X1SkewDistPop,Y1SkewDistPop,color=Color1,linewidth=2)
    X2SkewDistPop, Y2SkewDistPop = PlotNormSkwedDist(Y,bins,binssize)
    ax = plt.plot(X2SkewDistPop,Y2SkewDistPop,color=Color2,linewidth=2)
    return ax         

def SwarmBoxPlot (X,Color):
    ax = sns.swarmplot(data=X,palette=sns.color_palette(Color),zorder=0.5)#,size=2)
    ax = sns.boxplot(data=X,palette=sns.color_palette([[125/255,125/255,125/255],[125/255,125/255,125/255]]),saturation=0.8)
    for patch in ax.artists:
         r, g, b, a = patch.get_facecolor()
         patch.set_facecolor((r, g, b, .4))         
    return ax

''' TextBox Definitions: '''
class TextBoxPlot:
    def __init__(self,DistProperties,Norm=-1,Skew=-1):
        InfoArray = [None]*5
        self.Norm = Norm
        self.Skew = Skew
#        print(self.Norm,self.Skew)        
        if self.Norm > -1 and self.Skew > -1: 
            if self.Skew == 1 and self.Norm == 1:
                self.Distribution = 'Skewed Normal Distribution'
            elif self.Skew == 0 and self.Norm == 1: 
                self.Distribution = 'Non-Skewed Normal Distribution'
            elif self.Skew == 1 and self.Norm == 0: 
                self.Distribution = 'Skewed Non-Normal Distribution'
            else:
                self.Distribution = 'Non-Normal Distribution'
            InfoArray[0] = self.Distribution +'\n'
        else:
            InfoArray[0] = '? Distribution'+'\n'
            
        InfoArray[1] = '%.f Values' %DistProperties.NumValues +'\n'
        InfoArray[2] = 'Mean: %.2f'%DistProperties.mean + ' +/- %.2f'%DistProperties.std + '\n'
        InfoArray[3] = 'Median: %.2f' % DistProperties.median+'\n'
        InfoArray[4] = 'Percentiles: 95th : %.2f; ' % DistProperties.perc95 + '75th : %.2f; ' % DistProperties.perc75+ '25th : %.2f; ' % DistProperties.perc25 + '5th : %.2f' % DistProperties.perc5  + '\n'
        self.InfoText = ''.join(InfoArray)

''' B: Single Distributions: '''
class DistributionStatSingle:
    def __init__(self, X, Parametername='TBA',PrintShow = 1):
        self.Data = X
        self.PrintShow = PrintShow
        self.Title = Parametername
        
        # Converte Data if nessesary:
        self.Data = Convert(self.Data) 
        # Get MainValues:
        self.NumValues = len(self.Data)
        self.mean = np.nanmean(self.Data)
        self.std = np.nanstd(self.Data)
        self.median = np.nanmedian(self.Data)
        self.perc95 = np.nanpercentile(self.Data,95)
        self.perc75 = np.nanpercentile(self.Data,75)
        self.perc25 = np.nanpercentile(self.Data,25)
        self.perc5 = np.nanpercentile(self.Data,5)   
        
        # Calculate Normality and Skewness:
        self.Norm, self.NormPValue = NormalityCalc(self.Data,0)
        self.Skew, self.SkewPValue, self.SkewValue = SkewnessCalc(self.Data,0)
        
        # Plotting
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
            # Figure:            
            self.Figure = plt.figure()
            self.Figure.set_dpi(200)
            self.gs = gridspec.GridSpec(6,2)
            self.gs.update(left=0.1, bottom= 0.075, top = 0.9, right=0.95,wspace=0.35)            
            #
            self.Figure.suptitle(self.Title,fontsize = 16, fontweight='bold')
            # Histogram
            self.ax = plt.subplot(self.gs[0:4,0])
            self.ax1,self.bins,_ = plt.hist(self.Data,bins='auto',normed=0,label='Histogram',histtype = 'bar',facecolor=[249/255,142/255,82/255],edgecolor='k')
#            self.ax.set_ylabel('Count')
            self.ax.set_xlabel('Measure')
            # NormalDist
            self.XNorm,self.YNorm = PlotNormDist(self.Data,0)
            self.ax2 = plt.plot(self.XNorm,self.YNorm,label='Gaussian Distribution',color=[72/255,132/255,175/255])
            # Skewed Dist
            self.XSkewDist, self.YSkewDist = PlotNormSkwedDist(self.Data,self.bins,np.absolute(self.bins[1]-self.bins[0]))
            self.ax3 = plt.plot(self.XSkewDist,self.YSkewDist,label='Skewed Gaussian',color='g')#[74/255,74/255,74/255])
# only Normalised: sns.distplot(self.Data,hist=True,kde=False, fit=stats.gamma);
            # Legend:                
            plt.legend(fontsize=6,loc='best')
            # Beeplot with Boxplot
            self.ax = plt.subplot(self.gs[0:4,1])
            self.ax.set_ylabel('Measure')
            self.ax.set_xticklabels([])
            sns.swarmplot(data=self.Data,palette=sns.color_palette("RdYlBu", n_colors=3))
            sns.boxplot(data=self.Data,palette=sns.color_palette("Blues_r", n_colors=2))
            self.ax.set_ylabel('Measure')
            self.ax.set_xticklabels([])
            # Information in Plot
            # Table of Info:
            self.AxInfo = plt.subplot(self.gs[4,:])
            self.AxInfo.axis('off')
            self.InfoArray = [None]*6
            if self.Norm == 1:
                self.InfoArray[0] = 'NormalDistribution: Yes'+'\n'
            else:
                self.InfoArray[0] = 'NormalDistribution: Yes'+'\n'
            if self.Skew == 1:
                self.InfoArray[1] = 'Skewed Distribution: Yes' +'\n'  
            else:
                self.InfoArray[1] = 'Skewed Distribution: No' +'\n'  
            self.InfoArray[2] = '%.f Values' %self.NumValues +'\n'
            
            self.InfoArray[3] = 'Mean: %.2f'%self.mean + ' +/- %.2f'%self.std + '\n'
            self.InfoArray[4] = 'Median: %.2f ' % self.median+'\n'
            self.InfoArray[5] = 'Percentiles: 95th : %.2f; ' % self.perc95 + '75th : %.2f; ' % self.perc75+ '25th : %.2f; ' % self.perc25 + '5th : %.2f; ' % self.perc5  + '\n'
            self.InfoText = ''.join(self.InfoArray)
            self.AxInfo.text(0,-1.45, self.InfoText,fontsize=9) 
            
        # Get Table with Values: 
        if self.Skew == 1 and self.Norm == 1:
            self.Distribution = 'Skewed Normal Distribution'
        elif self.Skew == 0 and self.Norm == 1: 
            self.Distribution = 'Normal Distribution'
        elif self.Skew == 1 and self.Norm == 0: 
            self.Distribution = 'Skewed Distribution'
        else:
            self.Distribution = 'Non-Normal Distribution'   

        # ResultsTable:
        self.Table = pd.DataFrame(index=['Distribution','# Samples','Mean','SD','Median','95th Percentile','75th Percentile','25th Percentile','5th Percentile'], columns=[self.Title])
        self.Table.loc['Distribution',self.Title] = self.Distribution 
        self.Table.loc['# Samples',self.Title] = self.NumValues
        self.Table.loc['Mean',self.Title] = self.mean
        self.Table.loc['SD',self.Title] = self.std
        self.Table.loc['Median',self.Title] = self.median
        self.Table.loc['95th Percentile',self.Title] = self.perc95
        self.Table.loc['75th Percentile',self.Title] = self.perc75
        self.Table.loc['25th Percentile',self.Title] = self.perc25
        self.Table.loc['5th Percentile',self.Title] = self.perc5



''' TO DO '''        
class DistributionStatGroups: 
    def __init__(self, X, PramaeterGroup,PrintShow = 0):
        self.Data = X #InputData Sorted as Column: Group, Index: Sample
        self.PrintShow = PrintShow      



        
''' C: Distribution Properties to each other: '''
def VarianceCalc(X, Y, Xnorm, Xskew, Ynorm, Yskew):
    '''>>>>> Levene’s test is an alternative to Bartlett’s test bartlett 
    in the case where there are significant deviations from normality.
    Three variations of Levene’s test are possible. The possibilities and their recommended usages are:
        ‘median’ : Recommended for skewed (non-normal) distributions>
        ‘mean’ : Recommended for symmetric, moderate-tailed distributions.
        ‘trimmed’ : Recommended for heavy-tailed distributions.'''
    if Xnorm == 1 and Ynorm ==1 and Xskew == 0 and Yskew == 0:
        # Barlett's Test and Levene's test mean
        TestResults,PValue = stats.bartlett(X,Y)
        TestResults,PValue = stats.levene(X,Y,center='mean')
    elif Xnorm == 1 and Ynorm ==1 and Xskew == 1 and Yskew == 1:
        # Levene’s test: median:
        TestResults,PValue = stats.levene(X,Y,center='median')
    else:
        # Levene's test: median: 
        TestResults,PValue = stats.levene(X,Y,center='median')
    if PValue > 0.05:
        EqualVariance = 1
    else:
        EqualVariance = 0
        
    return TestResults,PValue, EqualVariance

class Variance2Groups:
    def __init__(self, X, Y = 0, Xnorm=-1, Xskew = -1, Ynorm = -1, Yskew = -1, ParameterName = 'TBA',GroupNames='TBA', PrintShow = 0):
        self.X = X
        self.Y = Y
        self.PrintShow = PrintShow
        self.Title = ParameterName

        # Converte Data if nessessary:
        self.X = Convert(self.X) 
        self.Y = Convert(self.Y)
        # Get Main Values:
        self.XValues = DistPropertiesCalc(self.X)
        self.YValues = DistPropertiesCalc(self.Y) 

        # Get Prerequesits: 
        if Xnorm < 0:
            Xnorm,_ = NormalityCalc(self.X,0)
        if Xskew < 0:
            Xskew,_,_ = SkewnessCalc(self.X,0)
        if Ynorm < 0:
            Ynorm,_ = NormalityCalc(self.Y,0)
        if Yskew < 0:
            Yskew,_,_ = SkewnessCalc(self.Y,0)
        # Test: 
        self.EqVarianceTestStat,self.EqVariancePValue, self.EqualVariance = VarianceCalc(self.X, self.Y, Xnorm, Xskew, Ynorm, Yskew)
        
        # Plotting:
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
        # Figure
            self.Figure = plt.figure()
            self.Figure.set_dpi(200)
            self.gs = gridspec.GridSpec(6,2)
            self.gs.update(left=0.1, bottom= 0.075, top = 0.9, right=0.95,wspace=0.35)            
            self.Figure.suptitle(self.Title,fontsize = 16, fontweight='bold')
            self.ColorsGoup =[[250/255,169/255,0/255],\
             [6/255,162/255,203/255],\
             [33/255,133/255,89/255],\
             [91/255,37/255,131/255],\
             [234/255,207/255,0/255],\
             [139/255,92/255,159/255]]
        # Histogram/ Gaussian Dist:
            _,self.bins,_ = plt.hist(self.X,bins='auto')
            self.bins = len(self.bins)
            self.ax = plt.subplot(self.gs[0:4,0])
            self.ax = HistGausPlot(self.X,self.Y,self.ColorsGoup[0],self.ColorsGoup[1])
        # BoxPlots:
            self.ax = plt.subplot(self.gs[0:4,1])
            self.ax = SwarmBoxPlot([self.X,self.Y],self.ColorsGoup)
            self.ax.set_ylabel('Measure')
            self.ax.set_xticklabels([])            
        # TextBoxes: 
            self.axXText = self.ax = plt.subplot(self.gs[5:6,0])
            self.axXText.axis('off')
            XText1 = TextBoxPlot(self.XValues,Xnorm,Xskew)
            self.axXText.text(1.05,-0.3, XText1.InfoText,fontsize=7, ha = 'right')             
            self.axYText = self.ax = plt.subplot(self.gs[5:6,1])
            self.axYText.axis('off')
            YText1 = TextBoxPlot(self.YValues,Ynorm,Yskew)
            self.axYText.text(-0.25,-0.3, YText1.InfoText,fontsize=7) 
            if self.EqualVariance == 1: 
                self.VarianceTest = 'Equal Variances'
            else:
                self.VarianceTest = 'No Equal Variances'
            self.axXText.text(0,1, self.VarianceTest,fontsize=8,fontweight='bold')

''' D: Compare Distributions: Significance Test'''
# Different Tests:
''' 
Add Coffezient Intervals: https://www.uvm.edu/~dhowell/methods7/Supplements/Confidence%20Intervals%20on%20Effect%20Size.pdf
'''
def EffectSizeTTest(X,Y):
    ''' Cohen's D with pooled standard deviation, adjusted for different sample sizes
    Sources:    Cohen (1988):'Statistical Power Analysis for the bahavioral science' 
                Nakagawa & Cuthill (2007): 'Effect size, confidence interval and statistical significance: a practical guide for biologists'
    '''
    Meanx = np.mean(X)
    Meany = np.mean(Y)
    SDx = np.std(X)
    SDy = np.std(Y)
    Lx = len(X)
    Ly = len(Y)
    SDpooled = np.sqrt((((Lx-1)*SDx**2)+((Ly-1)*SDy**2))/(Lx+Ly-2))
    CohensD = (Meanx - Meany) / SDpooled
    return CohensD

def EffectSizePairedTTest(X,Y):
    ''' 
    Source:     Nakagawa & Cuthill (2007): 'Effect size, confidence interval and statistical significance: a practical guide for biologists'
    '''
    Differences = X-Y
    MeanDiff = np.mean(Differences)
    SDDiff = np.std(Differences)
    CohensD = MeanDiff/SDDiff
    return CohensD

def EffectSizeMWU(U,N1,N2):
    ''' Transformation U-Statistic to Z-Statistic (Uni Zürich: http://www.methodenberatung.uzh.ch/de/datenanalyse/unterschiede/zentral/mann.html)
        Transformation Z to r: Pallant (2007): 'SPSS Survival Manual' ;Todd Grande: https://www.youtube.com/watch?v=ILD5Jvmokig
        r to CohensD: Rosenthal (1994): 'Parametric measures of effect size'
    Source:
    '''
    Z = (U-((N1*N2)/2))/np.sqrt((N1*N2*(N1+N2+1))/(12))
    r = Z/np.sqrt((N1+N2))
    CohensD = (2*r)/np.sqrt(1-r**2)
    return Z,CohensD

def EffectSizeWSR(W,N1):
    ''' 
    Source:
        Transformation W-Statistic to Z-Statistic (Wikipedia:' Wilcoxon-Vorzeichen-Rang-Test')
        Transformation Z to r to CohensD: Fritz, Morris, and Richler(2011): 'Effect size estimates: Current use, calculations, and interpretation'
    '''
    Z = (W-(N1*(N1+1))/4)/np.sqrt((N1*(N1+1)*(2*N1+1))/(24))
    r = Z/np.sqrt(N1)
    CohensD = np.sqrt(r**2/((r**2)+4))
    return Z,CohensD

def NonOverlappEffectSize(CohenD):
    '''
    Cohen (1988):'Statistical Power Analysis for the bahavioral science' ;pp. 21- 23
    '''
    Table = np.zeros(shape = (2,31))
    Table[0,0] = 0.0
    i = 1
    while i < 21:
        Table[0,i] =Table[0,i-1]+0.1
        i += 1
    while i < 31:
        Table[0,i] =Table[0,i-1]+0.2
        i += 1
    Table[1,0] = 0;     Table[1,1] = 7.7;   Table[1,2] = 14.7;  Table[1,3] = 21.3;  
    Table[1,4] = 27.4;  Table[1,5] = 33.0;  Table[1,6] = 38.2;  Table[1,7] = 43.0;   
    Table[1,8] = 47.4;  Table[1,9] =51.6;   Table[1,10] = 55.4; Table[1,11] = 58.9;
    Table[1,12] = 62.2; Table[1,13] = 65.3; Table[1,14] = 68.1; Table[1,15] = 70.7;   
    Table[1,16] = 73.1; Table[1,17] = 75.4; Table[1,18] = 77.4; Table[1,19] = 79.4;   
    Table[1,20] = 81.1; Table[1,21] = 84.3; Table[1,22] = 87.0; Table[1,23] = 89.3;
    Table[1,24] = 91.2; Table[1,25] = 92.8; Table[1,26] = 94.2; Table[1,27] = 95.3;
    Table[1,28] = 96.3; Table[1,29] = 97.0; Table[1,30] = 97.7;
    value = np.absolute(CohenD)
    idx = (np.abs(Table[0,:]-value)).argmin()
    NonOverlap = Table[1,idx]
    Text =  '%.1f' % NonOverlap + ' % Non-Overlap between Groups'
    if value > 4.0:
        Text = 'more  97.7 % Non-Overlap between Groups' 
        NonOverlap = 99.9
    return NonOverlap, Text

def Test2GroupsCalc(X, Y, Sided, Paired, EqualVariance, Xnorm, Xskew, Ynorm, Yskew):
    # Parametric Tests:
    if Xnorm == 1 and Xskew == 0 and Ynorm == 1 and Yskew == 0:   
        if Paired <= 0:
            print('1')
            if EqualVariance == 1:
                Test = '2 Sample T-Test'
                TestStatistic, PValue = stats.ttest_ind(X,Y,equal_var=True, nan_policy='propagate')
                CohensD = EffectSizeTTest(X,Y)
                Z_Stats = np.nan
                
            elif EqualVariance == 0:
                Test = '2 Sample T-Test with Welch Correction'
                TestStatistic, PValue = stats.ttest_ind(X,Y,equal_var=False, nan_policy='propagate')
                CohensD = EffectSizeTTest(X,Y)
                Z_Stats = np.nan
                
            if Sided == 1:
                Test = 'One Sided ' + Test
                PValue = 0.5*PValue
        
        elif Paired >= 1:
            Test = 'Paired T-Test'
            TestStatistic, PValue = stats.ttest_rel(X,Y, axis=0, nan_policy='propagate')
            CohensD = EffectSizePairedTTest(X,Y)
            Z_Stats = np.nan#
            
        else:
            PValue = 0
            Test ='8ung'
            TestStatistic = None
            Z_Stats = None
            CohensD = 0
            
        if PValue < 0.05:
            Significance = 1
        else:
            Significance = 0 
            
    # Non-Parametric Tests:
    '''
    > Cave: Samplesize for Non-Parametric Test has to be > 20! otherwise: caution!
    '''
    if Xnorm == 0 or Xskew == 1 or Ynorm == 0 or Yskew == 1:
        ''' 8ung: MannWhitneyU and Willcoxon: Use only when the number of observation in each sample is > 20 
        and you have 2 independent samples of ranks. + Distributions have to have the same shape! > shift in medians +
        any relevance!  '''
        if Sided == 2 and Paired <= 0:
            Test = 'Mann-Whitney U-Test'
            TestStatistic, PValue = stats.mannwhitneyu(X,Y,use_continuity=True, alternative='two-sided')
            Z_Stats,CohensD=EffectSizeMWU(TestStatistic,len(X),len(Y))
        elif Sided == 1 and Paired <= 0:
            Test = 'One sided Mann-Whitney U-Test'
            TestStatistic, PValue = stats.mannwhitneyu(X,Y,use_continuity=True, alternative=None)
            Z_Stats,CohensD=EffectSizeMWU(TestStatistic,len(X),len(Y))
        elif Paired >= 1:
            Test = 'Willcoxon signed rank Test'
            TestStatistic, PValue = stats.wilcoxon(Y,X, zero_method='wilcox', correction=False)
            Z_Stats,CohensD = EffectSizeWSR(TestStatistic,len(X))
        # Significance:
        if PValue < 0.05:
            Significance = 1
        else:
            Significance = 0
       
    return Test, Significance, PValue, TestStatistic, Z_Stats, CohensD
    
    
# Main Script: 
class Test2ContinGroups:
    def __init__(self, X, Y = 0, YisGroupVec = -1,Sided = -1, Paired = -1, EqualVariance = -1, Xnorm=-1, Xskew = -1, Ynorm = -1, Yskew = -1, ParameterName = 'TBA',GroupNames='TBA', PrintShow = 0):
        self.X = X
        self.Y = Y 
        self.Sided = Sided
        self.Paired = Paired
        self.EqualVariance = EqualVariance
        self.Xnorm = Xnorm
        self.Xskew = Xskew
        self.Ynorm = Ynorm
        self.Yskew = Yskew
        self.Title = ParameterName
        self.PrintShow = PrintShow
        
        # GroupVector:
        if YisGroupVec > 0:
            self.X, self.Y,_ = GroupVecToDist(X,Y)
        
        # Converte Data if nessessary:
        self.X = Convert(self.X) 
        self.Y = Convert(self.Y)
        
        # Get Main Values:
        self.XValues = DistPropertiesCalc(self.X)
        self.YValues = DistPropertiesCalc(self.Y) 
        
        # Get Prerequesits: 
        if self.Xnorm < 0:
            self.Xnorm,_ = NormalityCalc(self.X,0)
        if self.Xskew < 0:
            self.Xskew,_,_ = SkewnessCalc(self.X,0)
        if self.Ynorm < 0:
            self.Ynorm,_ = NormalityCalc(self.Y,0)
        if self.Yskew < 0:
            self.Yskew,_,_ = SkewnessCalc(self.Y,0)
        if self.EqualVariance < 0: 
            TestEqualVariance = Variance2Groups(self.X, self.Y, self.Xnorm, self.Xskew, self.Ynorm, self.Yskew, ParameterName,GroupNames,0)
            self.EqualVariance = TestEqualVariance.EqualVariance
            if self.EqualVariance == 1:
                TextEqualVariance = 'Groups with equal Variances'
            else:
                TextEqualVariance = 'Groups without equal Variances'   
            
        # Test:
        self.Test, self.Significance, self.PValue, self.TestStatistic, self.Z_Stats, self.CohensD = Test2GroupsCalc(self.X, self.Y, self.Sided, self.Paired, self.EqualVariance, self.Xnorm, self.Xskew, self.Ynorm, self.Yskew)
        
        # Plotting:
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
        # Figure
            self.Figure = plt.figure()
            self.Figure.set_dpi(200)
            self.gs = gridspec.GridSpec(6,2)
            self.gs.update(left=0.1, bottom= 0.075, top = 0.9, right=0.95,wspace=0.35)            
            self.Figure.suptitle(self.Title,fontsize = 16, fontweight='bold')
            self.ColorsGoup =[[250/255,169/255,0/255],\
             [6/255,162/255,203/255],\
             [33/255,133/255,89/255],\
             [91/255,37/255,131/255],\
             [234/255,207/255,0/255],\
             [139/255,92/255,159/255]]
        # Histogram/ Gaussian Dist:
            self.ax = plt.subplot(self.gs[0:4,0])
            self.ax.set_ylabel('Count',fontsize = 9, labelpad = -0.5)
            plt.setp(self.ax.get_xticklabels(), fontsize = 9)
            self.ax.set_xlabel('Measure',fontsize = 9, labelpad = -1)
            plt.setp(self.ax.get_yticklabels(), fontsize = 9)
            self.ax = HistGausPlot(self.X,self.Y,self.ColorsGoup[0],self.ColorsGoup[1])
            
        # BoxPlots:
            self.ax = plt.subplot(self.gs[0:4,1])
            if Paired <= 0:
                self.ax = SwarmBoxPlot([self.X,self.Y],self.ColorsGoup)
                self.ax.set_ylabel('Measure',fontsize = 9, labelpad = -1)
                plt.setp(self.ax.get_yticklabels(), fontsize = 9)
                self.ax.set_xticklabels([])  
            elif Paired == 1:
                self.ax = sns.boxplot(data=[self.X,self.Y],palette=sns.color_palette(self.ColorsGoup))#,saturation=0.8)
                self.PltVec = np.zeros(shape=(2,len(self.X)))
                self.PltVec[1,:] = 1          
                self.ax = plt.plot(self.PltVec,[self.X,self.Y],'--ko',color=[125/255,125/255,125/255],alpha=0.5)
        # TextBoxes: 
            self.axXText = self.ax = plt.subplot(self.gs[5:6,0])
            self.axXText.axis('off')
            XText1 = TextBoxPlot(self.XValues,self.Xnorm,self.Xskew)
            self.axXText.text(1.05,-0.3, XText1.InfoText,fontsize=7, ha = 'right')             
            self.axYText = self.ax = plt.subplot(self.gs[5:6,1])
            self.axYText.axis('off')
            YText1 = TextBoxPlot(self.YValues,self.Ynorm,self.Yskew)
            self.axYText.text(-0.25,-0.3, YText1.InfoText,fontsize=7) 
            self.GroupText = plt.subplot(self.gs[4,0])
            self.GroupText.axis('off')
            InfoArray = [None]*4
            InfoArray[0] = TextEqualVariance +'\n'
            InfoArray[1] = 'Test: '+ self.Test + '\n'
            if self.Significance == 1:
                InfoArray[2] = 'Significant Difference with p = %.6f' % self.PValue+'\n'
            else:
                InfoArray[2] = 'No Significant Difference with p = %.6f' % self.PValue+'\n'   
            NonOverlapPValue, NonOverlappEffectSizeText = NonOverlappEffectSize(self.CohensD)
            InfoArray[3] = 'Effect Size (Cohen`s d): %.2f' % self.CohensD + ' -> ' + NonOverlappEffectSizeText
            self.GroupInfo = ''.join(InfoArray)
            self.GroupText.text(0,-.225, self.GroupInfo,fontsize=8,fontweight='bold')
        
            # Saving 
            if self.PrintShow ==1 or self.PrintShow == 2:
                self.SavingName = self.Title
                if self.PrintShow ==1 :
                    ddPlotting.save(self.SavingName, ext="jpg", close=True, verbose=True)
                if self.PrintShow == 2:
                    ddPlotting.save(self.SavingName, ext="svg", close=True, verbose=True)
                plt.close('All')
                plt.ion()
        
        # Printing Values to Table
        self.Table = pd.DataFrame(index=['X # Samples','X Mean','X SD','X Median','X 95th Percentile','X 75th Percentile','X 25th Percentile','X 5th Percentile','X Normal Dist','X Skewed Dist',\
                                         'Y # Samples','Y Mean','Y SD','Y Median','Y 95th Percentile','Y 75th Percentile','Y 25th Percentile','Y 5th Percentile','Y Normal Dist','Y Skewed Dist',\
                                         'EqualVariances','Test','Significant','PValue','Cohen`s D','% NonOverlap'], columns=[self.Title])
        self.Table.loc['X # Samples',self.Title] = self.XValues.NumValues
        self.Table.loc['X Mean',self.Title] = self.XValues.mean
        self.Table.loc['X SD',self.Title] = self.XValues.std
        self.Table.loc['X Median',self.Title] = self.XValues.median
        self.Table.loc['X 95th Percentile',self.Title] = self.XValues.perc95
        self.Table.loc['X 75th Percentile',self.Title] = self.XValues.perc75
        self.Table.loc['X 25th Percentile',self.Title] = self.XValues.perc25
        self.Table.loc['X 5th Percentile',self.Title] = self.XValues.perc5
        self.Table.loc['X Normal Dist',self.Title] = self.Xnorm
        self.Table.loc['X Skewed Dist',self.Title] = self.Xskew
        self.Table.loc['Y # Samples',self.Title] = self.YValues.NumValues
        self.Table.loc['Y Mean',self.Title] = self.YValues.mean
        self.Table.loc['Y SD',self.Title] = self.YValues.std
        self.Table.loc['Y Median',self.Title] = self.YValues.median
        self.Table.loc['Y 95th Percentile',self.Title] = self.YValues.perc95
        self.Table.loc['Y 75th Percentile',self.Title] = self.YValues.perc75
        self.Table.loc['Y 25th Percentile',self.Title] = self.YValues.perc25
        self.Table.loc['Y 5th Percentile',self.Title] = self.YValues.perc5
        self.Table.loc['Y Normal Dist',self.Title] = self.Ynorm
        self.Table.loc['Y Skewed Dist',self.Title] = self.Yskew
        
        self.Table.loc['EqualVariances',self.Title] = self.EqualVariance
        self.Table.loc['Test',self.Title] = self.Test
        self.Table.loc['Significant',self.Title] = self.Significance
        self.Table.loc['PValue',self.Title] = self.PValue
        self.Table.loc['Cohen`s D',self.Title] =  self.CohensD       
        self.Table.loc['% NonOverlap',self.Title] = NonOverlapPValue


''' E) Compare Distributions: Correlation/Regression: '''
'''
From Graphpad Prism (https://www.graphpad.com/support/faqid/1141/): Difference Regression vs. Correlation:
    Correlation quantifies the degree to which two variables are related. Correlation does not fit a line through 
    the data points. You simply are computing a correlation coefficient (r) that tells you how much one variable 
    tends to change when the other one does; r is 0.0, there is no relationship; r > 0 (positive) -> trend that one 
    variable goes up as the other one goes up;  r < 0 (negative) -> trend that one variable goes up as the other one 
    goes down. Correlation is almost always used when you measure both variables. With correlation, you don't have 
    to think about cause and effect. It doesn't matter which of the two variables you call "X" and which you call "Y".
    You'll get the same correlation coefficient if you swap the two. The correlation coefficient itself is simply a 
    way to describe how two variables vary together, so it can be computed and interpreted for any two variables. 
    Further inferences, however, require an additional assumption -- that both X and Y are measured, and both are 
    sampled from Gaussian distributions.  This is called a bivariate Gaussian distribution. If those assumptions 
    are true, then you can interpret the confidence interval of r and the P value testing the null hypothesis that 
    there really is no correlation between the two variables (and any correlation you observed is a consequence of 
    random sampling).
    Correlation computes the value of the Pearson correlation coefficient, r. Its value ranges from -1 to +1.
    
    Linear regression finds the best line that predicts Y from X.  Correlation does not fit a line. It rarely is 
    appropriate when one variable is something you experimentally manipulate. Linear regression is usually used 
    when X is a variable you manipulate (time, concentration, etc.) The decision of which variable you call "X" 
    and which you call "Y" matters in regression, as you'll get a different best-fit line if you swap the two. 
    The line that best predicts Y from X is not the same as the line that predicts X from Y (however both those 
    lines have the same value for R2). With linear regression, the X values can be measured or can be a variable 
    controlled by the experimenter. The X values are not assumed to be sampled from a Gaussian distribution. The 
    vertical distances of the points from the best-fit line (the residuals) are assumed to follow a Gaussian 
    distribution, with the SD of the scatter not related to the X or Y values.
    Linear regression quantifies goodness of fit with r2, sometimes shown in uppercase as R2.  If you put the same 
    data into correlation (which is rarely appropriate; see above), the square of r from correlation will equal r2 
    from regression. 
'''
def linearCorrelationCalc (X, Y, EqualVariance, Xnorm, Xskew, Ynorm, Yskew):
    '''Pearson Correlation:level of measurement (=  continuous Variables, ordinal in measurement, then a Spearman correlation),
    related pairs (> Get rid of NaNs!), absence of outliers(Typically, an outlier is defined as a value that is 3.29 standard 
    deviations from the mean, or a standardized value of less than ±3.29!!), normality of variables, linearity > ???
    Spearman's correlation determines the strength and direction of the monotonic 
    relationship between your two variables rather than the strength and direction of the linear relationship 
    between your two variables, which is what Pearson's correlation determines.
    '''
    if Xnorm == 1 and Ynorm ==1:# and Xskew == 0 and Yskew == 0:# and EqualVariance == 1:
        Test = 'Pearson Correlation'
        TestResults,PValue = stats.pearsonr(X,Y) 
    else:
        Test = 'Spearman’s rank correlation'
        TestResults,PValue = stats.spearmanr(X,Y,axis=0) 
        
    if PValue < 0.05:
        Significance = 1
    else:
        Significance = 0
        
    return Test,TestResults,PValue,Significance     


class LinearCorrelation:
    def __init__(self,X,Y, YisGroupVec = -1,EqualVariance = -1, Xnorm=-1, Xskew = -1, Ynorm = -1, Yskew = -1,ParameterNames=['TBA','TBA'], PrintShow = 0):
        self.X = X
        self.Y = Y
        self.Title = ParameterNames[0] +' vs. ' + ParameterNames[1]
        self.PrintShow = PrintShow
        
        # GroupVector:
        if YisGroupVec > 0:
            self.X, self.Y,_ = GroupVecToDist(X,Y)
            X,Y, _ = GroupVecToDist(X,Y)
        # Converte Data if nessessary:
        self.X = Convert(self.X) 
        self.Y = Convert(self.Y)
        
        # Get rid of NaNs
        self.xnans = np.argwhere(np.isnan(self.X))
        self.X = np.delete(self.X, self.xnans)
        self.Y = np.delete(self.Y, self.xnans)
        self.ynans = np.argwhere(np.isnan(self.Y))
        self.X = np.delete(self.X, self.ynans)
        self.Y = np.delete(self.Y, self.ynans)
        
        # Get Main Values:
        self.XValues = DistPropertiesCalc(self.X)
        self.YValues = DistPropertiesCalc(self.Y) 
        
        # Prerequisits:
        # Exclude outliers: 3.29 standard deviations (http://www.statisticssolutions.com/pearson-correlation-assumptions/)
        self.xOutlier = np.argwhere(np.absolute(self.X-self.XValues.mean) >= 3.29*self.XValues.std)
        self.X = np.delete(self.X, self.xOutlier)
        self.Y = np.delete(self.Y, self.xOutlier)
        self.yOutlier = np.argwhere(np.absolute(self.Y-self.YValues.mean) >= 3.29*self.YValues.std)
        self.X = np.delete(self.X, self.yOutlier)
        self.Y = np.delete(self.Y, self.yOutlier)
        
        # Normal Distribution:
        if Xnorm < 0:
            Xnorm,_ = NormalityCalc(self.X,0)
        if Xskew < 0:
            Xskew,_,_ = SkewnessCalc(self.X,0)
        if Ynorm < 0:
            Ynorm,_ = NormalityCalc(self.Y,0)
        if Yskew < 0:
            Yskew,_,_ = SkewnessCalc(self.Y,0)
        if EqualVariance < 0: 
            TestEqualVariance = Variance2Groups(self.X,self.Y, Xnorm, Xskew, Ynorm, Yskew,0)
            EqualVariance = TestEqualVariance.EqualVariance
            if EqualVariance == 1:
                TextEqualVariance = 'Groups with equal Variances'
            else:
                TextEqualVariance = 'Groups without equal Variances'  
                
        # Significance Test: 
        self.Test,self.TestResults,self.PValue,self.Significance = linearCorrelationCalc (self.X, self.Y, EqualVariance, Xnorm, Xskew, Ynorm, Yskew)
        # Regression Analysis
        self.slope, self.intercept, self.r_value, self.p_value, std_err = stats.linregress(self.X,self.Y)
        
         # Plotting:
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
        # Figure
            self.Figure = plt.figure()
            self.Figure.set_dpi(200)
            self.Figure.set_size_inches(5.9/1.25,8.2/1.25)
            self.gs = gridspec.GridSpec(3,2)
            self.gs.update(left=0.15, bottom= 0.075, top = 0.9, right=0.95,wspace=0.35)            
            self.Figure.suptitle(self.Title,fontsize = 16, fontweight='bold',y=0.945)
            self.ColorsGoup =[[250/255,169/255,0/255],\
             [6/255,162/255,203/255],\
             [33/255,133/255,89/255],\
             [91/255,37/255,131/255],\
             [234/255,207/255,0/255],\
             [139/255,92/255,159/255]]
            
        # Linear Regression:
            self.ax = plt.subplot(self.gs[0:2,:])
            self.ax = plt.scatter(X,Y,marker = '+', color = 'r')
            self.ax = sns.regplot(x = self.X, y = self.Y, color=[125/255,125/255,125/255])
            # Inlet:
            InletArray = [None]*2
            if self.Test =='Pearson Correlation':
                InletArray[0] =  'PearsonR = %.2f' % self.TestResults+'; p = %.4f' % self.PValue+'\n'
            else:
                InletArray[0] =  'SpearmanR = %.2f' % self.TestResults+'; p = %.4f' % self.PValue+'\n'
            InletArray[1] = 'Slope: %.4f' % self.slope + '; Intercept: %.4f' % self.intercept
            Inlet = ''.join(InletArray)
            if self.slope < 0:
                self.anchored_text = AnchoredText(Inlet, loc=1,frameon=False,pad=0)
            else:
                self.anchored_text = AnchoredText(Inlet, loc=2,frameon=False,pad=0)
            self.ax.add_artist(self.anchored_text)
            # Axes:
            self.ax = plt.xlabel(ParameterNames[0])
            self.ax = plt.ylabel(ParameterNames[1]) 
        # InfoText Single Distributions:
            self.axXText = plt.subplot(self.gs[2,0])
            self.axXText.axis('off')
            XText1 = TextBoxPlot(self.XValues,Xnorm,Xskew)
            self.axXText.text(1,0.15, XText1.InfoText,fontsize=5, ha = 'right')
            self.axXText.text(1,0.45, ParameterNames[0],fontsize=10, ha = 'right')
            self.axYText = self.ax = plt.subplot(self.gs[2,1])
            self.axYText.axis('off')
            YText1 = TextBoxPlot(self.YValues,Ynorm,Yskew)
            self.axYText.text(-0.25,0.15, YText1.InfoText,fontsize=5) 
            self.axYText.text(-0.25,0.45, ParameterNames[1],fontsize=10)
        # InfoText Correlation:
            self.GroupText = plt.subplot(self.gs[2,0])
            self.GroupText.axis('off')
            InfoArray = [None]*4
            InfoArray[0] = TextEqualVariance +'\n'
            InfoArray[1] = 'Test: '+ self.Test + '\n'
            if self.Significance == 1:
                InfoArray[2] = 'Significant Correlation with p = %.6f' % self.PValue+'\n'
            else:
                InfoArray[2] = 'No Significant Correlation with p = %.6f' % self.PValue+'\n'  
            InfoArray[3] = 'Slope: %.4f' % self.slope + '; Intercept: %.4f' % self.intercept
            self.GroupInfo = ''.join(InfoArray)
            self.GroupText.text(0,0.615, self.GroupInfo,fontsize=8,fontweight='bold')
            
            # Saving 
            # Saving 
            if self.PrintShow ==1 or self.PrintShow == 2:
                self.SavingName = self.Title
                if self.PrintShow ==1 :
                    ddPlotting.save(self.SavingName, ext="jpg", close=True, verbose=True)
                if self.PrintShow == 2:
                    ddPlotting.save(self.SavingName, ext="svg", close=True, verbose=True)
                plt.close('All')
                plt.ion() 
            
        # Printing Values to Table
        self.Table = pd.DataFrame(index=['X # Samples','X Mean','X SD','X Median','X 95th Percentile','X 75th Percentile','X 25th Percentile','X 5th Percentile','X Normal Dist','X Skewed Dist',\
                                         'Y # Samples','Y Mean','Y SD','Y Median','Y 95th Percentile','Y 75th Percentile','Y 25th Percentile','Y 5th Percentile','Y Normal Dist','Y Skewed Dist',\
                                         'EqualVariances','Test','Significant','PValue','Slope','YIntercept'], columns=[self.Title])
        self.Table.loc['X # Samples',self.Title] = self.XValues.NumValues
        self.Table.loc['X Mean',self.Title] = self.XValues.mean
        self.Table.loc['X SD',self.Title] = self.XValues.std
        self.Table.loc['X Median',self.Title] = self.XValues.median
        self.Table.loc['X 95th Percentile',self.Title] = self.XValues.perc95
        self.Table.loc['X 75th Percentile',self.Title] = self.XValues.perc75
        self.Table.loc['X 25th Percentile',self.Title] = self.XValues.perc25
        self.Table.loc['X 5th Percentile',self.Title] = self.XValues.perc5
        self.Table.loc['X Normal Dist',self.Title] = Xnorm
        self.Table.loc['X Skewed Dist',self.Title] = Xskew
        self.Table.loc['Y # Samples',self.Title] = self.YValues.NumValues
        self.Table.loc['Y Mean',self.Title] = self.YValues.mean
        self.Table.loc['Y SD',self.Title] = self.YValues.std
        self.Table.loc['Y Median',self.Title] = self.YValues.median
        self.Table.loc['Y 95th Percentile',self.Title] = self.YValues.perc95
        self.Table.loc['Y 75th Percentile',self.Title] = self.YValues.perc75
        self.Table.loc['Y 25th Percentile',self.Title] = self.YValues.perc25
        self.Table.loc['Y 5th Percentile',self.Title] = self.YValues.perc5
        self.Table.loc['Y Normal Dist',self.Title] = Ynorm
        self.Table.loc['Y Skewed Dist',self.Title] = Yskew
        
        self.Table.loc['EqualVariances',self.Title] = EqualVariance
        self.Table.loc['Test',self.Title] = self.Test
        self.Table.loc['Significant',self.Title] = self.Significance
        self.Table.loc['PValue',self.Title] = self.PValue
        self.Table.loc['Slope',self.Title] =  self.slope       
        self.Table.loc['YIntercept',self.Title] = self.intercept            
        
        
''' F) Compare MultiGroups: Significance Test '''
'''
One Way Anova Assumtions:   The samples are independent. stats.f_oneway(sample1, sample2, ... : array_like)
                            Each sample is from a normally distributed population. 
                            Equal Variances: The population standard deviations of the groups are all equal. This property is known as homoscedasticity.        
        > Post-Hoc: Turkey Kramer Test, get P-Values and Effect-Size
KruskalWallisTest
        > Post-Hoc: Dunnes Test > get P-Values and Effect Size
'''     
def OwnOneWayANOVA (DataList,Data,GroupVariable):
    Test = 'OneWayANOVA'
    # Test:
    TestStatistic, PValue = stats.f_oneway(*DataList)
    
    ''' Effect Size OneWayANOVA: 
    https://stats.stackexchange.com/questions/67926/understanding-the-one-way-anova-effect-size-in-scipy '''
    Data_OneVec = np.hstack( np.concatenate( DataList))
    ss_total = sum( (Data_OneVec - np.mean( Data_OneVec)) **2)
    grand_mean = np.mean(Data_OneVec)
    ss_btwn = 0
    for a in DataList:
        ss_btwn += ( len(a) * ( np.mean( a) - grand_mean) **2)    
    eta_squared =  ss_btwn / ss_total
    CohenD = ((eta_squared)/(1-eta_squared))
    
    ''' Post-Hoc Turkey Kramer Test with P-Value
    > https://stackoverflow.com/questions/16049552/what-statistics-module-for-python-supports-one-way-anova-with-post-hoc-tests-tu
    > http://jpktd.blogspot.de/2013/03/multiple-comparison-and-tukey-hsd-or_25.html
    > https://code.google.com/archive/p/qsturng-py/ '''
    res2 = pairwise_tukeyhsd(Data,GroupVariable)
    rs = res2.meandiffs / res2.std_pairs
    NumGroups = len(res2.groupsunique)
    DegressOfFreedom = res2.df_total
    Single_PValues = psturng(np.abs(rs), NumGroups, DegressOfFreedom)
        
    # Single Effect Sizes: Cohnens D: with pooled std over all Groups
    i = 0
    SampleSize =[None]*len(DataList)
    SampleStd =[None]*len(DataList)
    while i < len(DataList):
        SampleSize[i]=len(DataList[i])
        SampleStd[i] = (SampleSize[i]-1)*np.std(DataList[i])**2
        i += 1
    PooledSD = np.sqrt(sum(SampleStd)/(sum(SampleSize)-len(DataList)))
    SingleCohnsD = res2.meandiffs/PooledSD
    return Test, TestStatistic, PValue, CohenD, res2, Single_PValues, SingleCohnsD


def KrusWallis (DataList,GroupNumber):
    '''Kruskal-Wallis 1-way ANOVA with Dunn's multiple comparison test
        https://gist.github.com/alimuldal/fbb19b73fa25423f02e8 '''
    Test = 'Kruskal-Wallis OneWayANOVA'
    KW_TestStatistic, KW_PValue, Z_Pairs, Single_PValues, rejected = dunn.kw_dunn(DataList, to_compare=None, alpha=0.05, method='bonf')
    ''' Effect-Size Kruskal-Wallis OneWayANOVA
    > http://psytistics.com/tests/kruskal-wallis: η2 = χ2/(N-1) '''
    NumOb = len(np.hstack(np.concatenate(DataList)))
    eta_squared = KW_TestStatistic/(NumOb-1)
    CohenD = ((eta_squared)/(1-eta_squared))
    ''' Effect-Size Pairwise Test 
    > Post-Hoc: http://yatani.jp/teaching/doku.php?id=hcistats:kruskalwallis
    > Single Z Statistics for r = Z/sqrt(N; is the total number of the samples) 
    > r to d r to CohensD: Rosenthal (1994): 'Parametric measures of effect size '''
    r = Z_Pairs/np.sqrt(NumOb)
    SingleCohensD = (2*r)/np.sqrt(1-r**2)
    return Test, KW_TestStatistic, KW_PValue,CohenD, Z_Pairs, Single_PValues, rejected,SingleCohensD
   
class TestContinMultiGroup:
    def __init__(self, X, Y = 0, YisGroupVec = 1, ParameterName = 'TBA',GroupNames='TBA', PrintShow = 0):
        self.Data = X
        self.GroupVec = Y
        self.PrintShow = PrintShow
        self.ParameterName = ParameterName
        self.Title = ParameterName
        
        # Get Value Array:
        _,_,self.DataList = GroupVecToDist(self.Data,self.GroupVec)
        self.GroupNum = len(self.DataList)
        self.GroupVecOne = np.ndarray.flatten(self.GroupVec)
        # ParameterNames
        if GroupNames == 'TBA':
            self.GroupNames = ['TBA']*self.GroupNum 
        else:
            self.GroupNames = GroupNames
        
        # Single Distribution Values:
        i = 0
        self.DistValues=[None]*self.GroupNum
        while i < self.GroupNum:
            self.DistValues[i] = DistPropertiesCalc(self.DataList[i])
            i += 1
        
        # Test for Normality and Skewness:
        i = 0
        self.NormAll=[None]*self.GroupNum
        self.SkewAll=[None]*self.GroupNum
        while i < self.GroupNum:
            self.NormAll[i],_ = NormalityCalc(self.DataList[i],0)
            self.SkewAll[i],_,_ = SkewnessCalc(self.DataList[i],0)
            i += 1
        
        # Test for EqualVariance:
        ''' Cave: Zimmerman, D.W. (2004): "A note on preliminary tests of equality of variances."
            1) If sample sizes are equal, you don't have a problem. ANOVA is quite robust to 
                different variances if the n's are equal.
            2) testing equality of variance before deciding whether to assume it is recommended 
                against by a number of studies. If you're in any real doubt that they'll be close 
                to equal, it's better to simply assume they're unequal.
        '''
        if np.sum(self.NormAll) == self.GroupNum and np.sum(self.SkewAll) == 0: 
            self.EqualVarianceStatistic, self.EqualVariancePValue = stats.levene(*self.DataList,center='mean')
        else:
            self.EqualVarianceStatistic, self.EqualVariancePValue = stats.levene(*self.DataList,center='median')
        if np.sum(self.NormAll) == self.GroupNum:
            self.AllNorm = 1
        else:
            self.AllNorm = 0
        if np.sum(self.SkewAll) == self.GroupNum:
            self.AllSkew = 1
        else:
            self.AllSkew = 0
            
        if self.EqualVariancePValue > 0.05:
            self.EqualVariance = 1
            self.VarianceText = 'All Groups with equal Variance'
        else:
            self.EqualVariance = 0
            self.VarianceText = 'All Groups without equal Variance'
        
        # Testing ANOVA or KruskalWallisTest
        if np.sum(self.NormAll) == self.GroupNum:#  and self.EqualVariance == 1: #and np.sum(self.SkewAll) == 0:
            self.Test, self.TestStatistic, self.PValue,self.CohenD, self.res2, self.Single_PValues,self.SingleCohenD = OwnOneWayANOVA (self.DataList,self.Data,self.GroupVecOne)
        else:
            self.Test, self.TestStatistic, self.PValue, self.CohenD, self.Z_Pairs, self.Single_PValues, rejected, self.SingleCohenD = KrusWallis (self.DataList,self.GroupVecOne)
        if self.PValue < 0.05:
            self.Significance = 1
        else:
            self.Significance = 0
                
        
        # ResultStructur-PostHoc Tested Groups:
        to_compare = tuple(combinations(range(len(self.DataList)), 2))
        self.GroupStructur = np.asarray(to_compare)

        # Plotting:
        if self.PrintShow <= 1:
            plt.ioff()
        else:
            plt.ion()
        if self.PrintShow >=1:
        # Figure
            self.Figure = plt.figure()
            self.Figure.set_dpi(200)
            self.gs = gridspec.GridSpec(3,self.GroupNum)
            self.gs.update(left=0.1, bottom= 0.075, top = 0.9, right=0.95,wspace=0.35)       
            self.Title = self.Title + ' ' + self.Test + ': p = %.4f' % self.PValue +'; d = %.4f' %self.CohenD 
            self.Figure.suptitle(self.Title,fontsize = 9, fontweight='bold')
            self.ColorsGoup =[[250/255,169/255,0/255],\
             [6/255,162/255,203/255],\
             [33/255,133/255,89/255],\
             [91/255,37/255,131/255],\
             [234/255,207/255,0/255],\
             [139/255,92/255,159/255]]
            
        # BoxPlot:
            self.ax = plt.subplot(self.gs[0:2,:])
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax = SwarmBoxPlot(self.DataList,self.ColorsGoup)
            
            # General Layout:
            self.ax.set_xticklabels(self.GroupNames,ha='right')
            self.ax.set_ylabel(self.ParameterName)
            
            # statistical annotation:
            i = 0
            while i < len(self.GroupStructur):
                x =  np.hstack(np.concatenate([self.GroupStructur[i],self.GroupStructur[i]]))
                x.sort()
                yhelp = np.max(self.Data)+ np.std(self.Data)/4*(i+1)
                yhelp2 = yhelp-np.std(self.Data)/8
                y =[yhelp2, yhelp, yhelp, yhelp2]
                if self.Single_PValues[i]<0.05:
                    self.ax = plt.plot(x,y,'k')  
                    if self.Single_PValues[i]<0.0001: 
                        SigText = '*** (p = %.4f' % self.Single_PValues[i] +'; d: %.2f' % self.SingleCohenD[i] +')'
                    elif self.Single_PValues[i]<0.001: 
                        SigText = '** (p = %.4f' % self.Single_PValues[i] +'; d: %.2f' % self.SingleCohenD[i] +')'
                    elif self.Single_PValues[i]<0.05: 
                        SigText = '* (p = %.4f' % self.Single_PValues[i] +'; d: %.2f' % self.SingleCohenD[i] +')'
                    self.ax = plt.text(sum(x)*.25, yhelp, SigText, ha='center', va='bottom',fontsize=6)
                else:
                    self.ax = plt.plot(x,y,color =[225/255,225/255,225/255])  
                    SigText = 'ns (p = %.4f' % self.Single_PValues[i] +'; d: %.2f' % self.SingleCohenD[i] +')'
                    self.ax = plt.text(sum(x)*.25, yhelp, SigText, ha='center', va='bottom',fontsize=6,color =[225/255,225/255,225/255])
                i += 1
            
        # Single Dist Text:
            i = 0
            self.axDistText =[None]*self.GroupNum
            while i < self.GroupNum:
                self.axDistText[i] = plt.subplot(self.gs[2,i])
                self.axDistText[i].axis('off')
                InfoArray = [None]*9
                InfoArray[0] = '%.f Values' %self.DistValues[i].NumValues +'\n'
                InfoArray[1] = 'Mean: %.2f'%self.DistValues[i].mean + ' +/- %.2f'%self.DistValues[i].std + '\n'
                InfoArray[2] = 'Median: %.2f' %self.DistValues[i].median+'\n'
                InfoArray[3] = '95th Pctl: %.2f; ' %self.DistValues[i].perc95 + '\n'
                InfoArray[4] = '75th Pctl: %.2f; ' %self.DistValues[i].perc75 + '\n'
                InfoArray[5] = '25th Pctl: %.2f; ' %self.DistValues[i].perc25 + '\n'
                InfoArray[6] = '5th Pctl: %.2f' % self.DistValues[i].perc5  + '\n'
                InfoArray[7] = 'Norm: %.f' %self.NormAll[i]  + '\n'
                InfoArray[8] = 'Skew: %.f' % self.SkewAll[i]  + '\n'
                self.InfoText = ''.join(InfoArray)
                self.axDistText[i].text(0,0, self.InfoText,fontsize=6,ha='left',va='bottom',ma='center')
                i += 1
                
            self.axAll = plt.subplot(self.gs[2,0])
            self.axAll.axis('off')
            self.axAll.text(0,0,self.VarianceText,fontsize=6,ha='left',va='top')#,ma='center')

            # Saving 
            if self.PrintShow ==1 or self.PrintShow == 2:
                self.SavingName = self.Title
                if self.PrintShow ==1 :
                    ddPlotting.save(self.SavingName, ext="jpg", close=True, verbose=True)
                if self.PrintShow == 2:
                    ddPlotting.save(self.SavingName, ext="svg", close=True, verbose=True)
                plt.close('All')
                plt.ion()

        # ResultTable:
        # General:
        self.Table1 = pd.DataFrame(index=['Num Groups','Test','Significant','PValue','CohenD','EqualVariances','All Norm','All NonSkew'], columns=[self.ParameterName])
        self.Table1.loc['Num Groups',self.ParameterName] = self.GroupNum
        self.Table1.loc['Test',self.ParameterName] = self.Test
        self.Table1.loc['Significant',self.ParameterName] = self.Significance
        self.Table1.loc['PValue',self.ParameterName] = self.PValue
        self.Table1.loc['CohenD',self.ParameterName] = self.CohenD
        self.Table1.loc['EqualVariances',self.ParameterName] = self.EqualVariance
        self.Table1.loc['All Norm',self.ParameterName] = self.AllNorm
        self.Table1.loc['All NonSkew',self.ParameterName] = self.AllSkew
        
        
        # PostHoc-Significance:
        self.GroupStrucTableText1 =[None]*len(self.GroupStructur)
        self.GroupStrucTableText2 =[None]*len(self.GroupStructur)
        i = 0
        while i < len(self.GroupStructur):
            self.GroupStrucTableText1[i] = 'P-Value: %.0f' %(self.GroupStructur[i,0]+1) + ' vs. ' + '%.0f' %(self.GroupStructur[i,1]+1)   
            self.GroupStrucTableText2[i] = 'CohenD: %.0f' %(self.GroupStructur[i,0]+1) + ' vs. ' + '%.0f' %(self.GroupStructur[i,1]+1)   
            i += 1
            
        self.Table2 = pd.DataFrame(index=self.GroupStrucTableText1, columns=[self.ParameterName])
        self.Table3 = pd.DataFrame(index=self.GroupStrucTableText2, columns=[self.ParameterName])
        i = 0
        while i < len(self.GroupStructur):
            self.Table2.loc[self.GroupStrucTableText1[i],self.ParameterName] = self.Single_PValues[i]    
            self.Table3.loc[self.GroupStrucTableText2[i],self.ParameterName] = self.SingleCohenD[i]    
            i += 1
        self.Tableframes = [self.Table1,self.Table2,self.Table3]
        self.Table = pd.concat(self.Tableframes)
        



''' For Testing: '''
''' Test Data: '''
#TestDataPrev = np.random.normal(100, 1 ,11)
#TestDataPrev1 = np.random.normal(100, 1 ,12)
TestDataPrev = stats.skewnorm.rvs(1.3, size=50,loc=0.525)
TestDataPrev1 = stats.skewnorm.rvs(0.2, size=25,loc=1.25)
TestDataPrev2 = stats.skewnorm.rvs(0.3, size=25,loc=0.25)
TestDataPrev2[2:15] = np.nan
TestDataPrev3 = stats.skewnorm.rvs(20.3, size=25,loc=0.25)
#TestDataPrev1 = TestDataPrev*8+np.random.normal(0,5,12)
#TestDataPrev1 [-1] = 1127
#TestData = pd.DataFrame(data=TestData)
TestGroup = [TestDataPrev,TestDataPrev1,TestDataPrev2,TestDataPrev3]
TestGroup = np.hstack(np.concatenate(TestGroup))

GroupVec1 = np.ones(shape=(1,125))
GroupVec1[0,50:100] = 2
GroupVec1[0,75:100] = 3
GroupVec1[0,100:125] = 4

''' Function To Call:'''
#TestData = Convert (TestDataPrev)
#plt.figure
#plt.hist(TestData)
#A = DistributionStatSingle(TestDataPrev,'TBA',2)
#B = Variance2Groups(TestDataPrev1,TestDataPrev, PrintShow = 2)
#C = Test2ContinGroups(TestDataPrev1,TestDataPrev, Sided = 2, Paired = 0, PrintShow = 2,ParameterName = 'Fiction [Mega]')
#D = C.Table
#E = LinearCorrelation(TestDataPrev1,TestDataPrev2,ParameterNames = ['Fic','Ima'], PrintShow = 2)
#F = E.Table

#G = TestContinMultiGroup(TestGroup, GroupVec1,PrintShow=2)
#G1 = G.Table
#G2 = G.Table2
#G3 = G.Table3