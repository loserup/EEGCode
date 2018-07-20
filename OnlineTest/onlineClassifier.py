# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:03:43 2018

@author: Long
"""
import numpy as np
from sklearn.externals import joblib
import scipy.io as sio
import scipy.signal as sis

class OnlineClassifier():
    
    def __init__(self):
        """__init__: OnlineClassifier类的初始化

        Parameters:
        -----------     
        - feat_input: MATLAB传来的特征数据
        """        
        self.eeg_data = sio.loadmat('data.mat')['data']
        self.classifier = joblib.load('SVM.m')
        self.csp = sio.loadmat('csp.mat')['csp']
        
        self.fs = 512
        self.win_width = 384
        
    def outputCmd(self):
        """outputCmd: 用分类器预测输入的特征数据类别，并返回类别
        """
        out_eeg_band0 = self.bandpass(self.eeg_data,upper=0.3,lower=3)
        out_eeg_band1 = self.bandpass(self.eeg_data,upper=4,lower=7)
        out_eeg_band2 = self.bandpass(self.eeg_data,upper=8,lower=13)
        out_eeg_band3 = self.bandpass(self.eeg_data,upper=13,lower=30)
        test_eeg = np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3))
        Z = np.dot(self.csp, test_eeg)
        varances = list(np.var(Z, axis=1))
        test_feat = np.array([np.log(x/sum(varances)) for x in varances]) # 标准化
        test_feat = test_feat.reshape(1,len(self.csp)) # classifier.predict需要和fit时相同的数据结构，所以要reshape
        
        return int(self.classifier.predict(self.test_feat))
    
    def bandpass(self, data,upper,lower):
        Wn = [2 * upper / self.fs, 2 * lower / self.fs] # 截止频带0.1-1Hz or 8-30Hz
        b,a = sis.butter(4, Wn, 'bandpass')
    
        filtered_data = np.zeros([32, self.win_width])
        for row in range(32):
            filtered_data[row] = sis.filtfilt(b,a,data[row,:]) 
    
        return filtered_data 

            