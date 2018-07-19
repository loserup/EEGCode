# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:03:43 2018

@author: Long
"""
import numpy as np
from sklearn.externals import joblib

class OnlineClassifier():
    
    def __init__(self, data_history, count, win_width):
        """__init__: OnlineClassifier类的初始化

        Parameters:
        -----------     
        - data_history: MATLAB传来的历史脑电信号
        - count: MATLAB传来的脑电信号采样数
        """        
        self.feat_input = np.mat(feat_input).reshape(1,8)
        self.classifier = joblib.load('SVM.m')
        
    def outputCmd(self):
        """outputCmd: 用分类器预测输入的特征数据类别，并返回类别
        """
        return int(self.classifier.predict(self.feat_input))

            