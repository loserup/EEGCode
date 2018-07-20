# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:03:43 2018

@author: Long
"""
import numpy as np
from sklearn.externals import joblib

class OnlineClassifier():
    
    def __init__(self, feat_input):
        """__init__: OnlineClassifier类的初始化

        Parameters:
        -----------     
        - feat_input: MATLAB传来的特征数据
        """        
        self.feat_input = np.mat(feat_input).reshape(1,8)
        self.classifier = joblib.load('SVM.m')
        
    def outputCmd(self):
        """outputCmd: 用分类器预测输入的特征数据类别，并返回类别
        """
        return int(self.classifier.predict(self.feat_input))

            