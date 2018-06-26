# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:03:43 2018

@author: Long
"""
from sklearn.svm import SVC
import scipy.io as sio
from sklearn.utils import shuffle
import numpy as np

class OnlineClassifier():
    
    def __init__(self, feat_input):
        """__init__: OnlineClassifier类的初始化

        Parameters:
        -----------     
        - feat_input: MATLAB传来的特征数据
        """        
        self.feat_input = np.mat(feat_input).reshape(1,8)
        self.feats_all = sio.loadmat('features.mat')['features']
        self.feats, self.labels = \
            shuffle(self.feats_all[:,:-1],self.feats_all[:,-1],\
                    random_state=100)
        self.params = \
            {'kernel':'rbf','probability':True, 'class_weight':'balanced'}
        self.classifier = SVC(**(self.params))
        self.classifier.fit(self.feats,self.labels)
        
    def outputCmd(self):
        """outputCmd: 用分类器预测输入的特征数据类别，并返回类别
        """
        return int(self.classifier.predict(self.feat_input))

            