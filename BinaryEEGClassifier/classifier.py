# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:21:17 2018

@author: Long

% 说明:

% 第五步

% 输入数据最后一列为标签，0表示无意图，1表示有意图
% 数据其余列为特征值
"""

from sklearn.svm import SVC
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn import cross_validation
import numpy as np

id_subject = 3 # 【受试者的编号】

if id_subject < 10:
    feats_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0'+\
                            str(id_subject)+'_Data\\Subject_0'+\
                            str(id_subject)+'_features.mat')
else:
    feats_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+\
                            str(id_subject)+'_Data\\Subject_'+\
                            str(id_subject)+'_features.mat')

feats_all = feats_mat['features']

# 随机打乱特征顺序
for i in range(10):
    feats, labels = shuffle(feats_all[:,:-1],feats_all[:,-1],\
                            random_state=np.random.randint(0,100))
    # 建立SVM模型
    params = {'kernel':'rbf','probability':True}
    classifier = SVC(**params)
    classifier.fit(feats,labels)
    accuracy = cross_validation.cross_val_score(classifier, feats, labels,\
                                            scoring='accuracy',cv=3)
    print ('Accuracy of the classifier: '+str(round(100*accuracy.mean(),2))+'%')