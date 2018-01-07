# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:21:17 2018

@author: Long

% 说明:

% 第六步

% 输入数据最后一列为标签，0表示无意图，1表示有意图
% 数据其余列为特征值
"""

from sklearn.svm import SVC
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn import cross_validation

id_subject = 3 # 【受试者的编号】

if id_subject < 10:
    feats_mat = sio.loadmat('E:\EEGExoskeleton\EEGProcessor2\Subject_0'+str(id_subject)+\
                '_feature\\Subject_0'+str(id_subject)+'_features.mat')
else:
    feats_mat = sio.loadmat('E:\EEGExoskeleton\EEGProcessor2\Subject_'+str(id_subject)+\
                '_feature\\Subject_'+str(id_subject)+'_features.mat')
feats_all = feats_mat['features']
    
# 随机打乱特征顺序
feats, labels = shuffle(feats_all[:,:-1],feats_all[:,-1],random_state=50000)
"""
num_train = int(0.8*len(feats)) # 设置80%的数据用来训练
feats_trian, labels_train = feats[:num_train],labels[:num_train]
feats_test, labels_test = feats[num_train:],labels[num_train:]
"""
# 建立SVM模型
params = {'kernel':'rbf','probability':True}
classifier = SVC(**params)
classifier.fit(feats,labels)

accuracy = cross_validation.cross_val_score(classifier, feats, labels,\
                                            scoring='accuracy',cv=3)
print ('Accuracy of the classifier: '+str(round(100*accuracy.mean(),2))+'%')