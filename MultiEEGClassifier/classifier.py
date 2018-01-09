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

id_subject = 3 # 【受试者的编号】

if id_subject < 10:
    feats_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0'+\
                            str(id_subject)+'_Data\\Subject_0'+\
                            str(id_subject)+'_features_4class.mat')
else:
    feats_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+\
                            str(id_subject)+'_Data\\Subject_'+\
                            str(id_subject)+'_features_4class.mat')

feats_all = feats_mat['features']

# 随机打乱特征顺序
feats, labels = shuffle(feats_all[:,:-1],feats_all[:,-1],random_state=3)

# 建立SVM模型
params = {'kernel':'rbf','probability':True} # 类别0明显比其他类别数目多，但加了'class_weight':'balanced'平均各类权重准确率反而更低了
classifier = SVC(**params)
classifier.fit(feats,labels)

accuracy = cross_validation.cross_val_score(classifier, feats, labels,\
                                            scoring='accuracy',cv=3)
print ('Accuracy of the classifier: '+str(round(100*accuracy.mean(),2))+'%')