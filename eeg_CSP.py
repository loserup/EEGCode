# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:07:25 2017

@author: Long
"""

import scipy.io as sio
import numpy as np
import scipy.linalg as la # 线性代数库

eeg = sio.loadmat('EEG_win1.mat')
eeg_1 = eeg['EEG_win1'] # 障碍物高度3
eeg = sio.loadmat('EEG_win2.mat')
eeg_2 = eeg['EEG_win2'] # 障碍物高度2
eeg = sio.loadmat('EEG_win3.mat')
eeg_3 = eeg['EEG_win3'] # 障碍物高度1
eeg = sio.loadmat('EEG_win4.mat')
eeg_4 = eeg['EEG_win4'] # 障碍物高度1
eeg = sio.loadmat('EEG_win5.mat')
eeg_5 = eeg['EEG_win5'] # 障碍物高度2
eeg = sio.loadmat('EEG_win6.mat')
eeg_6 = eeg['EEG_win6'] # 障碍物高度3
eeg = sio.loadmat('EEG_win7.mat')
eeg_7 = eeg['EEG_win7'] # 障碍物高度2
eeg = sio.loadmat('EEG_win8.mat')
eeg_8 = eeg['EEG_win8'] # 障碍物高度3
eeg = sio.loadmat('EEG_win9.mat')
eeg_9 = eeg['EEG_win9'] # 障碍物高度1

hight_1 = (eeg_3, eeg_4, eeg_9)
hight_2 = (eeg_2, eeg_5, eeg_7)
hight_3 = (eeg_1, eeg_6, eeg_8)

task = (hight_1, hight_2)

# 获取EEG窗的标准化空间协方差矩阵
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

filters = ()
iterator = range(0, len(task)) # 类别个数

for x in iterator:
    Rx = covarianceMatrix(task[x][0])
    for t in range(1, len(task[x])):
        Rx += covarianceMatrix(task[x][t])
    Rx = Rx / len(task[x]) # 获得某一个类别的标准化协方差矩阵Rx
    
    count = 0
    not_Rx = Rx * 0
    for not_x in [element for element in iterator if element != x]:
        for t in range(0, len(task[not_x])):
            not_Rx += covarianceMatrix(task[not_x][t])
            count += 1
    not_Rx = not_Rx / count # 获得其它类别的标准化协方差矩阵not_Rx
    
    # 计算空间滤波器
    R = Rx + not_Rx # 不同类别的复合空间协方差矩阵
    E,U = la.eig(R) # 获取复合空间协方差矩阵的特征值E和特征向量U
    
    order = np.argsort(E) # 升序排序
    order = order[::-1] # 翻转以使特征值降序排序
    E = E[order] 
    U = U[:,order]
    
    P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U)) # 获取白化变换矩阵
    # 获取白化变换后的协方差矩阵
    Sa = np.dot(P,np.dot(Rx,np.transpose(P))) 
    Sb = np.dot(P,np.dot(not_Rx,np.transpose(P)))
    
    E1,U1 = la.eig(Sa)
    E2,U2 = la.eig(Sb)
    # 至此有np.diag(E1)+np.diag(E2)=I以及U1=U2
     
    # 求得矩阵W,其列向量即CSP滤波器
    W = np.transpose(np.dot(np.transpose(U1),P))
    
    filters += (W,)
    # 二分类时两个滤波器矩阵相同