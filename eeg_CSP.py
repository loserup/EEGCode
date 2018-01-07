# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:07:25 2017

@author: SingleLong

% 说明：

% 第四步

% 通过CSP求区别有无意图的投影矩阵
"""

import scipy.io as sio
import numpy as np
import scipy.linalg as la # 线性代数库
import os

id_subject = 3 # 【受试者的编号】

fileNum = 0
def visitDir(path):
    global fileNum
    for lists in os.listdir(path):
        sub_path = os.path.join(path, lists)
        print(sub_path)
        if os.path.isfile(sub_path):
            fileNum = fileNum+1 # 统计文件数量
    return fileNum
if id_subject < 10:
    num_file = visitDir('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_0'+str(id_subject)+'_wineeg') # 获取受试对象EEG窗的数量
else:
    num_file = visitDir('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_'+str(id_subject)+'_wineeg')

eegwin_0 = [] # 存放标记为0的EEG窗
eegwin_1 = [] # 存放标记为1的EEG窗
for i in range(int(num_file/2)):
    if id_subject < 10:
        label0_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_0'+str(id_subject)+'_wineeg\\label0_Subject_0'+str(id_subject)+'_'+str(i+1)+'.mat')
        label1_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_0'+str(id_subject)+'_wineeg\\label1_Subject_0'+str(id_subject)+'_'+str(i+1)+'.mat')
        eegwin_0.append(label0_mat['label0_Subject_0'+str(id_subject)+'_'+str(i+1)][0][0])
        eegwin_1.append(label1_mat['label1_Subject_0'+str(id_subject)+'_'+str(i+1)][0][0])
    else:
        label0_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_'+str(id_subject)+'_wineeg\\label0_Subject_'+str(id_subject)+'_'+str(i+1)+'.mat')
        label1_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_'+str(id_subject)+'_wineeg\\label1_Subject_'+str(id_subject)+'_'+str(i+1)+'.mat')
        eegwin_0.append(label0_mat['label0_Subject_'+str(id_subject)+'_'+str(i+1)][0][0])
        eegwin_1.append(label1_mat['label1_Subject_'+str(id_subject)+'_'+str(i+1)][0][0])
        
task = (eegwin_0, eegwin_1)

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

num_pair = 6 # 【从CSP投影矩阵filters里取得特征对数】
output = np.zeros([num_pair*2,np.shape(filters[0])[0]]) # 提取特征的投影矩阵
output[0:num_pair,:] = filters[0][0:num_pair,:] # 取投影矩阵前几行
output[num_pair:,:] = filters[0][np.shape(filters[0])[1]-num_pair:,:] # 对应取投影矩阵后几行

if id_subject < 10:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_0'+str(id_subject)+'_CSP\\Subject_0'+str(id_subject)+'_CSP.mat', {'Subject_0'+str(id_subject)+'_CSP':output})
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_0'+str(id_subject)+'_CSP\\Subject_0'+str(id_subject)+'_label0_win.mat', {'Subject_0'+str(id_subject)+'_label0_win':eegwin_0})
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_0'+str(id_subject)+'_CSP\\Subject_0'+str(id_subject)+'_label1_win.mat', {'Subject_0'+str(id_subject)+'_label1_win':eegwin_1})
else:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_'+str(id_subject)+'_CSP\\Subject_'+str(id_subject)+'_CSP.mat', {'Subject_'+str(id_subject)+'_CSP':output})
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_'+str(id_subject)+'_CSP\\Subject_'+str(id_subject)+'_label0_win.mat', {'Subject_'+str(id_subject)+'_label0_win':eegwin_0})
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_'+str(id_subject)+'_CSP\\Subject_'+str(id_subject)+'_label1_win.mat', {'Subject_'+str(id_subject)+'_label1_win':eegwin_1})
    
