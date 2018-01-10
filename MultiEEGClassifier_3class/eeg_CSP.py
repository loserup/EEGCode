# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:07:25 2017

@author: SingleLong

% 说明：

% 第四步

% 通过CSP求区别有无意图的投影矩阵
% 并通过CSP投影矩阵提取EEG窗方差特征
"""

import scipy.io as sio
import numpy as np
import scipy.linalg as la # 线性代数库

id_subject = 3 # 【受试者的编号】
num_pair = 6 # 【从CSP投影矩阵里取得特征对数】

if id_subject < 10:
    input_eegwin_dict = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0'+\
                                    str(id_subject)+'_Data\\Subject_0'+\
                                    str(id_subject)+'_WinEEG_4class.mat')
else:
    input_eegwin_dict = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+\
                                    str(id_subject)+'_Data\\Subject_'+\
                                    str(id_subject)+'_WinEEG_4class.mat')

input_eegwin = input_eegwin_dict['WinEEG']


eegwin_1 = [] # 存放标记为1的EEG窗
eegwin_2 = [] # 存放标记为2的EEG窗
eegwin_3 = [] # 存放标记为3的EEG窗
for i in range(len(input_eegwin)):
    if int(input_eegwin[i][1]) == 1:
        eegwin_1.append(input_eegwin[i][0])
    elif int(input_eegwin[i][1]) == 2:
        eegwin_2.append(input_eegwin[i][0])
    else:
        eegwin_3.append(input_eegwin[i][0])
        
task = (eegwin_1, eegwin_2, eegwin_3)

# 获取EEG窗的标准化空间协方差矩阵
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

### 多分类CSP算法：求一个类别和其它类别总和的区别最大的投影矩阵
filters = ()
iterator = range(0, len(task)) # 类别个数

for x in iterator:
    C_0 = covarianceMatrix(task[x][0])
    for t in range(1, len(task[x])):
        C_0 += covarianceMatrix(task[x][t])
    C_0 = C_0 / len(task[x]) # 获得某一个类别的标准化协方差矩阵Rx
    
    count = 0
    not_C_0 = C_0 * 0
    for not_x in [element for element in iterator if element != x]:
        for t in range(0, len(task[not_x])):
            not_C_0 += covarianceMatrix(task[not_x][t])
            count += 1
    not_C_0 = not_C_0 / count # 获得其它类别的标准化协方差矩阵not_Rx
    
    # 计算空间滤波器
    C = C_0 + not_C_0 # 不同类别的复合空间协方差矩阵
    E,U = la.eig(C) # 获取复合空间协方差矩阵的特征值E和特征向量U
    
    order = np.argsort(E) # 升序排序
    order = order[::-1] # 翻转以使特征值降序排序
    E = (E[order]).real 
    U = (U[:,order]).real
    
    P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U)) # 获取白化变换矩阵
    # 获取白化变换后的协方差矩阵
    S_0 = np.dot(P,np.dot(C_0,np.transpose(P))) 
    not_S_0 = np.dot(P,np.dot(not_C_0,np.transpose(P)))
    
    E_0,U_0 = la.eig(S_0)
    not_E_0,not_U_0 = la.eig(not_S_0)
    # 至此有np.diag(E1)+np.diag(E2)=I以及U1=U2
    
    order = np.argsort(E_0) # 升序排序
    order = order[::-1] # 翻转以使特征值降序排序
    E_0 = (E_0[order]).real
    U_0 = (U_0[:,order]).real
    
    #not_E_0 = (not_E_0[order]).real; not_U_0 = (not_U_0[:,order]).real # 测试是否满足np.diag(E_0)+np.diag(not_E_0)=I和U_0=not_U_0
     
    # 求得矩阵W,其列向量即CSP滤波器
    W = np.dot(np.transpose(U_0),P)
    
    filters += (W,)
    # 二分类时两个滤波器矩阵相同

csp = [] # 提取特征的投影矩阵
for i in range(len(task)):
    temp = np.zeros([num_pair*2,np.shape(W)[0]]) 
    temp[0:num_pair,:] = filters[i][0:num_pair,:] # 取投影矩阵前几行
    temp[num_pair:,:] = filters[i][np.shape(W)[1]-num_pair:,:] # 对应取投影矩阵后几行
    csp.append(temp)

# 利用投影矩阵提取EEG窗特征
features = []
for i in range(len(eegwin_1)):
    Z = np.dot(csp[0], eegwin_1[i])
    varances = list(np.log(np.var(Z, axis=1)))
    varances.append(1)
    features.append(varances)
for i in range(len(eegwin_2)):
    Z = np.dot(csp[1], eegwin_2[i])
    varances = list(np.log(np.var(Z, axis=1)))
    varances.append(2)
    features.append(varances)
for i in range(len(eegwin_3)):
    Z = np.dot(csp[2], eegwin_3[i])
    varances = list(np.log(np.var(Z, axis=1)))
    varances.append(3)
    features.append(varances)
    
if id_subject < 10:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0'+str(id_subject)+\
                '_Data\\Subject_0'+str(id_subject)+'_features_4class.mat',\
                {'features' : features})
else:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+str(id_subject)+\
                '_Data\\Subject_'+str(id_subject)+'_features_4class.mat',\
                {'features' : features})