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

id_subject = 1 # 【受试者的编号】
num_pair = 4 # 【从CSP投影矩阵里取得特征对数】

if id_subject < 10:
    input_eegwin_dict = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0'+\
                                    str(id_subject)+'_Data\\Subject_0'+\
                                    str(id_subject)+'_WinEEG.mat')
else:
    input_eegwin_dict = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_'+\
                                    str(id_subject)+'_Data\\Subject_'+\
                                    str(id_subject)+'_WinEEG.mat')

input_eegwin = input_eegwin_dict['WinEEG']

eegwin_0 = [] # 存放标记为-1的EEG窗
eegwin_1 = [] # 存放标记为1的EEG窗
for i in range(len(input_eegwin)):
    if int(input_eegwin[i][1]) == -1:
        # 若EEG窗标记为0
        eegwin_0.append(input_eegwin[i][0])
    elif int(input_eegwin[i][1]) == 1:
        eegwin_1.append(input_eegwin[i][0])
        
task = (eegwin_0, eegwin_1)

# 获取EEG窗的标准化空间协方差矩阵
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

### CSP算法
filters = ()
C_0 = covarianceMatrix(task[0][0])
for i in range(1,len(task[0])):
    C_0 += covarianceMatrix(task[0][i])
C_0 = C_0 / len(task[0]) # 获得标记为0的EEG窗的标准化协方差对称矩阵

C_1 = 0 * C_0 # 用C_1 = np.empty(C_0.shape)有些极小的随机非0数，会导致输出结果每次都会改变
for i in range(0,len(task[1])):
    C_1 += covarianceMatrix(task[1][i])
C_1 = C_1 / len(task[1]) # 获得标记为1的EEG窗的标准化协方差对称矩阵

C = C_0 + C_1 # 不同类别的复合空间协方差矩阵,这是一个对称矩阵
E,U = la.eig(C) # 获取复合空间协方差矩阵的特征值E和特征向量U,这里C可以分解为C=np.dot(U,np.dot(np.diag(E),U.T))
#E = E.real # E取实部；取实部后不能实现np.diag(E_0)+np.diag(E_1)=I

order = np.argsort(E) # 升序排序
order = order[::-1] # 翻转以使特征值降序排序
E = E[order] 
U = U[:,order]

P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U)) # 获取白化变换矩阵

# 获取白化变换后的协方差矩阵
S_0 = np.dot(P,np.dot(C_0, np.transpose(P)))
S_1 = np.dot(P,np.dot(C_1, np.transpose(P)))

E_0,U_0 = la.eig(S_0)
# 至此有np.diag(E_0)+np.diag(E_1)=I以及U_0=U_1

# 这里特征值也要按降序排序
order = np.argsort(E_0)
order = order[::-1]
E_0 = E_0[order]
U_0 = U_0[:,order]

#E_1,U_1 = la.eig(S_1);E_1 = E_1[order];U_1 = U_1[:,order] #测试是否满足np.diag(E_0)+np.diag(E_1)=I

# 求得CSP投影矩阵W
W = np.dot(np.transpose(U_0),P)

csp = np.zeros([num_pair*2,np.shape(W)[0]]) # 提取特征的投影矩阵
csp[0:num_pair,:] = W[0:num_pair,:] # 取投影矩阵前几行
csp[num_pair:,:] = W[np.shape(W)[1]-num_pair:,:] # 对应取投影矩阵后几行

# 利用投影矩阵提取EEG窗特征
features = []
for i in range(len(eegwin_0)):
    Z = np.dot(csp, eegwin_0[i])
    varances = list(np.log(np.var(Z, axis=1))) # axis=1即求每行的方差
    varances = [np.log(x/sum(varances)) for x in varances] # 方差标准化
    varances.append(-1)
    features.append(varances)

for i in range(len(eegwin_1)):  
    Z = np.dot(csp, eegwin_1[i])
    varances = list(np.log(np.var(Z, axis=1)))
    varances = [np.log(x/sum(varances)) for x in varances]
    varances.append(1)
    features.append(varances)
    
if id_subject < 10:
    sio.savemat('E:\\EEGExoskeleton\\Data\\Subject_0'+str(id_subject)+\
                '_Data\\Subject_0'+str(id_subject)+'_features.mat',\
                {'features' : features})
    sio.savemat('E:\\EEGExoskeleton\\Data\\Subject_0'+str(id_subject)+\
                '_Data\\Subject_0'+str(id_subject)+'_csp.mat',\
                {'csp' : csp})
else:
    sio.savemat('E:\\EEGExoskeleton\\Data\\Subject_'+str(id_subject)+\
                '_Data\\Subject_'+str(id_subject)+'_features.mat',\
                {'features' : features})
    sio.savemat('E:\\EEGExoskeleton\\Data\\Subject_'+str(id_subject)+\
                '_Data\\Subject_'+str(id_subject)+'_csp.mat',\
                {'csp' : csp})