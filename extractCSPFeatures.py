# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 20:28:20 2018

@author: Long

% 说明：

% 第五步

% 通过CSP投影矩阵提取EEG窗方差特征
"""

import scipy.io as sio
import numpy as np

id_subject = 3 # 【受试者的编号】

if id_subject < 10:
    CSP_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_0'+\
                          str(id_subject)+'_CSP\\Subject_0'+str(id_subject)+\
                          '_CSP.mat')
    csp = CSP_mat['Subject_0'+str(id_subject)+'_CSP']
    eegwin_0_mat =  sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_0'+\
                          str(id_subject)+'_CSP\\Subject_0'+str(id_subject)+\
                          '_label0_win.mat')
    eegwin_0 = eegwin_0_mat['Subject_0'+str(id_subject)+'_label0_win']
    num_file = np.shape(eegwin_0)[0] # 获取受试对象EEG窗的数量
    eegwin_1_mat =  sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_0'+\
                          str(id_subject)+'_CSP\\Subject_0'+str(id_subject)+\
                          '_label1_win.mat')
    eegwin_1 = eegwin_1_mat['Subject_0'+str(id_subject)+'_label1_win']
else:
    CSP_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_'+\
                          str(id_subject)+'_CSP\\Subject_'+str(id_subject)+\
                          '_CSP.mat')
    csp = CSP_mat['Subject_'+str(id_subject)+'_CSP']
    eegwin_0_mat =  sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_'+\
                          str(id_subject)+'_CSP\\Subject_'+str(id_subject)+\
                          '_label0_win.mat')
    eegwin_0 = eegwin_0_mat['Subject_'+str(id_subject)+'_label0_win']
    num_file = np.shape(eegwin_0)[0] # 获取受试对象EEG窗的数量
    eegwin_1_mat =  sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_'+\
                          str(id_subject)+'_CSP\\Subject_'+str(id_subject)+\
                          '_label1_win.mat')
    eegwin_1 = eegwin_1_mat['Subject_'+str(id_subject)+'_label1_win']

output = []
for i in range(num_file):
    z_0 = np.dot(csp, eegwin_0[i])
    z_1 = np.dot(csp, eegwin_1[i])
    varances_0 = list(np.log(np.var(z_0, axis=1))) # axis=1即求每行的方差
    varances_1 = list(np.log(np.var(z_1, axis=1)))
    varances_0.append(0)
    varances_1.append(1)
    output.append(varances_0)
    output.append(varances_1)

if id_subject < 10:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_0'+str(id_subject)+\
                '_feature\\Subject_0'+str(id_subject)+'_features',\
                {'features':output})
else:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor2\\Subject_'+str(id_subject)+\
                '_feature\\Subject_'+str(id_subject)+'_features',\
                {'features':output})

    