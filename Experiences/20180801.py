# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 20:04:09 2018

画图，看看直接从特征能看出和步态的对应关系

@author: Long
"""

import scipy.io as sio
import numpy as np
import scipy.signal as sis
import matplotlib.pyplot as plt

id_subject = 1 # 【受试者的编号】

if id_subject < 10:
    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0' +\
                               str(id_subject) + '_Data\\Subject_0' +\
                               str(id_subject) + '_CutedEEG.mat')['CutedEEG']
    gait_mat_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0' +\
                                str(id_subject) + '_Data\\Subject_0' +\
                                str(id_subject) + '_FilteredMotion.mat')['FilteredMotion']
    csp = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0' +\
                      str(id_subject) + '_Data\\Subject_0' +\
                      str(id_subject) + '_csp.mat')['csp']
else:
    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_' +\
                               str(id_subject) + '_Data\\Subject_' +\
                               str(id_subject) + '_CutedEEG.mat')['CutedEEG']
    gait_mat_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_' +\
                                str(id_subject) + '_Data\\Subject_' +\
                                str(id_subject) + '_FilteredMotion.mat')['FilteredMotion']
    csp = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_' +\
                      str(id_subject) + '_Data\\Subject_' +\
                      str(id_subject) + '_csp.mat')['csp']

A = 1 # id_trail
eeg = eeg_mat_data[0][A]
gait = gait_mat_data[0][A][0]


test_feat_all = [] # 记录喂给分类器的特征

fs = 512 # 【采样频率512Hz】
win_width = 384 # 【窗宽度】384对应750ms窗长度
def bandpass(data,upper,lower):
    Wn = [2 * upper / fs, 2 * lower / fs] # 截止频带0.1-1Hz or 8-30Hz
    b,a = sis.butter(4, Wn, 'bandpass')
    
    filtered_data = np.zeros([32, win_width])
    for row in range(32):
        filtered_data[row] = sis.filtfilt(b,a,data[row,:]) 
    
    return filtered_data 

    
for i in range(np.shape(eeg)[1]):
    if i < win_width: # 初始阶段没有完整的750ms窗，384对应750ms窗长度
        continue 
    elif i % 26 != 0: # 每隔50ms取一次窗
        continue
    else:
        test_eeg = eeg[:,(i-win_width):i]
        out_eeg_band0 = bandpass(test_eeg,upper=0.3,lower=3)
        out_eeg_band1 = bandpass(test_eeg,upper=4,lower=7)
        out_eeg_band2 = bandpass(test_eeg,upper=8,lower=13)
        out_eeg_band3 = bandpass(test_eeg,upper=13,lower=30)
        test_eeg = np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3))
        Z = np.dot(csp, test_eeg)
        varances = list(np.var(Z, axis=1))
        test_feat = np.array([np.log(x/sum(varances)) for x in varances]) # 标准化
        test_feat = test_feat.reshape(1,len(csp)) # classifier.predict需要和fit时相同的数据结构，所以要reshape
        test_feat_all.append(test_feat)

feat = np.array(test_feat_all).reshape(len(test_feat_all),len(csp))

plt.figure(figsize=[15,8]) 
axis = [j for j in range(len(feat))]
plt.subplot(311)
plt.plot(axis, feat)

plt.figure(figsize=[15,8]) 
axis = [j for j in range(len(gait))]
plt.subplot(312)
plt.plot(axis, gait)

feat_rms = []
for i in range(len(feat)):
    ssum = 0
    # 计算每个特征的均方根RMS
#    for j in range(len(feat[i])):
#        ssum = ssum + feat[i][j]**2
#    feat_rms.append(np.sqrt(ssum/len(feat[i])))
    feat_rms.append(feat[i][0]**2)

plt.figure(figsize=[15,8]) 
axis = [j for j in range(len(feat_rms))]
plt.subplot(313)
plt.plot(axis, feat_rms)


for j in range(8):
    feat_rms = []
    for i in range(len(feat)):
#        ssum = 0
    # 计算每个特征的均方根RMS
#    for j in range(len(feat[i])):
#        ssum = ssum + feat[i][j]**2
#    feat_rms.append(np.sqrt(ssum/len(feat[i])))
        feat_rms.append(feat[i][j]**2)
        
    plt.figure(figsize=[15,4]) 
    axis = [j for j in range(len(feat_rms))]
    plt.subplot(211)
    plt.plot(axis, feat_rms)
    
    plt.figure(figsize=[15,4])
    axis = [j for j in range(len(gait))]
    plt.subplot(212)
    plt.plot(axis, gait)
    



