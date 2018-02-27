# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:32:50 2017

@author: SingleLong

% 说明：

% 第二步

% 将两次打标之间的EEG从原始EEG信号中提取出来
% 根据两次打标位置截取原始步态数据并对截取步态数据进行低通滤波
"""

import scipy.io as sio
import numpy as np
import scipy.signal as sis

id_subject = 2 # 【受试者的编号】

if id_subject < 10:
    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0' +\
                               str(id_subject) + '_Data\\Subject_0' +\
                               str(id_subject) + '_RawEEG.mat')
    gait_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0' +\
                                str(id_subject) + '_Data\\Subject_0' +\
                                str(id_subject) + '_RawMotion.mat')
else:
    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_' +\
                               str(id_subject) + '_Data\\Subject_' +\
                               str(id_subject) + '_RawEEG.mat')
    gait_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_' +\
                                str(id_subject) + '_Data\\Subject_' +\
                                str(id_subject) + '_RawMotion.mat')

eeg_data = eeg_mat_data['rawEEG']
gait_data = gait_mat_data['rawMotion']

num_trial = np.shape(eeg_data)[1] # 获取受试者进行试验的次数

# 对动作信号【低通滤波】
fs = 121 # 采样频率121Hz
Wn = 0.9 # 截止频率0.9Hz
def lowpass(data, num_sample):
    b,a = sis.butter(4, 2*Wn/fs, 'lowpass')
    
    filtered_data = np.zeros(num_sample)
    filtered_data = sis.filtfilt(b,a,data) 
    
    return filtered_data

# 找两次打标位置
for i in range(num_trial):
    label_index = 0 # 打标位置

    # 通过上升沿找两次打标位置
    temp = eeg_data[0][i][32][0]
    test_temp = eeg_data[0][i][32] # 用来检验EEG信号是否有效
    for data_label in eeg_data[0][i][32]:
        if data_label <= temp:
            label_index += 1
            temp = data_label
            continue
        else:
            label_index += 1
            temp = data_label
            break
        
    label_index_1 = label_index # 第一次打标位置
    
    for data_label in eeg_data[0][i][32][label_index_1:]:
        if data_label <= temp:
            label_index += 1
            temp = data_label
            continue
        else:
            label_index += 1
            break
    
    label_index_2 = label_index # 第二次打标位置
    
    # 截取两次打标之间的数据
    eeg_data[0][i] = eeg_data[0][i][0:32, label_index_1:label_index_2]
    
    # 截取步态数据并低通滤波
    if label_index_2 == len(test_temp):
        # 如果该次试验的EEG数据无效
        # 无效数据为没有打标或者只打了一次标
        gait_data[0][i] = []
    else:
        gait_temp = gait_data[0][i].T
        num_sample = len(gait_temp[0])
        gait_end_index = round((label_index_2 - label_index_1) * fs / 512) # 第二次打标对应步态数据的位置
        
        r_pass = lowpass(gait_temp[0],num_sample)[:gait_end_index] # 右膝数据低通滤波
        l_pass = lowpass(gait_temp[1],num_sample)[:gait_end_index] # 左膝数据低通滤波

        gait_data[0][i] = (gait_data[0][i].T)[:,:gait_end_index]
        gait_data[0][i][0] = r_pass
        gait_data[0][i][1] = l_pass


if id_subject < 10:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0'+str(id_subject)+\
                '_Data\\Subject_0' + str(id_subject) +'_CutedEEG.mat',\
                {'CutedEEG' : eeg_data})
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0'+str(id_subject)+\
                '_Data\\Subject_0' + str(id_subject) +'_FilteredMotion.mat',\
                {'FilteredMotion' : gait_data})
else:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+str(id_subject)+\
                '_Data\\Subject_' + str(id_subject) +'_CutedEEG.mat',\
                {'CutedEEG' : eeg_data})
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+str(id_subject)+\
                '_Data\\Subject_' + str(id_subject) +'_FilteredMotion.mat',\
                {'FilteredMotion' : gait_data})