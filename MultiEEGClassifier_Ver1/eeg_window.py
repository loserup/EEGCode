# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:25:28 2017

@author: SingleLong

% 说明：

% 第三步

% 根据步态信号建立划窗
% 划窗截取EEG信号
% 生成指定受试对象的有意图和无意图区域的EEG窗
% 标签类别：无意图窗-0，跨越障碍高1-1，跨越障碍高3-2，跨越障碍高5-3
"""

import scipy.io as sio
import numpy as np
import scipy.signal as sis
import matplotlib.pyplot as plt


id_subject = 3 # 【受试者的编号】
work_trial = 18 # 【设置有效的极值点数】即跨越时的极值点

if id_subject < 10:
    gait_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0' +\
                                str(id_subject) + '_Data\\Subject_0' +\
                                str(id_subject) + '_FilteredMotion.mat')
    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0' +\
                               str(id_subject) + '_Data\\Subject_0' +\
                               str(id_subject) + '_CutedEEG.mat')
else:
    gait_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_' +\
                                str(id_subject) + '_Data\\Subject_' +\
                                str(id_subject) + '_FilteredMotion.mat')
    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_' +\
                               str(id_subject) + '_Data\\Subject_' +\
                               str(id_subject) + '_CutedEEG.mat')

gait_data = gait_mat_data['FilteredMotion'][0]
eeg_data = eeg_mat_data['CutedEEG']

num_trial = len(gait_data) # 获取受试者进行试验的次数

# 绘图-测试用
def Window_plotor(num_axis, data, peakind_sorted):
    data_axis = [i for i in range(num_axis)]
    plt.figure(figsize=[15,4])
    plt.plot(data_axis, data)
    for i in peakind_sorted:
        plt.scatter(i, data[i])

# 找步态数据中的极大值
def find_peak_point(dataset):
    peakind = [] # 存放极大值的索引
    index = 0 
    for data in dataset:
        if index != 0 and index != len(dataset)-1:
            if data >= dataset[index-1] and data >= dataset[index+1]:
                peakind.append(index)
                index += 1
            else:
                index += 1
                continue
        else:
            index += 1
            continue
    return peakind

# 找步态数据中跨越障碍极大值点前的极小值
def find_valley_point(dataset, peakind_sorted):
    valleyind = [] # 存放极小值的索引
    index = 0 
    for data in dataset:
        if index != 0 and index != len(dataset)-1:
            if data <= dataset[index-1] and data <= dataset[index+1]:
                valleyind.append(index)
                index += 1
            else:
                index += 1
                continue
        else:
            index += 1
            continue
    
    valleyind_sorted = [] # 存放跨越前的极小值
    for peak in peakind_sorted:
        index = 0
        for valley in valleyind:
            if valleyind[index+1] > peak:
                valleyind_sorted.append(valley)
                break # 找到这个极大值点前的极值点即可开始找下一个极大值点的极小值点了
            else:
                index += 1
                continue
            
    return valleyind_sorted

# 对EEG信号带通滤波
fs = 512 # 【采样频率512Hz】
bias_0 = 300 #【无意图窗偏移量】
bias_1 = -300 #【有意图窗偏移量】
win_width = 350 # 【窗宽度】
def bandpass(data,upper=1,lower=4):
    Wn = [2 * upper / fs, 2 * lower / fs] # 截止频带0.1-1Hz or 8-30Hz
    b,a = sis.butter(4, Wn, 'bandpass')
    
    filtered_data = np.zeros([32, win_width])
    for row in range(32):
        filtered_data[row] = sis.filtfilt(b,a,data[row,:]) 
    
    return filtered_data 

out_count = 0 # 输出文件批数
output = []
for i in range(num_trial):
    if len(gait_data[i]) and (i != 1): # 受试对象3的第二次trial效果不好，故去掉
        # 当步态数据不是空集时（有效时）
        peakind = find_peak_point(gait_data[i][0])
        peak = [gait_data[i][0][j] for j in peakind] # 获取极值点
        peak_sorted = sorted(peak, reverse=True) # 将极值点降序排序
        peakind_sorted = [] # 对应降序排序极值点的索引
        for j in peak_sorted[:work_trial]:
            peakind_sorted.append(list(gait_data[i][0]).index(j))
        peakind_sorted = np.array(sorted(peakind_sorted))
        
        num_axis = len(gait_data[i][0])
        #Window_plotor(num_axis, gait_data[i][0], peakind_sorted); plt.title(str(i+1) + 'th trial\'s peak points') # 测试绘图，观察跨越极大值点位置是否找对
        
        valleyind_sorted = np.array(find_valley_point(gait_data[i][0], peakind_sorted)) # 跨越前的极小值点
        #Window_plotor(num_axis, gait_data[i][0], valleyind_sorted); plt.title(str(i+1) + 'th trial\'s valley points') # 测试绘图，观察跨越前极小值点位置是否找对

        # 取无跨越意图EEG窗，标记为0   
        win_index = peakind_sorted + bias_0 # 窗起始索引
        win_index = win_index * 512 / fs_gait
        
        # 截取4-7,8-13,13-30三个频带的EEG窗
        for k in range(work_trial):
            out_eeg =  eeg_data[0][i][:,int(win_index[k]):(int(win_index[k])+win_width)]
            out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
            out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
            out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
            out_eeg = [np.hstack((out_eeg_band1,out_eeg_band2,out_eeg_band3)),0]
            output.append(out_eeg)
                
        # 取有跨越意图EEG窗，标记为1,2,3
        win_index = valleyind_sorted + bias_1
        win_index = win_index * 512 / fs_gait
        
        for k in range(work_trial):
            out_eeg =  eeg_data[0][i][:,int(win_index[k]-win_width):int(win_index[k])]
            out_eeg = bandpass(out_eeg)
            
            out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
            out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
            out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
            # 跨越三种障碍的标签，现在手动打标签
            # 18次跨越的标签排序为2,1,3,3,1,2,2,1,3,3,1,2,2,1,3,3,1,2
            if k % 6 == 1 or k%6 == 4:
                out_eeg = [np.hstack((out_eeg_band1,out_eeg_band2,out_eeg_band3)),1] # 跨越障碍高1
            elif k % 6 == 0 or k%6 == 5:
                out_eeg = [np.hstack((out_eeg_band1,out_eeg_band2,out_eeg_band3)),2] # 跨越障碍高3
            else:
                out_eeg = [np.hstack((out_eeg_band1,out_eeg_band2,out_eeg_band3)),3] # 跨越障碍高5     
            output.append(out_eeg)
                
        out_count += 1
    else:
        continue
    
if id_subject < 10:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0'+str(id_subject)+\
                '_Data\\Subject_0'+str(id_subject)+'_WinEEG_4class.mat',\
                {'WinEEG':output})
else:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+str(id_subject)+\
                '_Data\\Subject_'+str(id_subject)+'_WinEEG_4class.mat',\
                {'WinEEG':output})
