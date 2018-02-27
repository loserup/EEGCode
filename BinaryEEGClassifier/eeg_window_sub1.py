# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:25:28 2017

@author: SingleLong

% 说明：

% 第三步

% 根据步态信号建立划窗
% 划窗截取EEG信号
% 生成指定受试对象的有意图和无意图区域的EEG窗

% 专门针对第一个受试对象的划窗函数
受试对象1共进行了20次trail
第1次：打标失败
第2次：往返1次
第3次：往返1次
第4次：往返1次
第5次：往返1次
第6次：打标失败
第7次：往返1次
第8次：往返2次
第9次：往返2次
第10次：往返2次
第11次：往返2次
第12次：往返2次
第13次：往返3次
第14次：往返3次；数据不好处理，去掉
第15次：往返3次
第16次：打标失败
第17次：往返3次
第18次：往返3次；数据不好处理，去掉
第19次：往返3次
第20次：往返3次；数据不好处理，去掉
备注：经测试，受试对象基本为右腿跨越，偶有左腿跨越
"""

import scipy.io as sio
import numpy as np
import scipy.signal as sis
import matplotlib.pyplot as plt


id_subject = 1 # 【受试者的编号】
work_trial_1 = 6 # 往返1次的跨越次数
work_trial_2 = 12 # 往返2次的跨越次数
work_trial_3 = 18 # 往返3次的跨越次数


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
    
    valleyind_sorted = [] # 存放跨越前的极小值索引
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
fs_gait = 121 # 【步态数据采样频率121Hz】
def bandpass(data,upper,lower):
    Wn = [2 * upper / fs, 2 * lower / fs] # 截止频带0.1-1Hz or 8-30Hz
    b,a = sis.butter(4, Wn, 'bandpass')
    
    filtered_data = np.zeros([32, win_width])
    for row in range(32):
        filtered_data[row] = sis.filtfilt(b,a,data[row,:]) 
    
    return filtered_data 

out_count = 0 # 输出文件批数
output = []
for i in range(num_trial):
    if len(gait_data[i]) and i>=0 and i<=6: # 前7次往返1次
        # 当步态数据不是空集时（有效时）   
        # 取右膝跨越极值点索引
        r_peakind = find_peak_point(gait_data[i][0])
        r_peak = [gait_data[i][0][j] for j in r_peakind] # 获取极值点
        r_peak_sorted = sorted(r_peak, reverse=True) # 将极值点降序排序
        r_peakind_sorted = [] # 对应降序排序极值点的索引
        for j in r_peak_sorted[:work_trial_1]:
            r_peakind_sorted.append(list(gait_data[i][0]).index(j))
        r_peakind_sorted = np.array(sorted(r_peakind_sorted))
        
        # 取左膝跨越极值点索引
        l_peakind = find_peak_point(gait_data[i][1])
        l_peak = [gait_data[i][1][j] for j in l_peakind] # 获取极值点
        l_peak_sorted = sorted(l_peak, reverse=True) # 将极值点降序排序
        l_peakind_sorted = [] # 对应降序排序极值点的索引
        for j in l_peak_sorted[:work_trial_1]:
            l_peakind_sorted.append(list(gait_data[i][1]).index(j))
        l_peakind_sorted = np.array(sorted(l_peakind_sorted))
        
        r_valleyind_sorted = np.array(find_valley_point(gait_data[i][0], r_peakind_sorted)) # 右膝跨越前的极小值点
        l_valleyind_sorted = np.array(find_valley_point(gait_data[i][1], l_peakind_sorted)) # 左膝跨越前的极小值点
        num_axis = len(gait_data[i][0])
       
        #Window_plotor(num_axis, gait_data[i][0], r_peakind_sorted); plt.title(str(i+1) + 'th trial\'s peak points') # 测试绘图，观察跨越极大值点位置是否找对
        #Window_plotor(num_axis, gait_data[i][0], r_valleyind_sorted); plt.title(str(i+1) + 'th trial\'s valley points') # 测试绘图，观察跨越前极小值点位置是否找对

        # 取无跨越意图EEG窗，标记为0   
        rp_win_index = r_peakind_sorted + bias_0 # 窗起始索引
        rp_win_index = rp_win_index * 512 / fs_gait
        lp_win_index = l_peakind_sorted + bias_0 # 窗起始索引
        lp_win_index = lp_win_index * 512 / fs_gait
        # 取有跨越意图EEG窗，标记为1
        rv_win_index = r_valleyind_sorted + bias_1
        rv_win_index = rv_win_index * 512 / fs_gait
        lv_win_index = l_valleyind_sorted + bias_1
        lv_win_index = lv_win_index * 512 / fs_gait
        
        for k in range(work_trial_1):
            if r_peakind_sorted[k] < l_peakind_sorted[k]:
                # 先跨右腿
                #print('r') # 测试用，观察跨越用的腿是否一致
                # 无跨越意图窗
                out_eeg = eeg_data[0][i][:,int(rp_win_index[k]):(int(rp_win_index[k])+win_width)]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),0]
                output.append(out_eeg)
                # 有跨越意图窗
                out_eeg =  eeg_data[0][i][:,int(rv_win_index[k]-win_width):int(rv_win_index[k])]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),1]
                output.append(out_eeg)
            else:
                #print('l') # 测试用，观察跨越用的腿是否一致
                # 无跨越意图窗
                out_eeg = eeg_data[0][i][:,int(lp_win_index[k]):(int(lp_win_index[k])+win_width)]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),0]
                output.append(out_eeg)
                # 有跨越意图窗
                out_eeg =  eeg_data[0][i][:,int(lv_win_index[k]-win_width):int(lv_win_index[k])]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),1]
                output.append(out_eeg)
                     
        out_count += 1
        
    elif len(gait_data[i]) and i>=7 and i<=11: # 第8到12次往返2次
        # 当步态数据不是空集时（有效时）   
        # 取右膝跨越极值点索引
        r_peakind = find_peak_point(gait_data[i][0])
        r_peak = [gait_data[i][0][j] for j in r_peakind] # 获取极值点
        r_peak_sorted = sorted(r_peak, reverse=True) # 将极值点降序排序
        r_peakind_sorted = [] # 对应降序排序极值点的索引
        for j in r_peak_sorted[:work_trial_2]:
            r_peakind_sorted.append(list(gait_data[i][0]).index(j))
        r_peakind_sorted = np.array(sorted(r_peakind_sorted))
        
        # 取左膝跨越极值点索引
        l_peakind = find_peak_point(gait_data[i][1])
        l_peak = [gait_data[i][1][j] for j in l_peakind] # 获取极值点
        l_peak_sorted = sorted(l_peak, reverse=True) # 将极值点降序排序
        l_peakind_sorted = [] # 对应降序排序极值点的索引
        for j in l_peak_sorted[:work_trial_2]:
            l_peakind_sorted.append(list(gait_data[i][1]).index(j))
        l_peakind_sorted = np.array(sorted(l_peakind_sorted))
        
        r_valleyind_sorted = np.array(find_valley_point(gait_data[i][0], r_peakind_sorted)) # 右膝跨越前的极小值点
        l_valleyind_sorted = np.array(find_valley_point(gait_data[i][1], l_peakind_sorted)) # 左膝跨越前的极小值点
        num_axis = len(gait_data[i][0])
       
        #Window_plotor(num_axis, gait_data[i][0], r_peakind_sorted); plt.title(str(i+1) + 'th trial\'s peak points') # 测试绘图，观察跨越极大值点位置是否找对
        #Window_plotor(num_axis, gait_data[i][0], r_valleyind_sorted); plt.title(str(i+1) + 'th trial\'s valley points') # 测试绘图，观察跨越前极小值点位置是否找对

        # 取无跨越意图EEG窗，标记为0   
        rp_win_index = r_peakind_sorted + bias_0 # 窗起始索引
        rp_win_index = rp_win_index * 512 / fs_gait
        lp_win_index = l_peakind_sorted + bias_0 # 窗起始索引
        lp_win_index = lp_win_index * 512 / fs_gait
        # 取有跨越意图EEG窗，标记为1
        rv_win_index = r_valleyind_sorted + bias_1
        rv_win_index = rv_win_index * 512 / fs_gait
        lv_win_index = l_valleyind_sorted + bias_1
        lv_win_index = lv_win_index * 512 / fs_gait
        
        for k in range(work_trial_2):
            if r_peakind_sorted[k] < l_peakind_sorted[k]:
                # 先跨右腿
                #print('r') # 测试用，观察跨越用的腿是否一致
                # 无跨越意图窗
                out_eeg = eeg_data[0][i][:,int(rp_win_index[k]):(int(rp_win_index[k])+win_width)]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),0]
                output.append(out_eeg)
                # 有跨越意图窗
                out_eeg =  eeg_data[0][i][:,int(rv_win_index[k]-win_width):int(rv_win_index[k])]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),1]
                output.append(out_eeg)
            else:
                #print('l') # 测试用，观察跨越用的腿是否一致
                # 无跨越意图窗
                out_eeg = eeg_data[0][i][:,int(lp_win_index[k]):(int(lp_win_index[k])+win_width)]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),0]
                output.append(out_eeg)
                # 有跨越意图窗
                out_eeg =  eeg_data[0][i][:,int(lv_win_index[k]-win_width):int(lv_win_index[k])]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),1]
                output.append(out_eeg)
                     
        out_count += 1
        
    elif len(gait_data[i]) and i>=12 and i<=19 and i!=13 and i!=17 and i!=19: # 第13到20次往返3次
        # 当步态数据不是空集时（有效时）   
        # 取右膝跨越极值点索引
        r_peakind = find_peak_point(gait_data[i][0])
        r_peak = [gait_data[i][0][j] for j in r_peakind] # 获取极值点
        r_peak_sorted = sorted(r_peak, reverse=True) # 将极值点降序排序
        r_peakind_sorted = [] # 对应降序排序极值点的索引
        for j in r_peak_sorted[:work_trial_3]:
            r_peakind_sorted.append(list(gait_data[i][0]).index(j))
        r_peakind_sorted = np.array(sorted(r_peakind_sorted))
        
        # 取左膝跨越极值点索引
        l_peakind = find_peak_point(gait_data[i][1])
        l_peak = [gait_data[i][1][j] for j in l_peakind] # 获取极值点
        l_peak_sorted = sorted(l_peak, reverse=True) # 将极值点降序排序
        l_peakind_sorted = [] # 对应降序排序极值点的索引
        for j in l_peak_sorted[:work_trial_3]:
            l_peakind_sorted.append(list(gait_data[i][1]).index(j))
        l_peakind_sorted = np.array(sorted(l_peakind_sorted))
        
        r_valleyind_sorted = np.array(find_valley_point(gait_data[i][0], r_peakind_sorted)) # 右膝跨越前的极小值点
        l_valleyind_sorted = np.array(find_valley_point(gait_data[i][1], l_peakind_sorted)) # 左膝跨越前的极小值点
        num_axis = len(gait_data[i][0])
       
        #Window_plotor(num_axis, gait_data[i][0], r_peakind_sorted); plt.title(str(i+1) + 'th trial\'s peak points') # 测试绘图，观察跨越极大值点位置是否找对
        #Window_plotor(num_axis, gait_data[i][0], r_valleyind_sorted); plt.title(str(i+1) + 'th trial\'s valley points') # 测试绘图，观察跨越前极小值点位置是否找对

        # 取无跨越意图EEG窗，标记为0   
        rp_win_index = r_peakind_sorted + bias_0 # 窗起始索引
        rp_win_index = rp_win_index * 512 / fs_gait
        lp_win_index = l_peakind_sorted + bias_0 # 窗起始索引
        lp_win_index = lp_win_index * 512 / fs_gait
        # 取有跨越意图EEG窗，标记为1
        rv_win_index = r_valleyind_sorted + bias_1
        rv_win_index = rv_win_index * 512 / fs_gait
        lv_win_index = l_valleyind_sorted + bias_1
        lv_win_index = lv_win_index * 512 / fs_gait
        
        for k in range(work_trial_3):
            if r_peakind_sorted[k] < l_peakind_sorted[k]:
                # 先跨右腿
                #print('r') # 测试用，观察跨越用的腿是否一致
                # 无跨越意图窗
                out_eeg = eeg_data[0][i][:,int(rp_win_index[k]):(int(rp_win_index[k])+win_width)]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),0]
                output.append(out_eeg)
                # 有跨越意图窗
                out_eeg =  eeg_data[0][i][:,int(rv_win_index[k]-win_width):int(rv_win_index[k])]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),1]
                output.append(out_eeg)
            else:
                #print('l') # 测试用，观察跨越用的腿是否一致
                # 无跨越意图窗
                out_eeg = eeg_data[0][i][:,int(lp_win_index[k]):(int(lp_win_index[k])+win_width)]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),0]
                output.append(out_eeg)
                # 有跨越意图窗
                out_eeg =  eeg_data[0][i][:,int(lv_win_index[k]-win_width):int(lv_win_index[k])]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),1]
                output.append(out_eeg)
                     
        out_count += 1
    else:
        continue
    
if id_subject < 10:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0'+str(id_subject)+\
                '_Data\\Subject_0'+str(id_subject)+'_WinEEG.mat',\
                {'WinEEG':output})
else:
    sio.savemat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+str(id_subject)+\
                '_Data\\Subject_'+str(id_subject)+'_WinEEG.mat',\
                {'WinEEG':output})