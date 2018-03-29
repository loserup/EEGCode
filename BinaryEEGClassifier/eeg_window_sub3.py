# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:25:28 2017

@author: SingleLong

% 说明：

% 第三步

% 根据步态信号建立划窗
% 划窗截取EEG信号
% 生成指定受试对象的有意图和无意图区域的EEG窗

% 专门针对第三个受试对象的划窗函数
受试对象3共进行了15次trail
第1次：往返3次
第2次：往返3次；数据不好处理，去掉
第3次：往返3次
第4次：往返3次
第5次：往返3次
第6次：打标失败，去掉
第7次：往返3次
第8次：往返3次
第9次：往返3次
第10次：打标失败，去掉
第11次：往返3次
第12次：往返3次
第13次：往返3次
第14次：往返3次
第15次：往返3次
备注：受试对象被告知用右腿跨越障碍
最后有效trail有12组，共往返36次，跨越216次，共432个窗
"""

import scipy.io as sio
import numpy as np
import scipy.signal as sis
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

gait_data = gait_mat_data['FilteredMotion'][0] # 每个元素是受试者走的一次trail；每个trail记录双膝角度轨迹，依次是右膝和左膝
eeg_data = eeg_mat_data['CutedEEG'] # eeg_data[0][i]表示第i次trial的EEG，共32行（频道）

num_trial = len(gait_data) # 获取受试者进行试验的次数

# 绘图-测试用
def Window_plotor_peak(num_axis, data, index_sorted, bias,win_width):
    # 绘制峰值点以及相应划窗
    data_axis = [i for i in range(num_axis)]
    plt.figure(figsize=[15,4])
    ax = plt.gca() # 创建子图ax，用来画窗框
    plt.plot(data_axis, data)
    for i in index_sorted:
        plt.scatter(i, data[i])
        rect = patches.Rectangle((i+bias,data[i]),win_width,-40,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

def Window_plotor_valley(num_axis, data, index_sorted, bias,win_width):
    # 绘制谷值点以及相应划窗
    data_axis = [i for i in range(num_axis)]
    plt.figure(figsize=[15,4])
    ax = plt.gca() # 创建子图ax，用来画窗框
    plt.plot(data_axis, data)
    for i in index_sorted:
        plt.scatter(i, data[i])
        rect = patches.Rectangle((i+bias,data[i]),-win_width,40,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

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
win_width = 128 # 【窗宽度】384对应750ms窗长度
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
peak_bias = 0 # 【设置从膝关节角度最大处的偏移值，作为划无意图窗的起点】
valley_bias = 0 # 【设置从膝关节角度最大处的偏移值，作为划无意图窗的起点】
for i in range(num_trial):
    if len(gait_data[i]) and i!=1: # 受试对象3的第二次trial效果不好，故去掉
        # 当步态数据不是空集时（有效时）
        
        # 取右膝跨越极值点索引
        r_peakind = find_peak_point(gait_data[i][0])
        r_peak = [gait_data[i][0][j] for j in r_peakind] # 获取极值点
        r_peak_sorted = sorted(r_peak, reverse=True) # 将极值点降序排序
        r_peakind_sorted = [] # 对应降序排序极值点的索引
        for j in r_peak_sorted[:work_trial]:
            r_peakind_sorted.append(list(gait_data[i][0]).index(j))
        r_peakind_sorted = np.array(sorted(r_peakind_sorted))
        
        # 取左膝跨越极值点索引
        l_peakind = find_peak_point(gait_data[i][1])
        l_peak = [gait_data[i][1][j] for j in l_peakind] # 获取极值点
        l_peak_sorted = sorted(l_peak, reverse=True) # 将极值点降序排序
        l_peakind_sorted = [] # 对应降序排序极值点的索引
        for j in l_peak_sorted[:work_trial]:
            l_peakind_sorted.append(list(gait_data[i][1]).index(j))
        l_peakind_sorted = np.array(sorted(l_peakind_sorted))
        
        r_valleyind_sorted = np.array(find_valley_point(gait_data[i][0], r_peakind_sorted)) # 右膝跨越前的极小值点
        l_valleyind_sorted = np.array(find_valley_point(gait_data[i][1], l_peakind_sorted)) # 左膝跨越前的极小值点
        num_axis = len(gait_data[i][0])
       
        Window_plotor_peak(num_axis, gait_data[i][0], r_peakind_sorted, peak_bias,win_width); plt.title(str(i+1) + 'th trial\'s peak points') # 测试绘图，观察跨越极大值点位置是否找对
        Window_plotor_valley(num_axis, gait_data[i][0], r_valleyind_sorted, valley_bias,win_width); plt.title(str(i+1) + 'th trial\'s valley points') # 测试绘图，观察跨越前极小值点位置是否找对

        # 取无跨越意图EEG窗，标记为0   
        rp_win_index = r_peakind_sorted + peak_bias # 窗起始索引
        rp_win_index = rp_win_index * 512 / fs_gait
        lp_win_index = l_peakind_sorted + peak_bias # 窗起始索引
        lp_win_index = lp_win_index * 512 / fs_gait
        # 取有跨越意图EEG窗，标记为1
        rv_win_index = r_valleyind_sorted + valley_bias
        rv_win_index = rv_win_index * 512 / fs_gait
        lv_win_index = l_valleyind_sorted + valley_bias
        lv_win_index = lv_win_index * 512 / fs_gait
        
        Window_plotor_peak(num_axis, gait_data[i][0], r_peakind_sorted, peak_bias, win_width); plt.title(str(i+1) + 'th trial\'s peak points') # 测试绘图，观察跨越极大值点位置是否找对
        Window_plotor_valley(num_axis, gait_data[i][0], r_valleyind_sorted, valley_bias, win_width); plt.title(str(i+1) + 'th trial\'s valley points') # 测试绘图，观察跨越前极小值点位置是否找对
        
        for k in range(work_trial):
            if r_peakind_sorted[k] < l_peakind_sorted[k]:
                # 先跨右腿
                #print('r') # 测试用，观察跨越用的腿是否一致
                # 无跨越意图窗
                out_eeg = eeg_data[0][i][:,int(rp_win_index[k]):(int(rp_win_index[k])+win_width)]
                out_eeg_band0 = bandpass(out_eeg,upper=0.3,lower=3)
                out_eeg_band1 = bandpass(out_eeg,upper=4,lower=7)
                out_eeg_band2 = bandpass(out_eeg,upper=8,lower=13)
                out_eeg_band3 = bandpass(out_eeg,upper=13,lower=30)
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),-1] # 将四种带通滤波后的EEG窗拼接起来合成一个更长的窗
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
                out_eeg = [np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3)),-1]
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