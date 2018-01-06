# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:25:28 2017

@author: PC-2
"""

import scipy.io as sio
import numpy as np
import scipy.signal as sis
import matplotlib.pyplot as plt

trail_num = 9 # 【设置】试验跨越次数

mat_data = sio.loadmat('filteredMotion.mat') # 读取mat文件
data = mat_data['filteredMotion']

# 获取极值点索引位置
#peakind_rh = sis.find_peaks_cwt(data[0], np.arange(1,10))
peakind_rk = sis.find_peaks_cwt(data[1], np.arange(1,10))
#peakind_lh = sis.find_peaks_cwt(data[2], np.arange(1,10))
peakind_lk = sis.find_peaks_cwt(data[3], np.arange(1,10))

"""
通过绘图可以观察到可以准确找到极值点，但是左髋数据不容易找到跨越动作
所以去掉髋关节
"""

data_rk = [data[1][i] for i in peakind_rk] # 获取右膝极值
data_lk = [data[3][i] for i in peakind_lk] # 获取左膝极值

# 降序排序
data_rk_sorted = sorted([data[1][i] for i in peakind_rk], reverse = True)
data_lk_sorted = sorted([data[3][i] for i in peakind_lk], reverse = True)

"""
# 判断跨越次数
count_rk = 0 # 右膝跨越次数
count_lk = 0 # 左膝跨越次数

std_temp = 0
for i in range(np.shape(data_rk_sorted)[0]):
    temp = np.std(data_rk_sorted[:(i+1)])
    if temp - std_temp < 1.6:
        count_rk = count_rk + 1
        std_temp = temp
        continue
    else:
        break
    
std_temp = 0
for i in range(np.shape(data_lk_sorted)[0]):
    temp = np.std(data_lk_sorted[:(i+1)])
    if temp - std_temp < 1.6:
        count_lk = count_lk + 1
        std_temp = temp
        continue
    else:
        break
"""
"""
假设极值点基本相同，用标准差的差值来判断是否是跨越动作极值点
"""
# 获得极值点索引并升序排序
rk_index = []
lk_index = []

for i in data_rk_sorted[:trail_num]:
    rk_index.append(list(data[1]).index(i))
for i in data_lk_sorted[:trail_num]:
    lk_index.append(list(data[3]).index(i))
    
rk_index_sorted = np.array(sorted(rk_index))
lk_index_sorted = np.array(sorted(lk_index))

# 判断哪只腿先跨
if rk_index_sorted[0] < lk_index_sorted[0]:
    # 右腿先跨
    rk_flag = True
    print("\nRight leg step over first\n")
    the_index = rk_index_sorted
    count = trail_num
else:
    # 左腿先跨
    rk_flag = False
    print("\nLeft leg step over first\n")
    the_index = lk_index_sorted
    count = trail_num

# 绘图-测试用
def Window_plotor(row):
    # num_row = np.shape(data)[0] # 获取原始数据行数
    num_col = np.shape(data)[1] # 获取原始数据列数
    data_axis = [i for i in range(num_col)]
    plt.figure(figsize=[20,4])
    plt.plot(data_axis, data[row])
    if rk_flag:
        for i in rk_index_sorted:
            plt.scatter(i, data[row][i])
    else:
        for i in lk_index_sorted:
            plt.scatter(i, data[row][i])

if rk_flag:
    Window_plotor(1)
else:
    Window_plotor(3)

# 确定EEG分段索引
"""
【设置】重要参数bias和win_width
"""
bias = 100 # 设定EEG窗终点距离跨越极值点的偏移量
win_width = 100 # 设置窗的长度

the_index = the_index - bias
eeg_index = the_index * 512 / 125
eeg_index_int = []
for i in range(np.shape(eeg_index)[0]):
    eeg_index_int.append(int(eeg_index[i]))
    
eeg_mat_data = sio.loadmat('labeledEEG.mat') # 读取mat文件
eeg_data = eeg_mat_data['labeledEEG']

# 对EEG信号带通滤波
def bandpass(data):
    fs = 512 # 采样频率512Hz
    upper = 8 # 上截止频率
    lower = 30 # 下截止频率
    Wn = [2 * upper / fs, 2 * lower / fs] # 截止频带0.1-1Hz or 8-30Hz
    b,a = sis.butter(4, Wn, 'bandpass')
    
    filtered_data = np.zeros([32, win_width])
    for row in range(32):
        filtered_data[row] = sis.filtfilt(b,a,data[row,:]) 
    
    return filtered_data

# 绘图查看滤波效果：测试用
def eeg_plotor(data):
    data_axis = [i for i in range(win_width)]
    plt.figure(figsize=[10,4])
    plt.plot(data_axis, data)  

# 输出滤波EEG窗
output_eeg = np.zeros([32, win_width])
count_1 = 1
count_2 = 1
count_3 = 1
for i in range(count):
    start_index = eeg_index_int[i]-win_width
    output_eeg = eeg_data[:,start_index:eeg_index_int[i]]
    #eeg_plotor(output_eeg[31]); plt.title('Origin' + str(i+1))
    output_eeg = bandpass(output_eeg)
    #eeg_plotor(output_eeg[31]); plt.title('Filtered' + str(i+1))
    if (i==2 or i==3 or i==8):
        sio.savemat('1_EEG_win'+str(count_1)+'.mat',{'1_EEG_win'+str(count_1):output_eeg})
        count_1 += 1
    elif (i==1 or i==4 or i==6):
        sio.savemat('2_EEG_win'+str(count_2)+'.mat',{'2_EEG_win'+str(count_2):output_eeg})
        count_2 += 1
    else:
        sio.savemat('3_EEG_win'+str(count_3)+'.mat',{'3_EEG_win'+str(count_3):output_eeg})
        count_3 += 1
    