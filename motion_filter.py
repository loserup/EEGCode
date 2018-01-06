# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:06:42 2017

@author: PC-2
"""

import scipy.io as sio
import numpy as np
import scipy.signal as sis
import matplotlib.pyplot as plt


mat_data = sio.loadmat('rawMotion.mat') # 读取mat文件
data = mat_data['rawMotion'].T

num_row = np.shape(data)[0] # 获取原始数据行数
num_col = np.shape(data)[1] # 获取原始数据列数

# 对动作信号低通滤波
def lowpass(data):
    fs = 125 # 采样频率125Hz
    Wn = 1 # 截止频率2Hz
    b,a = sis.butter(4, 2*Wn/fs, 'lowpass')
    
    filtered_data = np.zeros([num_row, num_col])
    for row in range(num_row):
        filtered_data[row] = sis.filtfilt(b,a,data[row,:]) 
    
    return filtered_data


filtered_data = lowpass(data)
sio.savemat('filteredMotion.mat', {'filteredMotion' : filtered_data})


# 绘图查看滤波效果：测试用
def origin_vs_filtered_plotor():
    data_axis = [i for i in range(num_col)]
    for row in range(num_row):
        plt.figure(figsize=[20,8])
        plt.subplot(2,1,1)
        plt.plot(data_axis, data[row])
        if row == 0:
            plt.title(u'Right Hip')
        elif row == 1:
            plt.title(u'Right Knee')
        elif row == 2:
            plt.title(u'Left Hip')
        elif row == 3:
            plt.title(u'Left Knee')
        plt.subplot(2,1,2)
        plt.plot(data_axis, filtered_data[row])  
        if row == 0:
            plt.title(u'Filtered Right Hip')
        elif row == 1:
            plt.title(u'Filtered Right Knee')
        elif row == 2:
            plt.title(u'Filtered Left Hip')
        elif row == 3:
            plt.title(u'Filtered Left Knee')

#origin_vs_filtered_plotor() # 原始数据与滤波数据做对比

def filtered_vs_filtered_ploter():
    data_axis = [i for i in range(num_col)]
    plt.figure(figsize=[20,4])
    plt.plot(data_axis, filtered_data[1], color = 'blue')
    plt.plot(data_axis, filtered_data[3], color = 'orange')
    
#filtered_vs_filtered_ploter()
"""
通过滤波后数据对比，右膝先跨越障碍物，之后取窗的时候
"""
  
    