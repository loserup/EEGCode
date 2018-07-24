# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:24:58 2018

研究脑控命令和步态时间轴对应关系

@author: Long
"""

# In[]
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# In[]
gait = sio.loadmat('FilteredMotion.mat')['FilteredMotion'][0][0][0] # 读取右腿步态数据
cmd = sio.loadmat('output_2.mat')['output_2'][0] # 读取输出命令序列

# In[]
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

# In[]
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

# In[获得跨越极值点索引peakind]
gait_peakind = find_peak_point(gait) # 步态数据极大值点索引
peak = [gait[j] for j in gait_peakind] # 获取极大值点
peak_sorted = sorted(peak, reverse=True) # 将极值点降序排序
peakind = [] # 对应降序排序极值点的索引
for j in peak_sorted[:12]:
    peakind.append(list(gait).index(j))
peakind = np.array(sorted(peakind))

# In[获得跨越极值点前谷值点索引valleyind]
valleyind = np.array(find_valley_point(gait, peakind))

# In[测试画图，观察是否正确找到谷值点]
#def test_plotor(gait, index):
#    # 绘制峰值点以及相应划窗
#    data_axis = [i for i in range(len(gait))]
#    plt.figure(figsize=[15,4])
#    plt.plot(data_axis, gait)
#    for i in index:
#        plt.scatter(i, gait[i])
#
#test_plotor(gait, valleyind)
#plt.xlabel('sampling point')
#plt.ylabel('knee angle(°)')

# In[]
def gait_plotor(gait, index):
    # 绘制峰值点以及相应划窗
    data_axis = [i*(1/121) for i in range(len(gait))]
    plt.figure(figsize=[15,3])
    plt.plot(data_axis, gait)
    for i in index:
        plt.scatter(i*(1/121), gait[i])

gait_plotor(gait, valleyind)
plt.xlabel('time (s)')
plt.ylabel('knee angle(°)')

# In[]
gait_time = [i*(1/121) for i in valleyind] # 谷值点的时刻，单位：s

# In[]
#plt.figure(figsize=[15,4])
#axis = [i for i in range(len(cmd))]
#plt.plot(axis, cmd)

# In[找cmd上升沿的点]
cmdind = [] # 上升沿索引
for i in np.arange(1,len(cmd)):
    if cmd[i-1] == -1 and cmd[i] == 1:
        cmdind.append(i)
#
#test_plotor(cmd, cmdind)
#plt.xlabel('sampling point')
#plt.ylabel('category')
        
# In[]
cmd_time = [((384+28*20)/512+i*28/512) for i in cmdind] # 命令上升沿时刻，单位：s

# In[]
def cmd_plotor(cmd, index):
    # 绘制峰值点以及相应划窗
    data_axis = [((384+28*20)/512+i*28/512) for i in range(len(cmd))]
    plt.figure(figsize=[15,3])
    plt.plot(data_axis, cmd)
    for i in index:
        plt.scatter(((384+28*20)/512+i*28/512), cmd[i])

cmd_plotor(cmd, cmdind)
plt.xlabel('time (s)')
plt.ylabel('knee angle(°)')

# In[]
plt.figure(figsize=[15,3])
axis = [i for i in range(len(gait_time))]
plt.plot(gait_time, axis, label = 'gait')
axis = [i for i in range(len(cmd_time))]
plt.plot(cmd_time, axis, label = 'cmd')
plt.legend(loc='upper left')
plt.xlabel('time (s)')
plt.ylabel('key point')
for i in gait_time:
    plt.scatter(i,gait_time.index(i))
for i in cmd_time:
    plt.scatter(i,cmd_time.index(i))