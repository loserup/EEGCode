# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 21:52:16 2018

在线out_store二次滤波绘图

@author: Long
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import copy

output_0 = sio.loadmat('output.mat')['out_store'][0]

WIN = 20 # 伪在线向前取WIN个窗的标签
THRED = 18 # WIN个窗中标签个数超过阈值THRED则输出跨越命令
thres = 5 # 当连续为跨越意图（1）的个数不超过阈值thres时，全部变成-1
thres_inver = 15 # 反向滤波阈值：将连续跨越意图间的短-1段补成1

count = 0

output_1 = []
        
# 一次滤波：伪在线向前取WIN个窗的标签，
# WIN个窗中标签个数超过阈值THRED则输出跨越命令
for i in np.arange(WIN,len(output_0)):
    for j in np.arange(i-WIN,i):
        if output_0[j] == 1:
            count += 1
        else:
            continue
    if count >= THRED:
        output_1.append(1)
        count = 0
    else:
        output_1.append(-1)
        count = 0
        
# 二次滤波
# 反向滤波：当连续为无跨越意图（-1）的个数不超过阈值thres_inter时，全部变成1
count = 0
output_2 = copy.deepcopy(output_1)    
for i in range(len(output_2)):
    if output_2[i] == -1:
        count = count + 1
    else:
        if count < thres_inver:
            for j in range(count):
                output_2[i-j-1] = 1
            count = 0
        else:
            count = 0
            continue
output_2[-1] = -1

# 正向滤波：当连续为跨越意图（1）的个数不超过阈值thres时，全部变成-1
count = 0

for i in range(len(output_2)):
    if output_2[i] == 1:
        if i == len(output_2)-1:
            for j in range(count):
                output_2[i-j-1] = -1
        else:
            count = count + 1
    else:
        if count < thres:
            for j in range(count):
                output_2[i-j-1] = -1
            count = 0
        else:
            count = 0
            continue

plt.figure(figsize=[15,8]) 
axis = [j for j in range(len(output_0))]
plt.subplot(311)
plt.plot(axis, output_0)

axis = [j for j in range(len(output_1))]
plt.subplot(312)
plt.plot(axis, output_1)

axis = [j for j in range(len(output_2))]
plt.subplot(313)
plt.plot(axis, output_2)