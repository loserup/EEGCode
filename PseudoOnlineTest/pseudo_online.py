# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:22:42 2018

伪在线测试

@author: Long
"""

# In[1]:
import scipy.io as sio
import numpy as np
import scipy.signal as sis
import matplotlib.pyplot as plt
import copy

from sklearn.externals import joblib

# In[2]:



classifier = joblib.load("E:\\EEGExoskeleton\\Data\\TestData\\SVM_2.m")
#eeg_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\TestData\\rawEEG_1.mat')['rawEEG']
eeg_data = sio.loadmat('E:\\EEGExoskeleton\\Data\\TestData\\data_history_2.mat')['data_history']
csp = sio.loadmat('csp.mat')['csp']
    


# In[4]:
# 对EEG信号带通滤波
fs = 512 # 【采样频率512Hz】
win_width = 384 # 【窗宽度】384对应750ms窗长度
def bandpass(data,upper,lower):
    Wn = [2 * upper / fs, 2 * lower / fs] # 截止频带0.1-1Hz or 8-30Hz
    b,a = sis.butter(4, Wn, 'bandpass')
    
    filtered_data = np.zeros([32, win_width])
    for row in range(32):
        filtered_data[row] = sis.filtfilt(b,a,data[row,:]) 
    
    return filtered_data 

test_feat_all = [] # 记录喂给分类器的特征
###以下是伪在线测试
def output(WIN,THRED,thres,thres_inver):
    """output : 依次输出指定受试对象的伪在线命令，滤波伪在线命令，二次滤波伪在线命令
    以及步态图像并保存图像文件.
    Parameters:
    -----------
    - No_trail: 指定数据来源的试验（trail）号
    - WIN: 伪在线向前取WIN个窗的标签
    - THRED: WIN个窗中标签个数超过阈值THRED则输出跨越命令
    - thres: 当连续为跨越意图（1）的个数不超过阈值thres时，全部变成-1
    - thres_inver: 反向滤波阈值：将连续跨越意图间的短-1段补成1
    """
    output_0,output_1,output_2 = [],[],[]
    
    for i in range(len(eeg_data[0])):
        if i < win_width: # 初始阶段没有完整的750ms窗，384对应750ms窗长度
            continue 
        elif i % 26 != 0: # 每隔50ms取一次窗
            continue
        else:
            test_eeg = eeg_data[:,(i-win_width):i]
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
            output_0.append(int(classifier.predict(test_feat)))
            
    count = 0
            
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
    
    return output_0,output_1,output_2


# 参数设置
WIN = 20 # 伪在线向前取WIN个窗的标签
THRED = 18 # WIN个窗中标签个数超过阈值THRED则输出跨越命令
thres = 5 # 当连续为跨越意图（1）的个数不超过阈值thres时，全部变成-1
thres_inver = 15 # 反向滤波阈值：将连续跨越意图间的短-1段补成1

output_0,output_1,output_2 = output(WIN,THRED,thres,thres_inver)

test_feat_all = np.array(test_feat_all).reshape(len(test_feat_all),len(csp)) 



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