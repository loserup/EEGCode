# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:32:50 2017

@author: SingleLong

% 说明：

% 第二步

% 将两次打标之间的EEG从原始EEG信号中提取出来
"""

import scipy.io as sio
import numpy as np

id_subject = 3 # 受试者的编号

if id_subject < 10:
    mat_data = sio.loadmat('rawEEG_0' + str(id_subject) + '.mat')
else:
    mat_data = sio.loadmat('rawEEG_' + str(id_subject) + '.mat')

data = mat_data['rawEEG']

num_trial = np.shape(data)[1] # 获取受试者进行试验的次数

# 找两次打标位置
for i in range(num_trial):
    label_index = 0 # 打标位置

    # 通过上升沿找两次打标位置
    temp = data[0][i][32][0]
    for data_label in data[0][i][32]:
        if data_label <= temp:
            label_index += 1
            temp = data_label
            continue
        else:
            label_index += 1
            temp = data_label
            break
        
    label_index_1 = label_index # 第一次打标位置
    
    for data_label in data[0][i][32][label_index_1:]:
        if data_label <= temp:
            label_index += 1
            temp = data_label
            continue
        else:
            label_index += 1
            break
    
    label_index_2 = label_index # 第二次打标位置
    
    # 截取两次打标之间的数据
    data[0][i] = data[0][i][0:32, label_index_1:label_index_2]

if id_subject < 10:
    sio.savemat('labeledEEG_0'+str(id_subject)+'.mat', {'labeledEEG' : data})
else:
    sio.savemat('labeledEEG_'+str(id_subject)+'.mat', {'labeledEEG' : data})
        
