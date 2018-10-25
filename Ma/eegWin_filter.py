# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:50:24 2018

第5步：带标签EEG窗的带通滤波器

@author: Long
"""

import scipy.io as sio
import numpy as np
import scipy.signal as sis

eeg = sio.loadmat('E:\\EEGExoskeleton\\Dataset\\Ma\\20180829\\labeledEEG.mat')
eeg = eeg['output']

eegwin_0 = [] # 存放标记为-1的EEG窗
eegwin_1 = [] # 存放标记为1的EEG窗

for i in range(len(eeg)):
    if int(eeg[i][1]) == -1:
        # 若EEG窗标记为0
        eegwin_0.append(eeg[i][0])
    elif int(eeg[i][1]) == 1:
        eegwin_1.append(eeg[i][0])