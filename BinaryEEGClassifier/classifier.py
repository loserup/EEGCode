# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:21:17 2018

@author: Long

% 说明:基于sklearn库的SVM分类器

% 第五步

% 输入数据最后一列为标签，0表示无意图，1表示有意图
% 数据其余列为特征值
"""

from sklearn.svm import SVC
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn import cross_validation
import numpy as np
import scipy.signal as sis


id_subject = 3 # 【受试者的编号】
if id_subject < 10:
    feats_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0'+\
                            str(id_subject)+'_Data\\Subject_0'+\
                            str(id_subject)+'_features.mat')
    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0' +\
                               str(id_subject) + '_Data\\Subject_0' +\
                               str(id_subject) + '_CutedEEG.mat')
    input_eegwin_dict = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0'+\
                                    str(id_subject)+'_Data\\Subject_0'+\
                                    str(id_subject)+'_WinEEG.mat')
else:
    feats_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+\
                            str(id_subject)+'_Data\\Subject_'+\
                            str(id_subject)+'_features.mat')
    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_' +\
                               str(id_subject) + '_Data\\Subject_' +\
                               str(id_subject) + '_CutedEEG.mat')
    input_eegwin_dict = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+\
                                    str(id_subject)+'_Data\\Subject_'+\
                                    str(id_subject)+'_WinEEG.mat')

eeg_data = eeg_mat_data['CutedEEG']
feats_all = feats_mat['features']
accuracy_sum = 0
count = 10.0 # 随机计算准确率的次数
# 随机打乱特征顺序
for i in range(int(count)):
    feats, labels = shuffle(feats_all[:,:-1],feats_all[:,-1],\
                            random_state=np.random.randint(0,100))
    # 建立SVM模型
    params = {'kernel':'rbf','probability':True, 'class_weight':'balanced'} # 类别0明显比其他类别数目多，但加了'class_weight':'balanced'平均各类权重准确率反而更低了
    classifier = SVC(**params)
    classifier.fit(feats,labels) # 训练SVM分类器
    accuracy = cross_validation.cross_val_score(classifier, feats, labels,\
                                            scoring='accuracy',cv=3)
    accuracy_sum += accuracy
    print ('Accuracy of the classifier: '+str(round(100*accuracy.mean(),2))+'%')
        
accuracy_avg = accuracy_sum / count
print ('\nAverage accuracy is ' + str(round(100*accuracy_sum.mean()/count,2))+'%')


# 对EEG信号带通滤波
fs = 512 # 【采样频率512Hz】
win_width = 128 # 【窗宽度】384对应750ms窗长度
def bandpass(data,upper,lower):
    Wn = [2 * upper / fs, 2 * lower / fs] # 截止频带0.1-1Hz or 8-30Hz
    b,a = sis.butter(4, Wn, 'bandpass')
    
    filtered_data = np.zeros([32, win_width])
    for row in range(32):
        filtered_data[row] = sis.filtfilt(b,a,data[row,:]) 
    
    return filtered_data 

# 需要先用离线数据训练出对应用户的csp投影矩阵
import scipy.linalg as la # 线性代数库
num_pair = 6 # 【从CSP投影矩阵里取得特征对数】

input_eegwin = input_eegwin_dict['WinEEG']

eegwin_0 = [] # 存放标记为0的EEG窗
eegwin_1 = [] # 存放标记为1的EEG窗
for i in range(len(input_eegwin)):
    if int(input_eegwin[i][1]) == -1:
        # 若EEG窗标记为0
        eegwin_0.append(input_eegwin[i][0])
    elif int(input_eegwin[i][1]) == 1:
        eegwin_1.append(input_eegwin[i][0])
        
task = (eegwin_0, eegwin_1)

# 获取EEG窗的标准化空间协方差矩阵
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

### CSP算法，训练投影矩阵csp，以便在伪在线中从EEG窗提取出特征
filters = ()
C_0 = covarianceMatrix(task[0][0])
for i in range(1,len(task[0])):
    C_0 += covarianceMatrix(task[0][i])
C_0 = C_0 / len(task[0]) # 获得标记为0的EEG窗的标准化协方差对称矩阵

C_1 = 0 * C_0 # 用C_1 = np.empty(C_0.shape)有些极小的随机非0数，会导致输出结果每次都会改变
for i in range(0,len(task[1])):
    C_1 += covarianceMatrix(task[1][i])
C_1 = C_1 / len(task[1]) # 获得标记为1的EEG窗的标准化协方差对称矩阵

C = C_0 + C_1 # 不同类别的复合空间协方差矩阵,这是一个对称矩阵
E,U = la.eig(C) # 获取复合空间协方差矩阵的特征值E和特征向量U,这里C可以分解为C=np.dot(U,np.dot(np.diag(E),U.T))
#E = E.real # E取实部；取实部后不能实现np.diag(E_0)+np.diag(E_1)=I

order = np.argsort(E) # 升序排序
order = order[::-1] # 翻转以使特征值降序排序
E = E[order] 
U = U[:,order]

P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U)) # 获取白化变换矩阵

# 获取白化变换后的协方差矩阵
S_0 = np.dot(P,np.dot(C_0, np.transpose(P))) 
S_1 = np.dot(P,np.dot(C_1, np.transpose(P)))

E_0,U_0 = la.eig(S_0)
# 至此有np.diag(E_0)+np.diag(E_1)=I以及U_0=U_1

# 这里特征值也要按降序排序
order = np.argsort(E_0)
order = order[::-1]
E_0 = E_0[order]
U_0 = U_0[:,order]

#E_1,U_1 = la.eig(S_1);E_1 = E_1[order];U_1 = U_1[:,order] #测试是否满足np.diag(E_0)+np.diag(E_1)=I

# 求得CSP投影矩阵W
W = np.dot(np.transpose(U_0),P)

csp = np.zeros([num_pair*2,np.shape(W)[0]]) # 提取特征的投影矩阵
csp[0:num_pair,:] = W[0:num_pair,:] # 取投影矩阵前几行
csp[num_pair:,:] = W[np.shape(W)[1]-num_pair:,:] # 对应取投影矩阵后几行

No_trail = 3 # 选择第No_trail+1次的trail数据进行测试 
output = []
  
for i in range(len(eeg_data[0][0][0])):
    if i < 128: # 初始阶段没有完整的750ms窗，384对应750ms窗长度
        continue 
    elif i % 26 != 0: # 每隔50ms取一次窗
        continue
    else:
        test_eeg = eeg_data[0][No_trail][:,(i-win_width):i]
        out_eeg_band0 = bandpass(test_eeg,upper=0.3,lower=3)
        out_eeg_band1 = bandpass(test_eeg,upper=4,lower=7)
        out_eeg_band2 = bandpass(test_eeg,upper=8,lower=13)
        out_eeg_band3 = bandpass(test_eeg,upper=13,lower=30)
        test_eeg = np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3))
        test_feat = (np.log(np.var(np.dot(csp, test_eeg), axis=1))).reshape(1,num_pair*2) # classifier.predict需要和fit时相同的数据结构，所以要reshape
        output.append(int(classifier.predict(test_feat)))

"""
# 对伪在线分类结果进行简单滤波
# 当连续为跨越意图（1）的个数不超过阈值thres时，全部变成0
count = 0
thres = 15
for i in range(len(output)):
    if output[i] == 1:
        if i == len(output)-1:
            for j in range(count):
                output[i-j-1] = -1
        else:
            count = count + 1
    else:
        if count < thres:
            for j in range(count):
                output[i-j-1] = -1
            count = 0
        else:
            count = 0
            continue
# 反向滤波：将连续跨越意图间的短0段补成1
count = 0
thres_inver = 15
for i in range(len(output)):
    if output[i] == -1:
        count = count + 1
    else:
        if count < thres_inver:
            for j in range(count):
                output[i-j-1] = 1
            count = 0
        else:
            count = 0
            continue
output[-1] = -1
"""

# 绘制测试结果，观察有/无跨越意图是否分界明显
import matplotlib.pyplot as plt
axis = [i for i in range(len(output))]
plt.figure(figsize=[15,4])
plt.plot(axis, output)
