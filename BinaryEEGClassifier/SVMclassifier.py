# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:21:17 2018

@author: Long

% 说明:基于SMO的SVM分类器

% 第五步

% 输入数据最后一列为标签，0表示无意图，1表示有意图
% 数据其余列为特征值
"""

import scipy.io as sio
from sklearn.utils import shuffle
import numpy as np


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

# 获得EEG信号的特征feats和标签labels
feats_all = feats_mat['features']
feats, labels = shuffle(feats_all[:,:-1],feats_all[:,-1],\
                            random_state=np.random.randint(0,100))

"""以下是序列最小优化SMO"""
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2))) #first column is valid flag
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(np.random.uniform(0,m))
    return j

def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: 
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: 
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): 
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): 
            oS.b = b2
        else: 
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('rbf', 10)):    # 序列最小优化SMO
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
            iter += 1
        if entireSet:
            entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True  
    return oS.b,oS.alphas
"""以下是序列最小优化SMO"""

"""以下是SVM分类"""
reach = 25 # 【到达率，rbf函数值跌落到0的速度参数】
b,alphas = smoP(feats, labels, 200, 0.0001, 10000, ('rbf', reach))
featMat=np.mat(feats); labelMat = np.mat(labels).transpose()
svInd=np.nonzero(alphas.A>0)[0]
sVs=featMat[svInd]
labelSV = labelMat[svInd];
print ("there are %d Support Vectors" % np.shape(sVs)[0])
m,n = np.shape(featMat)
errorCount = 0
for i in range(m):
    kernelEval = kernelTrans(sVs,featMat[i,:],('rbf', reach))
    predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
    if np.sign(predict)!=np.sign(labels[i]): 
        errorCount += 1
print ("The reach rate is: %i" % reach +"\n") 
print ("the training error rate is: %f" % (float(errorCount)/m))
"""以上是SVM分类"""

import scipy.signal as sis

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

### CSP算法
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
    if i < 384: # 初始阶段没有完整的750ms窗
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
        
        kernelEval = kernelTrans(sVs,test_feat,('rbf', 10))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        output.append(np.sign(predict))
        

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
thres_inver = 10
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

# 绘制测试结果，观察有/无跨越意图是否分界明显
import matplotlib.pyplot as plt
axis = [i for i in range(len(output))]
plt.figure(figsize=[15,4])
plt.plot(axis, output)



