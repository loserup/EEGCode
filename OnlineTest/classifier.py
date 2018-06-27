# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:21:17 2018

@author: Long

% 说明: 训练基于sklearn库的SVM分类器

% 第五步

% 输入数据最后一列为标签，0表示无意图，1表示有意图
% 数据其余列为特征值
"""
# In[]
import scipy.io as sio
from sklearn import grid_search
from sklearn import svm
from sklearn.externals import joblib
import time
#import scipy.signal as sis
#import matplotlib.pyplot as plt
#import copy

# In[]
#id_subject = 1 # 【受试者的编号】
#if id_subject < 10:
#    feats_mat = sio.loadmat('features.mat')
#    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0' +\
#                               str(id_subject) + '_Data\\Subject_0' +\
#                               str(id_subject) + '_CutedEEG.mat')
#    gait_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_0' +\
#                                str(id_subject) + '_Data\\Subject_0' +\
#                                str(id_subject) + '_FilteredMotion.mat')
#    csp = sio.loadmat('csp.mat')['csp']
#else:
#    feats_mat = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_'+\
#                            str(id_subject)+'_Data\\Subject_'+\
#                            str(id_subject)+'_features.mat')
#    eeg_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_' +\
#                               str(id_subject) + '_Data\\Subject_' +\
#                               str(id_subject) + '_CutedEEG.mat')
#    gait_mat_data = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_' +\
#                                str(id_subject) + '_Data\\Subject_' +\
#                                str(id_subject) + '_FilteredMotion.mat')
#    csp = sio.loadmat('E:\\EEGExoskeleton\\EEGProcessor\\Subject_' +\
#                                str(id_subject) + '_Data\\Subject_' +\
#                                str(id_subject) + '_csp.mat')['csp']

# In[]
#eeg_data = sio.loadmat('CutedEEG.mat')['CutedEEG']
#gait_data = gait_mat_data['FilteredMotion'][0] # 每个元素是受试者走的一次trail；每个trail记录双膝角度轨迹，依次是右膝和左膝
feats_all = sio.loadmat('features.mat')['features']

parameter_grid = [  {'kernel': ['linear'], 'C': [10 ** x for x in range(-1, 4)]},
                    {'kernel': ['poly'], 'degree': [2, 3]},
                    {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [10 ** x for x in range(-1, 4)]},
                 ]

X = feats_all[:,:-1]
y = feats_all[:,-1]

print("\n#### Searching optimal hyperparameters for precision")
classifier = grid_search.GridSearchCV(svm.SVC(), 
            parameter_grid, cv=5, scoring='precision_weighted')
classifier.fit(X, y) # 直接用实时收集到的数据进行训练，不把数据分出测试集了，直接用在线数据进行测试

print("\nScores across the parameter grid:")
for params, avg_score, _ in classifier.grid_scores_:
    print(params, '-->', round(avg_score, 3))
print("\nHighest scoring parameter set:", classifier.best_params_)

#joblib.dump(classifier, time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))+"_SVM.m") # 按当前时间命名保存训练好的分类器
joblib.dump(classifier, "SVM.m") # 保存训练好的分类器
# In[]
#max_accuracy = 0
#count = 10.0 # 随机计算准确率的次数
#num_feats = len(feats_all)
#ave_accuracy, ave_f1, ave_precision, ave_recall = [],[],[],[]
## 随机打乱特征顺序
#print('\n')
#for i in range(int(count)):
#    feats, labels = shuffle(feats_all[:,:-1],feats_all[:,-1],\
#                            random_state=np.random.randint(0,100))
#    
##    feats_train = feats[:round(num_feats*0.8),:]
##    feats_test= feats[round(num_feats*0.8):,:]
##    labels_train = labels[:round(num_feats*0.8)]
##    labels_test = labels[round(num_feats*0.8):]
#    # 建立SVM模型
#    params = {'kernel':'rbf','probability':True, 'class_weight':'balanced'} # 类别0明显比其他类别数目多，但加了'class_weight':'balanced'平均各类权重准确率反而更低了
#    classifier_cur = SVC(**params)
#    classifier_cur.fit(feats,labels) # 训练SVM分类器
#    
#    accuracy = cross_validation.cross_val_score(classifier_cur, feats, labels, \
#                                                scoring='accuracy',cv=5) # cv=5指五折交叉验证
#    f1 = cross_validation.cross_val_score(classifier_cur, feats, labels, \
#                                          scoring='f1', cv=5)
#    precision = cross_validation.cross_val_score(classifier_cur,feats,labels, \
#                                          scoring='precision', cv=5)
#    recall = cross_validation.cross_val_score(classifier_cur, feats, labels, \
#                                          scoring='recall', cv=5)
#
#    ave_accuracy.append(accuracy.mean())
#    ave_f1.append(f1.mean())
#    ave_precision.append(precision.mean())
#    ave_recall.append(recall.mean())
#    
#    if max_accuracy < accuracy.mean(): 
#        # 选取准确率最高的分类器做之后的分类工作
#        classifier = classifier_cur
#        max_accuracy = accuracy.mean() # arrayA.mean()指数。组arrayA中所有元素的平均值
#        
#    # 评分估计的平均得分和95%置信区间
#    print('Accuracy: %0.4f (± %0.4f)' % (accuracy.mean(),accuracy.std()**2))
#    print('F1: %0.4f (± %0.4f)' % (f1.mean(),f1.std()**2))
#    print('Precision: %0.4f (± %0.4f)' % (precision.mean(),precision.std()**2))
#    print('Recall: %0.4f (± %0.4f)' % (recall.mean(),recall.std()**2))
#
#    print('\n')
#        
#ave_accuracy = np.array(ave_accuracy)
#ave_f1 = np.array(ave_f1)
#ave_precision = np.array(ave_precision)
#ave_recall = np.array(ave_recall)
#
#print ('\nAverage Accuracy: %0.4f (± %0.4f)' % (ave_accuracy.mean(),ave_accuracy.std()**2))
#print ('Average F1: %0.4f (± %0.4f)' % (ave_f1.mean(),ave_f1.std()**2))
#print ('Average Precision: %0.4f (± %0.4f)' % (ave_precision.mean(),ave_precision.std()**2))
#print ('Average Recall: %0.4f (± %0.4f)\n' % (ave_recall.mean(),ave_recall.std()**2))


# In[]
## 对EEG信号带通滤波
#fs = 512 # 【采样频率512Hz】
#win_width = 384 # 【窗宽度】384对应750ms窗长度
#def bandpass(data,upper,lower):
#    Wn = [2 * upper / fs, 2 * lower / fs] # 截止频带0.1-1Hz or 8-30Hz
#    b,a = sis.butter(4, Wn, 'bandpass')
#    
#    filtered_data = np.zeros([32, win_width])
#    for row in range(32):
#        filtered_data[row] = sis.filtfilt(b,a,data[row,:]) 
#    
#    return filtered_data 
#
####以下是伪在线测试
#def output(No_trail,WIN,THRED,thres,thres_inver):
#    """output : 依次输出指定受试对象的伪在线命令，滤波伪在线命令，二次滤波伪在线命令
#    以及步态图像并保存图像文件.
#
#    Parameters:
#    -----------
#    - No_trail: 指定数据来源的试验（trail）号
#    - WIN: 伪在线向前取WIN个窗的标签
#    - THRED: WIN个窗中标签个数超过阈值THRED则输出跨越命令
#    - thres: 当连续为跨越意图（1）的个数不超过阈值thres时，全部变成-1
#    - thres_inver: 反向滤波阈值：将连续跨越意图间的短-1段补成1
#    """
#    output_0,output_1,output_2 = [],[],[]
#    
#    for i in range(len(eeg_data[0][No_trail][0])):
#        if i < win_width: # 初始阶段没有完整的750ms窗，384对应750ms窗长度
#            continue 
#        elif i % 26 != 0: # 每隔50ms取一次窗
#            continue
#        else:
#            test_eeg = eeg_data[0][No_trail][:,(i-win_width):i]
#            out_eeg_band0 = bandpass(test_eeg,upper=0.3,lower=3)
#            out_eeg_band1 = bandpass(test_eeg,upper=4,lower=7)
#            out_eeg_band2 = bandpass(test_eeg,upper=8,lower=13)
#            out_eeg_band3 = bandpass(test_eeg,upper=13,lower=30)
#            test_eeg = np.hstack((out_eeg_band0,out_eeg_band1,out_eeg_band2,out_eeg_band3))
#            test_feat = (np.log(np.var(np.dot(csp, test_eeg), axis=1))).reshape(1,len(csp)) # classifier.predict需要和fit时相同的数据结构，所以要reshape
#            output_0.append(int(classifier.predict(test_feat)))
#            
#    count = 0
#            
#    # 一次滤波：伪在线向前取WIN个窗的标签，
#    # WIN个窗中标签个数超过阈值THRED则输出跨越命令
#    for i in np.arange(WIN,len(output_0)):
#        for j in np.arange(i-WIN,i):
#            if output_0[j] == 1:
#                count += 1
#            else:
#                continue
#        if count >= THRED:
#            output_1.append(1)
#            count = 0
#        else:
#            output_1.append(-1)
#            count = 0
#        
#    # 二次滤波
#    # 反向滤波：当连续为无跨越意图（-1）的个数不超过阈值thres_inter时，全部变成1
#    count = 0
#    output_2 = copy.deepcopy(output_1)    
#    for i in range(len(output_2)):
#        if output_2[i] == -1:
#            count = count + 1
#        else:
#            if count < thres_inver:
#                for j in range(count):
#                    output_2[i-j-1] = 1
#                count = 0
#            else:
#                count = 0
#                continue
#    output_2[-1] = -1
#    
#    # 正向滤波：当连续为跨越意图（1）的个数不超过阈值thres时，全部变成-1
#    count = 0
#
#    for i in range(len(output_2)):
#        if output_2[i] == 1:
#            if i == len(output_2)-1:
#                for j in range(count):
#                    output_2[i-j-1] = -1
#            else:
#                count = count + 1
#        else:
#            if count < thres:
#                for j in range(count):
#                    output_2[i-j-1] = -1
#                count = 0
#            else:
#                count = 0
#                continue
#    
#    return output_0,output_1,output_2
#
#
## 参数设置
#WIN = 20 # 伪在线向前取WIN个窗的标签
#THRED = 15 # WIN个窗中标签个数超过阈值THRED则输出跨越命令
#thres = 5 # 当连续为跨越意图（1）的个数不超过阈值thres时，全部变成-1
#thres_inver = 15 # 反向滤波阈值：将连续跨越意图间的短-1段补成1
#
#for i in range(len(eeg_data[0])):
#    if id_subject == 1:
#        # 如果受试对象号为1，且去除以下指定的无效试验号数
#        if i != 0 and i != 5 and i != 13 and i != 15 and i != 17 and i != 19:
#            output_0,output_1,output_2 = output(i,WIN,THRED,thres,thres_inver)
#            # 绘制测试结果，观察有/无跨越意图是否分界明显
#            plt.figure(figsize=[15,8]) 
#            axis = [j for j in range(len(output_0))]
#            plt.subplot(411)
#            plt.plot(axis, output_0)
#            plt.title(str(i+1) + 'th trial\'s output_'+str(THRED)+\
#                      "_"+str(WIN)+"_"+str(thres)+"_"+str(thres_inver))
#        
#            axis = [j for j in range(len(output_1))]
#            plt.subplot(412)
#            plt.plot(axis, output_1)
#        
#            axis = [j for j in range(len(output_2))]
#            plt.subplot(413)
#            plt.plot(axis, output_2)
#        
#            axis = [j for j in range(len(gait_data[i][0]))]
#            plt.subplot(414)
#            plt.plot(axis, gait_data[i][0])
#        
#            plt.savefig("E:\EEGExoskeleton\EEGProcessor\Images_Subject"+\
#                        str(id_subject)+"\Subject"+\
#                        str(id_subject)+"_trail"+str(i+1)+"_"+\
#                        str(THRED)+"_"+str(WIN)+"_"+str(thres)+"_"+\
#                        str(thres_inver)+".png")
#    
#    if id_subject == 2:
#        # 如果受试对象号为2，且去除以下指定的无效试验号数
#        if i!=2 and i!=8 and i!=9 and i!=12 and i!=13 and i!=15:
#            output_0,output_1,output_2 = output(i,WIN,THRED,thres,thres_inver)
#            # 绘制测试结果，观察有/无跨越意图是否分界明显
#            plt.figure(figsize=[15,8]) 
#            axis = [j for j in range(len(output_0))]
#            plt.subplot(411)
#            plt.plot(axis, output_0)
#            plt.title(str(i+1) + 'th trial\'s output_'+str(THRED)+\
#                      "_"+str(WIN)+"_"+str(thres)+"_"+str(thres_inver))
#        
#            axis = [j for j in range(len(output_1))]
#            plt.subplot(412)
#            plt.plot(axis, output_1)
#        
#            axis = [j for j in range(len(output_2))]
#            plt.subplot(413)
#            plt.plot(axis, output_2)
#        
#            axis = [j for j in range(len(gait_data[i][0]))]
#            plt.subplot(414)
#            plt.plot(axis, gait_data[i][0])
#        
#            plt.savefig("E:\EEGExoskeleton\EEGProcessor\Images_Subject"+\
#                        str(id_subject)+"\Subject"+\
#                        str(id_subject)+"_trail"+str(i+1)+"_"+\
#                        str(THRED)+"_"+str(WIN)+"_"+str(thres)+"_"+\
#                        str(thres_inver)+".png")
#        
#    if id_subject == 3:
#        # 如果受试对象号为3，且去除以下指定的无效试验号数
#        if i!=1 and i!=5 and i!=9:
#            output_0,output_1,output_2 = output(i,WIN,THRED,thres,thres_inver)
#            # 绘制测试结果，观察有/无跨越意图是否分界明显
#            plt.figure(figsize=[15,8]) 
#            axis = [j for j in range(len(output_0))]
#            plt.subplot(411)
#            plt.plot(axis, output_0)
#            plt.title(str(i+1) + 'th trial\'s output_'+str(THRED)+\
#                      "_"+str(WIN)+"_"+str(thres)+"_"+str(thres_inver))
#        
#            axis = [j for j in range(len(output_1))]
#            plt.subplot(412)
#            plt.plot(axis, output_1)
#        
#            axis = [j for j in range(len(output_2))]
#            plt.subplot(413)
#            plt.plot(axis, output_2)
#        
#            axis = [j for j in range(len(gait_data[i][0]))]
#            plt.subplot(414)
#            plt.plot(axis, gait_data[i][0])
#        
#            plt.savefig("E:\EEGExoskeleton\EEGProcessor\Images_Subject"+\
#                        str(id_subject)+"\Subject"+\
#                        str(id_subject)+"_trail"+str(i+1)+"_"+\
#                        str(THRED)+"_"+str(WIN)+"_"+str(thres)+"_"+\
#                        str(thres_inver)+".png")
#            
#    if id_subject == 4:
#        # 如果受试对象号为4，且去除以下指定的无效试验号数
#        if i!=2 and i!=8 and i!=12 and i!=13 and i!=14:
#            output_0,output_1,output_2 = output(i,WIN,THRED,thres,thres_inver)
#            # 绘制测试结果，观察有/无跨越意图是否分界明显
#            plt.figure(figsize=[15,8]) 
#            axis = [j for j in range(len(output_0))]
#            plt.subplot(411)
#            plt.plot(axis, output_0)
#            plt.title(str(i+1) + 'th trial\'s output_'+str(THRED)+\
#                      "_"+str(WIN)+"_"+str(thres)+"_"+str(thres_inver))
#        
#            axis = [j for j in range(len(output_1))]
#            plt.subplot(412)
#            plt.plot(axis, output_1)
#        
#            axis = [j for j in range(len(output_2))]
#            plt.subplot(413)
#            plt.plot(axis, output_2)
#        
#            axis = [j for j in range(len(gait_data[i][0]))]
#            plt.subplot(414)
#            plt.plot(axis, gait_data[i][0])
#        
#            plt.savefig("E:\EEGExoskeleton\EEGProcessor\Images_Subject"+\
#                        str(id_subject)+"\Subject"+\
#                        str(id_subject)+"_trail"+str(i+1)+"_"+\
#                        str(THRED)+"_"+str(WIN)+"_"+str(thres)+"_"+\
#                        str(thres_inver)+".png")
