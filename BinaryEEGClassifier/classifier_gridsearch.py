# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:21:17 2018

@author: Long

% 说明:基于网格搜索训练的SVM分类器

% 第五步

% 输入数据最后一列为标签，0表示无意图，1表示有意图
% 数据其余列为特征值
"""
# In[1]:
import scipy.io as sio

from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.externals import joblib

# In[2]:
id_subject = 4 # 【受试者的编号】
if id_subject < 10:
    feats_mat = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0'+\
                            str(id_subject)+'_Data\\Subject_0'+\
                            str(id_subject)+'_features.mat')

else:
    feats_mat = sio.loadmat('E:\\EEGExoskeleton\\Data\\Subject_0'+\
                            str(id_subject)+'_Data\\Subject_'+\
                            str(id_subject)+'_features.mat')


feats_all = feats_mat['features']

# In[]
### 基于参数搜索的评估方法
#param_fixed = {'kernel': 'rbf'} # 设置SVM参数
parameter_grid = [  {'kernel': ['linear'], 'C': [10 ** x for x in range(-1, 4)]},
                    {'kernel': ['poly'], 'degree': [2, 3]},
                    {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [10 ** x for x in range(-1, 4)]},
                 ]

X = feats_all[:,:-1]
y = feats_all[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("\n#### Searching optimal hyperparameters for precision")

classifier = grid_search.GridSearchCV(svm.SVC(), 
             parameter_grid, cv=5, scoring="accuracy")
classifier.fit(X_train, y_train)
#classifier.fit(X,y)

print("\nScores across the parameter grid:")
for params, avg_score, _ in classifier.grid_scores_:
    print(params, '-->', round(avg_score, 3))

print("\nHighest scoring parameter set:", classifier.best_params_)

y_true, y_pred = y_test, classifier.predict(X_test)
print("\nFull performance report:\n")
print(classification_report(y_true, y_pred))

if id_subject < 10:
    joblib.dump(classifier, "Subject_0"+str(id_subject)+"_gridsearch_SVM.m")
else:
    joblib.dump(classifier, "Subject_"+str(id_subject)+"_gridsearch_SVM.m")
        

#SVM = SVC(random_state=0, class_weight='balanced', kernel=param_fixed['kernel'])

#n_feature_kept = 8 # LDA降维目标维数
#lda = LinearDiscriminantAnalysis(n_components=n_feature_kept)

#classifier = grid_search.GridSearchCV(estimator=SVM, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')
#classifier.fit(X_train, y_train)
#
#y_true, y_pred = y_test, classifier.predict(X_test)
#print(classification_report(y_true,y_pred))
#pipeline = make_pipeline(StandardScaler(), lda, clf_grid)
#pipeline = make_pipeline(StandardScaler(), clf_grid)

#warnings.filterwarnings('ignore', message='Changing the shape of non-C contiguous array') # 禁止joblib警告。
#pipeline.fit(X=X_train, y=y_train)

#y_pred_train = pipeline.decision_function(X_train)
#y_pred_test_SVC = pipeline.decision_function(X_test)

#def get_threshold_metrics(y_true, y_pred):
#    """
#    roc_curve返回假阳性率fpr，真阳性率tpr和分类阈值threshold
#    """
#    roc_columns = ['fpr', 'tpr', 'threshold']
#    roc_items = zip(roc_columns, roc_curve(y_true, y_pred)) 
#    roc_df = pd.DataFrame.from_items(roc_items)
#    auroc = roc_auc_score(y_true, y_pred) # 计算ROC曲线下AUC
#    return {'auroc': auroc, 'roc_df': roc_df}
#
#metrics_train = get_threshold_metrics(y_train, y_pred_train)
#metrics_test_SVC = get_threshold_metrics(y_test, y_pred_test_SVC)
#
#plt.style.use('seaborn-notebook')
#plt.figure()
#for label, metrics in ('Training', metrics_train), ('Testing', metrics_test_SVC):
#    roc_df = metrics['roc_df']
#    plt.plot(roc_df.fpr, roc_df.tpr,
#        label='{} (AUROC = {:.1%})'.format(label, metrics['auroc']))
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('RBF-SVM : Predicting TP53 mutation from gene expression (ROC curves)')
#plt.legend(loc='lower right')

# In[]
### 可能是错的评估方法
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
#    print('Accuracy: %0.4f (± %0.4f)' % (accuracy.mean(),accuracy.std()))
#
#    print('F1: %0.4f (± %0.4f)' % (f1.mean(),f1.std()))
#    print('Precision: %0.4f (± %0.4f)' % (precision.mean(),precision.std()))
#    print('Recall: %0.4f (± %0.4f)' % (recall.mean(),recall.std()))
#
#    print('\n')
#        
#ave_accuracy = np.array(ave_accuracy)
#ave_f1 = np.array(ave_f1)
#ave_precision = np.array(ave_precision)
#ave_recall = np.array(ave_recall)
#
#print ('\nAverage Accuracy: %0.4f (± %0.4f)' % (ave_accuracy.mean(),ave_accuracy.std()))
#print ('Average F1: %0.4f (± %0.4f)' % (ave_f1.mean(),ave_f1.std()))
#print ('Average Precision: %0.4f (± %0.4f)' % (ave_precision.mean(),ave_precision.std()))
#print ('Average Recall: %0.4f (± %0.4f)\n' % (ave_recall.mean(),ave_recall.std()))

# In[3]:
## 混淆矩阵
#
#from sklearn.metrics import confusion_matrix
#
## Show confusion matrix
#def plot_confusion_matrix(confusion_mat):
#    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray_r)
##    plt.title('Confusion matrix')
#    plt.colorbar()
#    tick_marks = np.arange(2)
#    plt.xticks(tick_marks, tick_marks)
#    plt.yticks(tick_marks, tick_marks)
#    plt.ylabel('True label',FontSize=18)
#    plt.xlabel('Predicted label',FontSize=18)
#    plt.savefig("E:\EEGExoskeleton\Data\Images_Subject"+\
#                str(id_subject)+"\Subject"+\
#                str(id_subject)+"_confumatrix.eps")
#    plt.show()
#    
#    
#pred = []
#for i in range(len(feats)):
#    pred.append(classifier.predict(feats[i].reshape(1,8)))
#true = list(labels)
#
#confusion_mat = confusion_matrix(true, pred)
#plot_confusion_matrix(confusion_mat)
#
#count = 0
#for i in range(len(labels)):
#    if int(labels[i]) == 1:
#        count = count + 1