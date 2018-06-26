说明：识别有无跨越和跨越高度1,3,5的EEG四分类器
1、第1步运行make_rawMat.m
2、第2步运行rawdata_processor.py
3、第3步运行eeg_window.py
4、第4步运行eeg_CSP.py
5、第5步运行classifier.py

用同一个窗的三种频带分别做一个窗，一个窗变成3个窗
4类共对应有12个CSP矩阵；
每种频带都提了一次特征，即每类提了三次特征，即特征数目是MultiEEGClassifier_4class_Ver1的三倍
可惜，分类准确率也是70%，看起来比MultiEEGClassifier_4class_Ver1的性能还稍差一些