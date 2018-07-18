说明：识别有无跨越障碍意图的EEG二分类器
1、第1步运行make_rawMat.m
2、第2步运行rawdata_processor.py
3、第3步运行eeg_window.py
4、第4步运行eeg_CSP.py
5、第5步运行classifier.py

log：
EEG窗在1-4Hz进行带通滤波时分类器表现的性能最好，接近90%

扩充数据用不归一化，不网格搜索的方法也是过拟合