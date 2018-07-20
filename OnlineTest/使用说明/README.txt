每次重新进行一次训练前，先把之前的txt文件删除，然后将bdf和bvh文件拷贝进该目录

0、将训练好的模型命名为SVM.m拷贝进该目录

提取用作在线训练分类器的特征以及csp矩阵
1、先运行bvh2txt.m
2、运行make_rawMat.m
3、运行rawdata_processor.py
4、运行eeg_window.py
5、运行eeg_CSP.py


在线测试
1、运行ActiView703-Lores.exe
2、启动ExoSocket\Server\bin\Debug下的Server.exe，点击【启动服务器】按钮
3、运行onlineReadData.m


修改Log：
2018/6/7
CSP提取特征标准化