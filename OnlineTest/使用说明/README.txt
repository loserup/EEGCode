每次重新进行一次训练前，先把之前的txt文件删除，然后将原始脑电数据文件bdf和原始步态数据bvh文件拷贝进该目录


提取用作在线训练分类器的特征以及csp矩阵
1、先运行bvh2txt.m
2、运行make_rawMat.m
3、运行rawdata_processor.py
4、运行eeg_window.py
5、运行eeg_CSP.py


在线测试
1、运行ActiView703-Lores.exe
2、运行onlineReadData.m


文件夹ExoGaitMonitorVer2
外骨骼上位机
1、回归原点后，点击侦听等待脑电matlab脚本onlineReadData.m运行与其通信


文件夹MouseKeyboardLibrary
采集训练原始数据时用于脑电信号的打标
1、动作捕捉软件点击录制后会弹出确认的对话框
2、脑电开始记录数据
3、打开路径\MouseKeyboardLibrary\SampleApplication\bin\Debug下的SampleApplication.exe
4、点击开始按钮
5、点击动作捕捉软件的OK按钮，开始记录步态数据，观察脑电下方是否打上开始标记（脑电设备需要插上并口）
6、采集完成后（期间鼠标不允许有任何点击操作），点击动作捕捉软件停止录制按钮，观察脑电下方是否打上结束标记
7、脑电记录pause按钮，并停止脑电，保存脑电信号


修改Log：
2018/6/7
CSP提取特征标准化