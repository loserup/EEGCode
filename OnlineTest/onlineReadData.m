function onlineReadData()

close all
clear all
clc
warning off

% 在线实验参数设置
win_len = 384; % 窗长，单位：采样点
csp = load('csp.mat');
csp = csp.csp; % 获取指定受试对象的CSP投影矩阵
interval = 28; % 窗分类间隔：每采样26个点后进行一次取窗分类 % 384点为750ms，26个点为50ms

% 输出数据初始化
global run; run = true; % 是否进行循环
data_history = []; % 保留EEG历史信息，时间越长，数据量越大，以后需要添加清空远古历史的功能
count = 0; % 采样点的计数器
out_store = []; % 记录输出指令


% TCPIP 参数设置
% configure % the folowing 4 values should match with your setings in Actiview and your network settings
% Decimation选择“1/4”;
port = 778;                % the port that is configured in Actiview , delault = 8888
port2 = 4484;
ipadress = 'localhost';    % the ip adress of the pc that is running Actiview
Channels = 32;             % set to the same value as in Actiview "Channels sent by TCP"
Samples = 4;               % set to the same value as in Actiview "TCP samples/channel" % Samples = fs/Channels/4
words = Channels*Samples;
data_current = zeros(Samples, Channels); % 本次采样获取的EEG数据

% open tcp connection % 与脑电采集服务器的通信Socket
tcpipClient = tcpip(ipadress, port, 'NetworkRole', 'client');
set(tcpipClient,'InputBufferSize',words*9); % input buffersize is 3 times the tcp block size % 1 word = 3 bytes
set(tcpipClient,'Timeout',5);
% open tcp connection % 与外骨骼上位机的通信Socket
% tcpipClient2 = tcpip('172.20.10.132', port2, 'NetworkRole', 'client');
% set(tcpipClient2,'InputBufferSize',words*9); % input buffersize is 3 times the tcp block size % 1 word = 3 bytes
% set(tcpipClient2,'Timeout',5);
try
    fopen(tcpipClient);
%     fopen(tcpipClient2);
catch
    disp('Actiview is unreachable please check if Actiview is running on the specified ip address and port number');
    run = false;
end

while run
    
    % 读取每次tcpip传送的数值
    [rawData,temp,msg] = fread(tcpipClient,[3 words],'uint8');
    if temp ~= 3*words
        run = false;
        disp(msg);
        disp('Is Actiview running with the same settings as this example?');
        break
    end
       
    % reorder bytes from tcp stream into 32bit unsigned words
    % normaldata = rawData(3,:)*(256^3) + rawData(2,:)*(256^2) + rawData(1,:)*256 + 0;
    % 2018-4-27-既然TCP传输数据比Labview保存数据多256倍，就把上式除以256得到下式 (但是会得到错误的正负符号)
    normaldata = rawData(3,:)*(256^3) + rawData(2,:)*(256^2) + rawData(1,:)*256 + 0;
    % reorder the channels into a array [samples channels]
    i = 0 : Channels : words-1; % words-1 because the vector starts at 0
    for d = 1 : Channels
        data_current(1:Samples,d) = typecast(uint32(normaldata(i+d)),'int32');   %create a data struct where each channel has a seperate collum     
    end

    data_current = data_current; % 除与不除256，得到的特征是相同的
    data_history = [data_history;data_current];
    count = count + 4; % EEG 512Hz的采样频率的话每次循环会读入4个点
    
    if count > win_len && mod(count,interval) == 0
        data = data_history';
        data = data(:,count-win_len+1:count);
        save('data.mat', 'data');
        pyObj = py.onlineClassifier.OnlineClassifier(); 

        out_store = [out_store str2double(char(pyObj.outputCmd()))]; % Python原始输入数据带属性，先转string再转数字去掉属性
        out_length = length(out_store);
        
        if out_length > 20
            output_cmd = onlinefilters(out_store) % 对out_store进行二次滤波
        end        
    end
end


data_history = (data_history(:,1:32))';

% save('data_current.mat', 'data_current');
save('data_history.mat', 'data_history');
save('output_cmd.mat', 'output_cmd');
% save('count.mat','count');
% save('feat.mat','feat');
save('out_store.mat','out_store');
% save('time.mat','time');
% save('count_win.mat','count_win');


%关闭tcpip
fclose(tcpipClient);
delete(tcpipClient);
% fclose(tcpipClient2);
% delete(tcpipClient2);

end