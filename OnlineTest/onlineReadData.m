function onlineReadData()

close all
clear all
clc
warning off

% 在线实验参数设置
win_len = 384; % 窗长，单位：采样点;【必须是4的整数】
csp = load('csp.mat');
csp = csp.csp; % 获取指定受试对象的CSP投影矩阵
interval = 28; % 窗分类间隔：每采样26个点后进行一次取窗分类 % 384点为750ms，26个点为50ms

% 输出数据初始化
global run; run = true; % 是否进行循环
data = []; % 存放实时EEG
count = 0; % 采样点的计数器
out_store = []; % 记录输出指令
output_cmd = []; % 二次滤波的输出指令


% TCPIP 参数设置
% configure % the folowing 4 values should match with your setings in Actiview and your network settings
% Decimation选择“1/4”;
port = 778;                % the port that is configured in Actiview , delault = 8888
port2 = 8080;
ipadress = 'localhost';    % the ip adress of the pc that is running Actiview
Channels = 32;             % set to the same value as in Actiview "Channels sent by TCP"
Samples = 4;               % set to the same value as in Actiview "TCP samples/channel" % Samples = fs/Channels/4
words = Channels*Samples;
data_current = zeros(Samples, Channels); % 本次采样获取的EEG数据

while run
    tcpipClient = tcpip(ipadress, port, 'NetworkRole', 'client');
    set(tcpipClient,'InputBufferSize',words*9); % input buffersize is 3 times the tcp block size % 1 word = 3 bytes
    set(tcpipClient,'Timeout',5);
    fopen(tcpipClient);
    
    tcpipClient2 = tcpip('172.20.15.186', port2, 'NetworkRole', 'client');
    set(tcpipClient2,'InputBufferSize',words*9); % input buffersize is 3 times the tcp block size % 1 word = 3 bytes
    set(tcpipClient2,'Timeout',5); 
    fopen(tcpipClient2);
    
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

    data_current_t = data_current'; % 除与不除256，得到的特征是相同的
    
    if length(data) ~= win_len
        % 刚开始录入数据
        data = [data data_current_t];
    else
        % 录入数据已达到设定窗长
        % 将实时EEG窗做成FIFO队列使得该窗长永远为win_len
        % 将前4列数据pop掉，在最后加入新的4列数据
        data = cat(2, data(:,5:end), data_current_t); 
        
        save('data.mat', 'data');
        pyObj = py.onlineClassifier.OnlineClassifier(); 
        
        if length(out_store) ~= 60
            % 输出命令序列未达到设置要求
            out_store = [out_store str2double(char(pyObj.outputCmd()))];
        else
            out_store = cat(2, out_store(2:end), str2double(char(pyObj.outputCmd())));
            
            output_cmd = onlinefilters(out_store) % 对out_store进行二次滤波
            
            if length(find(output_cmd(end) == 1)) == 1
                %当输出命令最后20个全是1时给外骨骼传1命令
                fwrite(tcpipClient2,'1');
            else
                fwrite(tcpipClient2,'-1');
            end

        end
    end
        

    fclose(tcpipClient);
    delete(tcpipClient);
    fclose(tcpipClient2);
    delete(tcpipClient2);
    
end

% save('data_current_t.mat', 'data_current_t');
% save('data.mat', 'data');
% save('output_cmd.mat', 'output_cmd');
% save('count.mat','count');
% save('feat.mat','feat');
% save('out_store.mat','out_store');
% save('time.mat','time');
% save('count_win.mat','count_win');

%关闭tcpip
fclose(tcpipClient);
delete(tcpipClient);
fclose(tcpipClient2);
delete(tcpipClient2);

end