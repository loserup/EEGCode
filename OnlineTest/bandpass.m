% 将data以up和low为上下截止频率进行带通滤波
function re = bandpass(data,up,low)
warning off

fs = 512; % EEG采集频率
row = size(data,1); % EEG数据通道数
col = size(data,2); % EEG窗长

Wn = [2*up/fs 2*low/fs];
[b,a] = butter(4,Wn,'bandpass');
data_filtered = zeros(row, col);

for i = 1 : row
    % data_filtered(i,:) = filter(b,a, data(i,:));
    data_filtered(i,:) = filtfilt(b,a, data(i,:));
end

re = data_filtered;

end