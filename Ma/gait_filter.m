% 将原始步态数据进行低通滤波

rawGait = load('E:\EEGExoskeleton\Dataset\Ma\20180829\RawMotion.mat');

fs = 121; % 动作捕捉系统采样频率121Hz
Wn = 1; % 截止频率1Hz
[b,a] = butter(4, 2*Wn/fs);

filteredMotion = cell(1,length(rawGait.rawMotion));
for cell_no = 1:length(rawGait.rawMotion)
    rawRightKnee = rawGait.rawMotion{1,cell_no}(:,1); % 右膝原始数据
    rawLeftKnee = rawGait.rawMotion{1,cell_no}(:,2); % 左膝原始数据
    
    % 对原始步态数据进行低通滤波
    rightKnee = filtfilt(b,a,rawRightKnee);
    leftKnee = filtfilt(b,a,rawLeftKnee);
    
    filteredMotion{1,cell_no} = [rightKnee leftKnee];
end

save('E:\EEGExoskeleton\Dataset\Ma\20180829\filteredMotion.mat','filteredMotion');