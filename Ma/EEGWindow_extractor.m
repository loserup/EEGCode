% 第4步：基于步态模式切换位置索引提取出带标签类别的EEG窗

eeg = load('E:\EEGExoskeleton\Dataset\Ma\20180829\cutEEG.mat');
gaitSwitch_index = load('E:\EEGExoskeleton\Dataset\Ma\20180829\gaitSwitchIndex.mat');
gait = load('E:\EEGExoskeleton\Dataset\Ma\20180829\filteredMotion.mat');

eeg = eeg.cutEEG;
gaitSwitch_index = gaitSwitch_index.gaitSwitchIndex;
gait = gait.filteredMotion;

fs_eeg = 512; % EEG sampling rate (Hz)
fs_gait = 121; % gait sampling rate (Hz)
eeg_winWidth = 384; % the width of eeg window (384 sample points = 750 ms)
gait_winWidth = fs_gait / fs_eeg * eeg_winWidth; % the width of eeg window in gait data

output = {}; % 储存带标签的EEG窗，最后输出为数据文件
count = 1; % output存入数据的计数器
for i = 1:length(gait)
    % 有步态切换意图的索引，分别为：静止进正常行走，正常行走进上下楼梯，上下楼梯进正常行走，正常行走进静止
    yep_index = gaitSwitch_index{i,1}; 
    % 没有步态切换意图的索引，定为有切换意图的点的中间点（因为中间点一般在该步态进行中间段）
    % 分别为：正常行走进上下楼梯间中点（正常行走段），上下楼梯进正常行走间中点（上下楼梯段），
    % 正常行走进静止间中点（正常行走段），静止到步态最后一个点间中点（静止段）
    nop_index = [(yep_index(1)+yep_index(2))/2, (yep_index(2)+yep_index(3))/2, (yep_index(3)+yep_index(4))/2, (yep_index(4)+length(gait{1,i}))/2];
    nop_index = round(nop_index); % 索引取整
    
    % 将步态索引转换为EEG索引
    eeg_yep_index = yep_index * fs_eeg / fs_gait;
    eeg_yep_index = round(eeg_yep_index);
    eeg_nop_index = nop_index * fs_eeg / fs_gait;
    eeg_nop_index = round(eeg_nop_index);
    
    for j = 1:length(eeg_yep_index)
        % 切换点往前取窗作为有切换意图窗，标签为1
        yep_eegWin = eeg{1,i}(:,eeg_yep_index(j)-eeg_winWidth+1:eeg_yep_index(j));
        output{count,1} = yep_eegWin;
        output{count,2} = 1;
        % 中间点往两边取窗作为无切换意图窗，标签为-1
        nop_eegWin = eeg{1,i}(:,eeg_nop_index(j)-eeg_winWidth/2+1:eeg_nop_index(j)+eeg_winWidth/2);
        output{count+1,1} = yep_eegWin;
        output{count+1,2} = -1;
        % 更新计数器
        count = count + 2;
    end
end

save('E:\EEGExoskeleton\Dataset\Ma\20180829\labeledEEG.mat','output');

% %% 划取样本窗的示意图
% test = gait{1,1}(:,2);
% index = gaitSwitch_index{1,1};
% figure % 有切换步态意图的窗的示意图
% hold on 
% plot(1:length(test), test)
% plot(index, test(index), 'k*')
% for i = 1:length(index)
%     rectangle('Position',[index(i) - gait_winWidth, test(index(i)), gait_winWidth, 40], 'EdgeColor','r')
% end
% 
% no_index = [(index(1)+index(2))/2, (index(2)+index(3))/2, (index(3)+index(4))/2, (index(4)+length(test))/2];
% no_index = round(no_index); % 无切换意图窗索引
% % figure % 无切换步态意图的窗的示意图
% % hold on
% % plot(1:length(test), test)
% plot(no_index, 50, 'k*')
% for i = 1:length(no_index)
%     rectangle('Position',[no_index(i) - gait_winWidth/2, 30, gait_winWidth, 40], 'EdgeColor','g')
% end
