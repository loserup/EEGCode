% 基于步态模式切换位置索引提取出带标签类别的EEG窗

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
