% 第2步

% 将某一受试对象（对应需要设置受试对象的编号id_subject）的
% EEG的bdf格式的原始数据和转成txt格式的步态数据分别存储为mat格式文件

% 根据受试对象进行的试验次数设置num_sample
% 只选取左右膝的数据，之后判断那条腿先迈障碍交给Python判断
% 列数对应：右髋-8；右膝-11；左髋-17；左膝20

% 最后生成的mat文件为元组，每一个成员对应受试对象一次试验的数据 

eeg_files = dir('*.bdf');
gait_files = dir('*.txt');
num_sample = length(eeg_files); % 样本文件数

rawEEG = cell(1,num_sample);
rawMotion = cell(1,num_sample);

count = 0; % 进度条
for n = 1:num_sample
    rawEEG{1,n} = eeg_read_bdf(eeg_files(n).name,'all','n');
    temp = load(gait_files(n).name);
    rawMotion{1,n} = [temp(:,11) temp(:,20)];% 动作数据只采用左右膝的数据
    count = count + 1
end

save('RawEEG.mat','rawEEG');
save('RawMotion.mat','rawMotion');

