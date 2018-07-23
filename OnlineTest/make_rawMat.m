%% 第1步

%批量修改文件格式
%bvh->txt
%把该文件拷贝到相应要修改的文件所在文件夹里即可
%bvh是步态数据文件

clear, clc;
files=dir('*.bvh');
files_count=length(files);
for i=1:files_count
    oldfilename=files(i).name;
    len=length(oldfilename);
    newfilename=[oldfilename(1:len-4), '.txt'];
    eval(['!rename' 32 oldfilename 32 newfilename]);
    i=i
end


% 删掉转换成txt的bvh文件的文件头，只留下数值

files=dir('.\*.txt');
line = 0;
count = 0; % 进度条
for i=1:length(files)
    filename=files(i).name;
    temp = textread(filename,'','headerlines',312);
    save(filename,'temp','-ascii');
    count = count + 1
    line = 0;
end


%% 第2步

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