 % 第一步

% 将某一受试对象（对应需要设置受试对象的编号id_subject）的
% EEG的bdf格式的原始数据和转成txt格式的步态数据分别存储为mat格式文件

% 根据受试对象进行的试验次数设置num_sample
% 只选取左右膝的数据，之后判断那条腿先迈障碍交给Python判断
% 列数对应：右髋-8；右膝-11；左髋-17；左膝20

% 最后生成的mat文件为元组，每一个成员对应受试对象一次试验的数据 

id_subject = 1; % 受试对象ID号
%motion_flag = 11; 

if id_subject < 10
    path = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\raw_EEG\*.bdf'];
else
    path = ['E:\EEGExoskeleton\Dataset\Subject_' num2str(id_subject) '\raw_EEG\*.bdf'];
end
num_sample = length(dir(path)); % 样本文件数


rawEEG = cell(1,num_sample);
rawMotion = cell(1,num_sample);

if id_subject < 10
    for n = 1:num_sample
        if n < 10
            motion_filename = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\txt_Gait\0' num2str(n) 'Char00_biped.txt']; % 读取步态文本文件的文件名
            eeg_filename = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\raw_EEG\0' num2str(n) '.bdf']; % 读取EEG文件的文件名
        else
            motion_filename = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\txt_Gait\' num2str(n) 'Char00_biped.txt'];
            eeg_filename = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\raw_EEG\' num2str(n) '.bdf'];
        end
        rawEEG{1,n} = eeg_read_bdf(eeg_filename,'all','n');
        temp = load(motion_filename);
        rawMotion{1,n} = [temp(:,11) temp(:,20)];% 动作数据只采用左右膝的数据
    end
else
    for n = 1:num_sample
        if n < 10
            motion_filename = ['E:\EEGExoskeleton\Dataset\Subject_' num2str(id_subject) '\txt_Gait\0' num2str(n) 'Char00_biped.txt']; % 读取步态文本文件的文件名
            eeg_filename = ['E:\EEGExoskeleton\Dataset\Subject_' num2str(id_subject) '\raw_EEG\0' num2str(n) '.bdf']; % 读取EEG文件的文件名
        else
            motion_filename = ['E:\EEGExoskeleton\Dataset\Subject_' num2str(id_subject) '\txt_Gait\' num2str(n) 'Char00_biped.txt'];
            eeg_filename = ['E:\EEGExoskeleton\Dataset\Subject_' num2str(id_subject) '\raw_EEG\' num2str(n) '.bdf'];
        end
        rawEEG{1,n} = eeg_read_bdf(eeg_filename,'all','n');
        temp = load(motion_filename);
        rawMotion{1,n} = [temp(:,11) temp(:,20)];% 动作数据只采用左右膝的数据
    end
end

if id_subject < 10
    save_eeg_filename = ['E:\EEGExoskeleton\EEGProcessor\Subject_0' num2str(id_subject) '_Data\Subject_0' num2str(id_subject) '_RawEEG.mat'];
    save_motion_filename = ['E:\EEGExoskeleton\EEGProcessor\Subject_0' num2str(id_subject) '_Data\Subject_0' num2str(id_subject) '_RawMotion.mat'];
    save(save_eeg_filename,'rawEEG');
    save(save_motion_filename,'rawMotion');
else
    save_eeg_filename = ['E:\EEGExoskeleton\EEGProcessor\Subject_' num2str(id_subject) '_Data\Subject_' num2str(id_subject) '_RawEEG.mat'];
    save_motion_filename = ['E:\EEGExoskeleton\EEGProcessor\Subject_' num2str(id_subject) '_Data\Subject_' num2str(id_subject) '_RawMotion.mat'];
    save(save_eeg_filename,'rawEEG');
    save(save_motion_filename,'rawMotion');
end
