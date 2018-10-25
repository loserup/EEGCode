% 第3步：将EEG信号剪裁至与步态信号同步

rawEEG = load('E:\EEGExoskeleton\Dataset\Ma\20180829\RawEEG.mat');
rawEEG = rawEEG.rawEEG;
cutEEG = cell(1,length(rawEEG));

for i = 1:length(rawEEG)
    flag_channel = rawEEG{1,i}(end,:); % 原始EEG数据中的打标通道数据
    
%     %% 绘制打标通道数据观察所有数据是否打标正确
%     figure
%     plot(1:length(flag_channel),flag_channel)
    
    %% 通过上升沿找两次打标位置
    temp = flag_channel(1);
    for j = 2:length(flag_channel)
        if flag_channel(j) <= temp
            temp = flag_channel(j);
            continue;
        else
            temp = flag_channel(j);
            break;
        end
    end
    
    firstFlag = j; % 第一次打标位置
    
    for j = (firstFlag+1):length(flag_channel)
        if flag_channel(j) <= temp
            temp = flag_channel(j);
            continue;
        else
            break;
        end
    end
    
    SecondFlag = j; % 第二次打标位置
    
    %% 截取两个打标位置间的EEG数据
    cutEEG{1,i} = rawEEG{1,i}(1:32,firstFlag:SecondFlag);
end

save('E:\EEGExoskeleton\Dataset\Ma\20180829\cutEEG.mat','cutEEG');