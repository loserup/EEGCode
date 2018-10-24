% 基于低通滤波后的步态数据寻找步态模式切换的位置索引·

filteredGait = load('E:\EEGExoskeleton\Dataset\Ma\20180829\filteredMotion.mat');

gaitSwitchIndex = cell(length(filteredGait.filteredMotion),2);
for cell_no = 1:length(filteredGait.filteredMotion)
    %% 获取右膝峰值点和谷值点索引
    rightKnee = filteredGait.filteredMotion{1,cell_no}(:,1); % 低通滤波过的右膝步态数据
       
    rightIndMax=find(diff(sign(diff(rightKnee)))<0)+1;   % 获得右膝局部最大值的位置
    rightIndMin=find(diff(sign(diff(rightKnee)))>0)+1;   % 获得右膝局部最小值的位置  

    % 去掉小于30度的峰值点索引
    index = [];
    for i = 1:length(rightIndMax)
       if rightKnee(rightIndMax(i)) < 30
          % 获取小于30度的峰值点索引
          index = horzcat(index,i);
       end
    end
    rightIndMax(index) = []; % 将小于30度的峰值点去掉
    
    % 获取上述峰值点前的谷值点索引
    index = [];
    for i = 1:length(rightIndMax)
       for j = 1:length(rightIndMin)
           if rightIndMin(j) > rightIndMax(i)
               index = horzcat(index,rightIndMin(j-1));
               break;
           else
               continue;
           end
       end
    end
    index = horzcat(index,rightIndMin(j)); % 认为满足上述要求的谷值点的后一个谷值点是由正常行走切换至静止的谷值点
    rightIndMin = index;
    
    % 只留下步态模式切换处的谷值点
    index = [1]; % 认为一个谷值点是静止切换至正常行走的位置
    for i = 1:length(rightIndMax)
        if i+1 < length(rightIndMax)
            % 避免索引超出矩阵维度
            if rightKnee(rightIndMax(i+1)) > 70 && rightKnee(rightIndMax(i)) < 60
                % 当后一个膝关节角度大于70而当前膝关节角度小于60时，说明正在进行由正常行走步态切换到上下楼梯步态
                index = horzcat(index,i+1);
            elseif rightKnee(rightIndMax(i+1)) < 60 && rightKnee(rightIndMax(i)) > 70
                % 当后一个膝关节角度小于60而当前膝关节角度大于70时，说明正在进行由上下楼梯步态切换到正常行走步态
                index = horzcat(index,i+1);
            end
        else
            break;
        end
    end
    index = horzcat(index, length(rightIndMin)); % 认为最后一个谷值点是正常行走切换至静止的位置
    rightIndMin = rightIndMin(index);

    %% 获取左膝峰值点和谷值点索引
    leftKnee = filteredGait.filteredMotion{1,cell_no}(:,2);  % 低通滤波过的左膝步态数据
    
    leftIndMax=find(diff(sign(diff(leftKnee)))<0)+1;   % 获得左膝局部最大值的位置
    leftIndMin=find(diff(sign(diff(leftKnee)))>0)+1;   % 获得左膝局部最小值的位置  
    
    % 去掉小于30度的峰值点索引
    index = [];
    for i = 1:length(leftIndMax)
       if leftKnee(leftIndMax(i)) < 30
          % 获取小于30度的峰值点索引
          index = horzcat(index,i);
       end
    end
    leftIndMax(index) = []; % 将小于30度的峰值点去掉
    
    % 获取上述峰值点前的谷值点索引
    index = [];
    for i = 1:length(leftIndMax)
       for j = 1:length(leftIndMin)
           if leftIndMin(j) > leftIndMax(i)
               index = horzcat(index,leftIndMin(j-1));
               break;
           else
               continue;
           end
       end
    end
    index = horzcat(index,leftIndMin(j)); % 认为满足上述要求的谷值点的后一个谷值点是由正常行走切换至静止的谷值点
    leftIndMin = index;
    
    % 只留下步态模式切换处的谷值点
    index = [1]; % 认为一个谷值点是静止切换至正常行走的位置
    for i = 1:length(leftIndMax)
        if i+1 < length(leftIndMax)
            % 避免索引超出矩阵维度
            if leftKnee(leftIndMax(i+1)) > 70 && leftKnee(leftIndMax(i)) < 60
                % 当后一个膝关节角度大于70而当前膝关节角度小于60时，说明正在进行由正常行走步态切换到上下楼梯步态
                index = horzcat(index,i+1);
            elseif leftKnee(leftIndMax(i+1)) < 60 && leftKnee(leftIndMax(i)) > 70
                % 当后一个膝关节角度小于60而当前膝关节角度大于70时，说明正在进行由上下楼梯步态切换到正常行走步态
                index = horzcat(index,i+1);
            end
        else
            break;
        end
    end
    index = horzcat(index, length(leftIndMin)); % 认为最后一个谷值点是正常行走切换至静止的位置
    leftIndMin = leftIndMin(index);
    
    % 画图检查峰谷值点查找是否正确
    figure
    hold on
    plot(rightKnee)
    plot(rightIndMax,rightKnee(rightIndMax),'k*')
    plot(rightIndMin,rightKnee(rightIndMin),'r^') 
   
    figure
    hold on
    plot(leftKnee)
    plot(leftIndMax,leftKnee(leftIndMax),'k*')
    plot(leftIndMin,leftKnee(leftIndMin),'r^') 

    %% 保存步态切换索引位置
    if rightIndMin(1) < leftIndMin(1)
        % 如果右腿先跨上楼梯，则保存右腿的索引作为指示步态切换的索引
        gaitSwitchIndex{cell_no,1} = rightIndMin;
        gaitSwitchIndex{cell_no,2} = 1; % label 1 指示为右腿
    else
        % 如果左腿先跨上楼梯，则保存左腿的索引作为指示步态切换的索引
        gaitSwitchIndex{cell_no,1} = leftIndMin;
        gaitSwitchIndex{cell_no,2} = 2; % label 2 指示为右腿
    end
end

save('E:\EEGExoskeleton\Dataset\Ma\20180829\gaitSwitchIndex.mat','gaitSwitchIndex');


