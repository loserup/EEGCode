filteredGait = load('E:\EEGExoskeleton\Dataset\Ma\20180829\filteredMotion.mat');

%% 获取右膝峰值点和谷值点索引
for cell_no = 1:length(filteredGait.filteredMotion)
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
    rightIndMin = index;
    
    % 画图检查峰谷值点查找是否正确
    figure
    hold on
    plot(rightKnee)
    plot(rightIndMax,rightKnee(rightIndMax),'k*')
    plot(rightIndMin,rightKnee(rightIndMin),'r^') 
end

%% 获取左膝峰值点和谷值点索引
for cell_no = 1:length(filteredGait.filteredMotion)
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
    leftIndMin = index;
    
    % 画图检查峰谷值点查找是否正确
    figure
    hold on
    plot(leftKnee)
    plot(leftIndMax,leftKnee(leftIndMax),'k*')
    plot(leftIndMin,leftKnee(leftIndMin),'r^') 
end


