% 输出命令的二次滤波
function re = onlinefilters(out_store)

BACK = 20; % 回溯点数
THRED = 18; % 命令决策阈值 % 在回溯点数中有意图窗个数超过该阈值则输出有意图命令
thres = 5; % 当连续为跨越意图（1）的个数不超过阈值thres时，全部变成-1
thres_inver = 15; % 反向滤波阈值：将连续跨越意图间的短-1段补成1
                
count_filter = 0;
out_length = length(out_store);
output_1 = [];

%% 一次滤波
% 一次滤波：伪在线向前取BACK个窗的标签，
% BACK个窗中标签个数超过阈值THRED则输出跨越命令
for i = BACK : out_length
   for j  = i - BACK + 1 : i
      if out_store(j) == 1
          count_filter = count_filter + 1;
      else
          continue
      end
   end
   
   if count_filter >= THRED
       output_1 = [output_1 1];
       count_filter = 0;
   else
       output_1 = [output_1 -1];
       count_filter = 0;
   end
end

%% 二次滤波
% 反向滤波：当连续为无跨越意图（-1）的个数不超过阈值thres_inter时，全部变成1
output_2 = output_1;
count_filter = 0;

for i = 1 : length(output_1)
    if output_2(i) == -1
        count_filter = count_filter + 1;
    else
        if count_filter < thres_inver
            for j = 1 : count_filter
                output_2(i-j) = 1;
            end
            count_filter = 0;
        else
            count_filter = 0;
            continue
        end
    end
end
output_2(end) = -1;

%% 正向滤波
% 当连续为跨越意图（1）的个数不超过阈值thres时，全部变成-1

count_filter = 0;
for i = 1 : length(output_2)
    if output_2(i) == 1
        if i == length(output_2) - 1
            for j = 1 : count_filter
                output_2(i-j) = -1;
            end
        else
            count_filter = count_filter + 1;
        end
    else
        if count_filter < thres
            for j = 1 : count_filter
                output_2(i-j) = -1;
            end
            count_filter = 0;
        else
            count_filter = 0;
        end
    end
end

re = output_2;

end