%第1步

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
    path_file=[files(i).folder, '\', filename];     
    fid = fopen(path_file, 'r');
    start = 1;
    while start
        tline=fgetl(fid);
        line = line + 1; % 统计文件头的行数
        if strcmpi(tline,char('Frame Time: 0.00800000'))
            start = 0;
            temp = textread(filename,'','headerlines',line);
            save(filename,'temp','-ascii');
            count = count + 1
            line = 0;
        end
        
        if line > 1000
            start = 0; % 避免死循环
        end
    end
end
