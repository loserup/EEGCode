function features = extractCSPFeatures(eeg, CSPMatrix, nFilterPairs)
%EXTRACTCSPFEATURE 此处显示有关此函数的摘要
%   此处显示详细说明
nTrials = length(eeg);
features = zeros(nTrials, 2*nFilterPairs+1);
Filter = CSPMatrix([1:nFilterPairs (end-nFilterPairs+1):end],:);

for t=1:nTrials     
    projectedTrial = Filter * eeg{t}.X;    
    variances = var(projectedTrial,0,2);    
    for f=1:length(variances)
        features(t,f) = log(variances(f));
    end
    features(t,end) = eeg{t}.y;    
end
end

