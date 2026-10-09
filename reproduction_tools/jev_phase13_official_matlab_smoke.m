% Run untouched repository official metric functions in native MATLAB.
kit = getenv('JEV_MATLAB_KIT');
output = getenv('JEV_MATLAB_OUTPUT');
assert(~isempty(kit) && ~isempty(output));
addpath(genpath(kit));
gt = [(1:4)', ones(4,1), repmat([10,10,20,20,1,1,1],4,1)];
cases = {gt, gt(1:3,:), [gt; 1,2,100,100,20,20,1,1,1], gt};
cases{4}(3:4,2) = 2;
names = {'perfect','miss','false_positive','id_split'};
expectedMOTA = [100,75,75,75];
expectedIDF1 = [100,600/7,800/9,50];
values = struct();
for k = 1:numel(cases)
    [mets,~,info] = CLEAR_MOT_HUN(gt,cases{k},0.5,0);
    ids = IDmeasures(gt,cases{k},0.5,0);
    assert(abs(mets(12)-expectedMOTA(k))<1e-7);
    assert(abs(ids.IDF1-expectedIDF1(k))<1e-7);
    values.(names{k}) = struct('MOTA',mets(12),'IDF1',ids.IDF1,...
                              'CLEAR',info,'Identity',ids);
end
report = struct('status','PASS','release',version('-release'),...
                'version',version,'fixtures',values,...
                'CLEAR_function',which('CLEAR_MOT_HUN'),...
                'Identity_function',which('IDmeasures'),...
                'MinCostMatching_MEX',which('MinCostMatching'),...
                'clearMOT_MEX',which('clearMOTMex'),...
                'costBlock_MEX',which('costBlockMex'));
fid = fopen(output,'w'); assert(fid>0);
fprintf(fid,'%s\n',jsonencode(report)); fclose(fid);
fprintf('NATIVE_OFFICIAL_MATLAB_METRIC_SMOKE_PASS\n');
