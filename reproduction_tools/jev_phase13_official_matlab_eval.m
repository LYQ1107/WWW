% Call the unchanged official evaluator on converter-compatible inputs.
cfg = jsondecode(fileread(getenv('JEV_MATLAB_CONFIG')));
cd(cfg.kit); addpath(genpath(cfg.kit));
results = cell(numel(cfg.jobs),1);
for k = 1:numel(cfg.jobs)
    job = cfg.jobs(k);
    [~,mets,ids,info,res] = evaluateTracking(job.scene,job.prediction,...
                            job.groundtruth,job.gt_directory,cfg.benchmark);
    results{k} = struct('scene',job.scene,'convention',job.convention,...
                       'Identity',ids,'CLEAR',info,'metrics',mets,...
                       'official_results',res);
end
report = struct('status','COMPLETE','version',version,'release',version('-release'),...
                'native_official_function',which('evaluateTracking'),...
                'results',{results});
fid = fopen(cfg.output,'w'); assert(fid>0);
fprintf(fid,'%s\n',jsonencode(report)); fclose(fid);
fprintf('NATIVE_OFFICIAL_MATLAB_VIDEO_EVALUATION_COMPLETE\n');
