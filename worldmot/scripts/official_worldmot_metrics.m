% Format-only driver: unchanged native VisionTrack evaluation functions.
cfg=jsondecode(fileread(getenv('WORLDMOT_MATLAB_CONFIG')));
cd(cfg.kit); addpath(genpath(cfg.kit));
results=cell(numel(cfg.jobs),1);
for k=1:numel(cfg.jobs)
    job=cfg.jobs(k);
    fprintf(1,'WORLDMOT_OFFICIAL_JOB %d/%d %s %s\n',k,numel(cfg.jobs),job.scene,job.convention);
    [~,mets,ids,info,res]=evaluateTracking(job.scene,job.prediction,...
        job.groundtruth,job.gt_directory,cfg.benchmark);
    results{k}=struct('scene',job.scene,'convention',job.convention,...
        'Identity',ids,'CLEAR',info,'metrics',mets,'official_results',res);
    partial=struct('status','PARTIAL','completed_jobs',k,'total_jobs',numel(cfg.jobs),'results',{results(1:k)});
    pfid=fopen([cfg.output '.partial.json'],'w');assert(pfid>0);fprintf(pfid,'%s\n',jsonencode(partial));fclose(pfid);
end
report=struct('status','COMPLETE','version',version,...
    'native_official_function',which('evaluateTracking'),'results',{results});
fid=fopen(cfg.output,'w');assert(fid>0);fprintf(fid,'%s\n',jsonencode(report));fclose(fid);
