cd('/data3/liuyeqiang/GMT_VisionTrack_repro/code/GMT/MOTChallengeEvalKit_cv_test');
addpath(genpath('matlab_devkit'));
addpath('/data3/liuyeqiang/GMT_VisionTrack_repro/tools/octave_mex');
names={'perfect','miss','false_positive','id_split'}; expected_mota=[100,75,75,75]; expected_idf=[100,600/7,800/9,50];
for k=1:numel(names)
 [name,mets,ids,extra,res]=evaluateTracking('synthetic', ['/data3/liuyeqiang/GMT_VisionTrack_repro/outputs/evaluation_smoke/',names{k},'.txt'], '/data3/liuyeqiang/GMT_VisionTrack_repro/outputs/evaluation_smoke/gt/synthetic.txt', '/data3/liuyeqiang/GMT_VisionTrack_repro/outputs/evaluation_smoke/gt/synthetic', 'MOT16');
 assert(abs(mets(15)-expected_mota(k))<1e-7); assert(abs(ids.IDF1-expected_idf(k))<1e-7);
 fprintf('CASE %s PASS MOTA=%.8f IDF1=%.8f\n',names{k},mets(15),ids.IDF1);
end
disp('OFFICIAL_EVALUATOR_OCTAVE_SYNTHETIC_PASS');
