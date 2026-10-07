"""Publish a reviewable Phase V evidence bundle; exclude large model/data caches."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/home/liuyeqiang/WWW_jev_phase5_runtime/20261007')
DEST=ROOT/'reports/WWW_JEV_PHASE5_20261007'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if DEST.exists():raise RuntimeError('refusing to overwrite evidence bundle')
    assert json.loads((OUT/'MINIMAL_RETRAINING.json').read_text())['status']=='COMPLETE'
    DEST.mkdir(parents=True);entries=[]
    def copy(p,relative=None,compress=False):
        p=Path(p);relative=Path(relative) if relative else p.relative_to(OUT);target=DEST/relative
        if compress:target=target.with_name(target.name+'.gz')
        target.parent.mkdir(parents=True,exist_ok=True)
        if compress:
            with p.open('rb') as source,target.open('wb') as output:
                with gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0,compresslevel=6) as handle:shutil.copyfileobj(source,handle)
        else:shutil.copyfile(p,target)
        entries.append({'archived':str(target.relative_to(DEST)),'archive_sha256':sha(target),'source':str(p),'source_sha256':sha(p),'source_bytes':p.stat().st_size,'archived_bytes':target.stat().st_size,'compression':'gzip,mtime=0' if compress else None})
    for p in sorted(OUT.glob('*.json')):copy(p)
    for name in ['regression_tests.log','rng_tests.log','B1_training.log','B2_training.log','C1_training.log','C2_training.log','B1_tracking.log','B2_tracking.log','C1_tracking.log','C2_tracking.log','B2_repeat_tracking.log','B2_decision_audit.log']:
        copy(OUT/name)
    copy(OUT/'paired_decisions.jsonl',compress=True)
    copy(OUT/'B2_paired_decisions.jsonl',compress=True)
    for kind,variants in [('ablations',['A0','A1','A2','A3','A4']),('minimal_tracking',['B1','B2','B2_repeat','C1','C2'])]:
        for variant in variants:
            root=OUT/kind/variant;method='gmt_off' if variant=='A0' else 'jev'
            for name in ['result.json','runtime_status.json']:copy(root/name)
            for directory,filename in [('tracking_predictions',method+'.json'),('tracking_decisions',method+'.json')]:copy(root/directory/filename,compress=True)
            copy(root/'online_decisions.jsonl',compress=True)
            evaluation=root/'tracking_eval_runtime_state'/method/'evaluation'
            copy(evaluation/'metrics.json')
            for name in ['pedestrian_summary.txt','pedestrian_detailed.csv']:copy(evaluation/'GMT'/name)
            copy(root/'tracking_eval_runtime_state'/method/'prepared'/'manifest.json')
    for task in ['memory','match_relabel']:
        root=OUT/task;copy(root/'plan.json');copy(root/'records.jsonl',compress=True)
        for p in sorted(root.glob('chunks/*/manifest.json')):copy(p)
        for p in sorted(root.glob('worker_*.log')):copy(p)
        if (root/'launch.json').exists():copy(root/'launch.json')
        for p in sorted(root.glob('*error.json')):copy(p)
    for name,contract in [('match_training','aligned'),('match_training_legacy','legacy')]:
        root=OUT/name
        for video in [6,7]:copy(root/f'video{video:02d}_{contract}_match.jsonl',compress=True)
        copy(root/'compact/manifest.json');copy(root/'policy_split.json')
    for condition in ['B1','B2','C1','C2']:
        root=OUT/'minimal_training'/condition
        for name in ['model.pth','metrics.json','binding.json','runtime_status.json','calibration/model_calibrated.pth','calibration/calibration_val_only.json']:copy(root/name)
    frozen=json.loads((OUT/'PHASE5_BASELINE_LOCK.json').read_text())['input_and_policy_hashes']['jev']['calibrated_checkpoint'];copy(frozen,'frozen_MATCH_checkpoint/model_calibrated.pth')
    baseline=Path('/home/liuyeqiang/WWW_jev_rng_v4_runtime/segmented_gate_20261007_v4')
    for video in [1,6,7]:copy(baseline/f'video{video:02d}_records.jsonl',f'frozen_records/video{video:02d}_records.jsonl',compress=True)
    copy(OUT/'native_feature_audit_view.jsonl',compress=True)
    (DEST/'README.md').write_text('''# Phase V 可核对证据

完整报告在 ../../docs/WWW_JEV_PHASE5_DIAGNOSTICS_20261007.md。

本目录包含所有冻结标签、修正 MATCH 标签、十次跟踪预测/决策、训练与校准结果、checkpoint、完整审计与日志。JSON/JSONL 的 gzip 文件使用固定 mtime=0；原文件 SHA 与压缩文件 SHA 对照见 ARCHIVE_MANIFEST.json。

核对目录内文件：

```bash
sha256sum -c SHA256SUMS
```

核对压缩预测的原文件 SHA，例如：

```bash
gzip -dc minimal_tracking/B2/tracking_predictions/jev.json.gz | sha256sum
```

结果应与该条件 result.json 的 predictions_sha256 一致。B2/B2_repeat 的预测、动作与完整在线 feature/context 日志 SHA 相同。模型加载与动作检查在 MLP_POLICY_SANITY.json；公平对照在 CORRECTED_THREE_WAY_COMPARISON.json。

在原项目环境重新运行 B2（必须选择新的输出目录）：

```bash
env CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/liuyeqiang/anaconda3/envs/GMT/bin/python /data1/liuyeqiang/WWW_jev_phase5/reproduction_tools/run_jev_phase5_ablation.py --variant A1 --checkpoint /data1/liuyeqiang/WWW_jev_phase5/reports/WWW_JEV_PHASE5_20261007/minimal_training/B2/calibration/model_calibrated.pth --output /home/liuyeqiang/WWW_jev_phase5_runtime/20261007/reviewer_B2_fresh
```

该命令仍需原项目的 detectron2/GMT 环境、冻结 GMT model_20000、TRAIN 数据与感知缓存。外部输入的固定 SHA 见 PHASE5_BASELINE_LOCK.json；这些大文件没有上传。正式 TEST 未运行；eval_dataset 内的 test 名称只是 TRAIN 诊断子集的评测格式。

新 counterfactual builder CLI 默认使用 cache0_annotation1 坐标。函数层 legacy 默认保留给冻结复现；新程序调用应显式传入 gt_coordinate_contract="cache0_annotation1"。反事实 utility 的 prefix-reset / 固定 OFF action-category continuation 限制仍在报告中保留。
''')
    manifest={'status':'COMPLETE','source_commit_at_archival':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'runtime':str(OUT),'official_test_read':False,'full24_authorized':False,'entries':entries,'excluded':'Large GMT checkpoint, full perception cache, raw dataset/GT images, generated eval_dataset symlinks and duplicate NumPy arrays; their immutable hashes/manifests remain available.'}
    (DEST/'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    files=sorted(p for p in DEST.rglob('*') if p.is_file())
    (DEST/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(DEST))+'\n' for p in files))
    print(json.dumps({'status':'COMPLETE','files':len(files)+1,'bytes':sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),'destination':str(DEST)}))
if __name__=='__main__':main()
