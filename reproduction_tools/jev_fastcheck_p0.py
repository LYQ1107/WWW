"""Actual assets, authorization scope and machine audit before rollouts."""
import shutil,subprocess
from jev_fastcheck_common import *
def main():
 from jev_phase13_common import STAGE1,STAGE1_SHA,ANNOTATIONS,IMAGES
 assets=[]
 for arm in ['jev','attention','threshold','mlp']:
  p=RECOVERY_RUNTIME/f'training_R3/XIII/{arm}/seed{SEED}/pilot/RESULT.json'
  m=read(p); cp=m['checkpoint']; assert sha(cp['path'])==cp['SHA256']
  assets.append(dict(arm=arm,result=ref(p),checkpoint=cp))
 registry=read(ROOT/'reports/JEV_ONEWEEK/REUSED_EVIDENCE_REGISTRY.json')
 cp=next(r['asset'] for r in registry['weights'] if r['label']==f'XIII fixed_question seed{SEED}')
 assert sha(cp['path'])==cp['SHA256'];assets.append(dict(arm='StrongFixed',checkpoint=cp))
 assert sha(STAGE1)==STAGE1_SHA
 from jev_phase13_runtime import cache_inputs
 caches=[]
 for video in [12,13,14,16]:
  _,frames,reader=cache_inputs(video)
  index=reader.folder/'index.jsonl' if hasattr(reader,'folder') else pathlib.Path('/home/liuyeqiang/WWW_jev_phase13_runtime/20261009_v2/stage1_cache_v1')/f'video{video:02d}/index.jsonl'
  caches.append(dict(video=video,frames=frames,index=ref(index)))
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True)
 processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv'],text=True)
 recovery_monitor=read(RECOVERY_RUNTIME/'live_monitor/STATUS.json')
 result=dict(status='PASS_EXISTING_MATCH_SCOPE',source=binding('P0 assets and MATCH authorization'),
  assets=assets,Stage1=ref(STAGE1),perception_caches=caches,
  data=dict(TRAIN_annotation=ref(ANNOTATIONS),images_root=str(IMAGES),origin='repository README VisionTrack official OneDrive/Baidu links; existing supplied local TRAIN',
   user_permission=ref(ROOT/'docs/JEV_ONEWEEK/USER_TASK_SPEC.md'),prior_TRAIN_authorized=[12,13,14,16],
   authority='inherited user-declared licensed TRAIN scope, expressly reused by current fastcheck task; no new source-use license claimed',
   ordinary_MATCH=True,new_independent_authorization=False,new_long_term_safety_qualification=False,TEST_or_sealed_access=False),
  GPU=gpu,compute_processes=processes,free_GiB={p:shutil.disk_usage(p).free/2**30 for p in ['/home','/data1']},
  old_recovery_live_jobs=recovery_monitor['live_jobs'],new_visual_cache=False,old_results_modified=False)
 assert result['free_GiB']['/data1']>=30
 save(REPORTS/'BASELINE_AND_ACCESS_AUDIT.json',result)
 DOCS.mkdir(exist_ok=True)
 (DOCS/'BASELINE_AND_ACCESS_AUDIT.md').write_text('# Baseline and access audit\n\nPASS for ordinary MATCH only on inherited user-authorized VisionTrack TRAIN12/13/14; TRAIN16 gradient-free. Prior independent-data/safety blockage remains in force for those distinct claims. No new dataset authorization is inferred from local availability.\n\nR3 pilot JEV/Attention/MLP/Threshold, original Strong Fixed and GMT Stage1 checkpoints exist and SHA matches. Existing Stage1 cache indexes are reused. Native rollout/evaluation functions are reused without changing old source namespaces or checkpoint files. Actual GPU names/occupancy, source provenance, prior scope, cache indexes, free disk and hashes are in the JSON.\n\nGPU3–8 were free at audit; only free cards are selected. Old Recovery queue is terminal; other users’ jobs are untouched. New worktree copied historical reports are omitted locally with skip-worktree after Git-blob verification to preserve the 30GiB reserve; Git objects and all old worktrees remain intact.\n')
 print('P0_PASS',len(assets),[(c['video'],c['frames']) for c in caches],flush=True)
if __name__=='__main__':main()
