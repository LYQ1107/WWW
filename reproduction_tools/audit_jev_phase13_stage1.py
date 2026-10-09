"""Actual checkpoints, function bodies and upstream training exposure, not directory names."""
import ast,collections
import torch,yaml
from jev_phase13_common import *
def group(model,prefix):
    h=hashlib.sha256();params=0
    for k,v in sorted(model.items()):
        if k.startswith(prefix):h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes());params+=v.numel()
    return {'SHA256':h.hexdigest(),'tensor_elements':params}
def main():
    protect();torch.set_num_threads(1)
    a=torch.load(STAGE1,map_location='cpu');b=torch.load(STAGE2,map_location='cpu');earlier=torch.load(STAGE1.parent.parent/'stage1/model_4000.pth',map_location='cpu')
    assert all(torch.isfinite(v).all() for v in a['model'].values() if v.is_floating_point())
    cfg=yaml.safe_load((STAGE1.parent/'config.yaml').read_text());cfg2=yaml.safe_load((STAGE2.parent/'config.yaml').read_text());assert cfg['REID'] is True and cfg2['REID'] is False
    annotation=Path('/data1/liuyeqiang/WWW/datasets/VisionTrack/images/annotations/train_stage1.json');d=json.loads(annotation.read_text());exposure=sorted({i['video_id'] for i in d['images']})
    funcs={}
    for name in ['gtr/modeling/roi_heads/gtr_roi_heads.py','gtr/modeling/roi_heads/association_head.py','gtr/modeling/roi_heads/transformer.py','gtr/modeling/meta_arch/gtr_rcnn.py']:
        text=(ROOT/name).read_text();lines=text.splitlines();funcs[name]={'file_SHA256':sha(ROOT/name),'functions':[{'name':n.name,'line':n.lineno,'end_line':n.end_lineno,'body_SHA256':hashlib.sha256('\n'.join(lines[n.lineno-1:n.end_lineno]).encode()).hexdigest()} for n in ast.walk(ast.parse(text)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]}
    groups={}
    for prefix in ['backbone.','proposal_generator.','roi_heads.asso_head.','roi_heads.classify_head.','roi_heads.s_t_head.','roi_heads.transformer.','roi_heads.asso_predictor.']:
        first=group(a['model'],prefix);previous=group(earlier['model'],prefix);second=group(b['model'],prefix)
        groups[prefix]={'Stage1':first,'Stage1_4000':previous,'Stage2':second,'changed_during_stage1_continuation':first!=previous,'changed_during_stage2':first!=second}
    metrics=[json.loads(x) for x in (STAGE1.parent/'metrics.json').read_text().splitlines() if x.strip()];loss=[x for x in metrics if 'loss_reid' in x]
    assert loss and groups['roi_heads.asso_head.']['changed_during_stage1_continuation']
    result={'status':'PASS_FEATURE_TRUTH_WITH_PRETRAIN_EXPOSURE_LIMITATION','binding':binding(),'Stage1':{'path':str(STAGE1),'SHA256':sha(STAGE1),'bytes':STAGE1.stat().st_size,'checkpoint_iteration':a.get('iteration'),'scheduler_last_epoch':a.get('scheduler',{}).get('last_epoch'),'config_SHA256':sha(STAGE1.parent/'config.yaml'),'REID':cfg['REID'],'VFCE_dim':cfg['MODEL']['ASSO_HEAD']['FC_DIM'],'model_keys':len(a['model']),'optimizer_present':'optimizer' in a,'all_model_finite':True,'actual_reid_loss_records':len(loss),'last_training_loss':loss[-1]},'Stage2':{'path':str(STAGE2),'SHA256':sha(STAGE2),'REID':cfg2['REID'],'RPCE_dim':cfg2['MODEL']['ASSO_HEAD']['CAT_DIM'],'main_feature_source':False},'module_digests':groups,'RPCE_stage1_trained':groups['roi_heads.s_t_head.']['changed_during_stage1_continuation'],'RPCE_stage2_changed':groups['roi_heads.s_t_head.']['changed_during_stage2'],'source_functions':funcs,'upstream_feature_pretraining_exposure':{'dataset':str(annotation),'SHA256':sha(annotation),'video_ids':exposure,'development_preexposed':[v for v in VAL if v in exposure],'controller_heldout_preexposed':[v for v in SEALED if v in exposure],'scope':'Metadata/provenance audit only; no heldout image, feature or inference read in Phase XIII. Existing frontend trained on all24 TRAIN videos, so inherited20/21/22 cannot establish end-to-end independent generalization.','G9_status':'NO_GO_PRETRAIN_EXPOSED_KEEP_SEALED'},'new_stage2_supervision_partition':{'TRAIN':TRAIN,'VAL':VAL,'no_VAL_training_labels':True},'main_input':'fresh Stage1 detector and unjittered ROI VFCE1024, never Stage2 RPCE1152 cache; all new neural baselines same frontend','official_TEST':False,'Full24':False}
    save(REPORTS/'STAGE1_CHECKPOINT_AUDIT.json',result)
    from jev_phase12_common import REPORTS as old_reports
    save(REPORTS/'GMT_BASELINE_FREEZE.json',{'status':'FROZEN','source':BASE,'original_GMT_checkpoint_SHA256':sha(STAGE2),'original_full_online_metrics_report_SHA256':sha(old_reports/'MATCH_VALIDATION_RESULTS.json'),'GMT_OFF_metrics':next(r for r in json.loads((old_reports/'MATCH_VALIDATION_RESULTS.json').read_text())['cases'] if r['case']['name']=='GMT_OFF')['pooled_metrics'],'scope':'unchanged complete original model strong comparator; not same Stage1 frontend causal ablation','OFF_old_source_unchanged':True})
    print('STAGE1_ACTUAL_VFCE_TRUTH_AND_EXPOSURE_AUDITED',flush=True)
if __name__=='__main__':main()
