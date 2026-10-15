"""Seal measured Phase0 evidence without relabeling failed numerical probes."""
import numpy as np
from common import *

NEW=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')
OLD=Path('/data1/liuyeqiang/WWW_vggt_geo_runtime')


def main():
    inputs={};reports=[]
    for tag in ('official_real5_math_v5','det_real5_math_v3'):
        path=NEW/'interfaces'/tag/'manifest.json';report=json.loads(path.read_text())
        assert report['status']=='PASS_REAL_TWO_VIEW_INTERFACE' and report['deterministic_math']
        assert report['strict_aggregator_weights_loaded'] and report['frozen']
        assert report['input_shape']==[1,2,3,518,518] and report['metadata']['patch_start_idx']==5
        assert all(shape==[1,2,1369,2048] for shape in report['layer_shapes'].values())
        for file,digest in report['source_sha256'].items():assert sha256(file)==digest;inputs[file]=digest
        features=path.parent/'probe_positions.npz';assert sha256(features)==report['probe_features_sha256']
        inputs[str(path)]=sha256(path);inputs[str(features)]=sha256(features);reports.append(report)
    assert reports[0]['image_ids']==reports[1]['image_ids'] and reports[0]['checkpoint_sha256']==reports[1]['checkpoint_sha256']
    official=np.load(NEW/'interfaces/official_real5_math_v5/probe_positions.npz')
    det=np.load(NEW/'interfaces/det_real5_math_v3/probe_positions.npz')
    comparisons={}
    for layer in ('4','11','17','23'):
        comparisons[layer]=dict(exact=bool(np.array_equal(official[layer],det[layer])),max_abs=float(np.abs(official[layer]-det[layer]).max()))
        assert comparisons[layer]['exact']
    diag=reports[0]['det_attention_equivalence']
    assert diag['incoming_attention_allclose'] and diag['actual_attention_output_exact'] and diag['official_hook_all_four_layers_exact']
    assert all(row['exact'] for row in diag['official_no_hook_repeat_differences'].values())
    prepath=NEW/'interfaces/locked_input_preflight_v1.json';pre=json.loads(prepath.read_text())
    assert pre['locked_files_unchanged'] and pre['status']=='LOCKED_INPUTS_VERIFIED_MODEL_LOAD_NOT_RUN';inputs[str(prepath)]=sha256(prepath)
    paritypath=ROOT/'results/trackvggt_phase0_off_prefix_parity_v1.json';parity=json.loads(paritypath.read_text())
    assert parity['status']=='PASS_NEW_BRANCH_OFF_RAW_PREFIX_PARITY';inputs[str(paritypath)]=sha256(paritypath)
    for key in ('new_prediction','reference_prediction','reference_manifest'):
        assert sha256(parity[key])==parity[key+'_sha256'];inputs[parity[key]]=parity[key+'_sha256']
    log=NEW/'logs/phase0_and_event_guards_v3.log';assert 'Ran 11 tests' in log.read_text() and log.read_text().rstrip().endswith('OK');inputs[str(log)]=sha256(log)
    for file in (ROOT/'docs/TRACKVGGT_FINAL_GOAL.md',ROOT/'manifests/trackvggt_design_v1.json'):inputs[str(file)]=sha256(file)
    baselines=[]
    for name in ('causal_B0_strict_suite_v1_metrics.json','native_prior_seed20261014_strict_suite_v1_metrics.json','causal_B7_fixed_k1_strict_suite_v1_metrics.json'):
        path=ROOT/'results'/name;value=json.loads(path.read_text())
        assert value['status']=='PASS' and value['full_video'] and value['causal_online'] and value['video_ids']==[1,10,11,18,19]
        inputs[str(path)]=sha256(path)
        for captured in value['run_manifests'].values():
            assert sha256(captured['path'])==captured['sha256'];inputs[captured['path']]=captured['sha256']
        for field in ('prediction_sha256','raw_gt_sha256'):
            for file,digest in value[field].items():assert sha256(file)==digest;inputs[file]=digest
        baselines.append(dict(suite=name,metrics=value['metrics'],scope='retained audited full strict5, not new full inference'))
    report=dict(status='PASS_PHASE0_SOURCE_INPUT_INTERFACE_GATES',paired_real_probe_layers=comparisons,official=reports[0],det=reports[1],
        inherited_primary_causal_references=baselines,
        off_prefix=parity,source_artifacts=inputs,tests=11,backbone='frozen official VGGT aggregator only, math SDPA and deterministic cuDNN scoped to extraction',
        limitations='one real two-view group;32 retained positions/view for cross-source comparison, all four full layers for within-official hook/repeat parity; not full-video identity efficacy or throughput; failed default-backend probes remain failures',provenance=provenance())
    save_json(ROOT/'results/trackvggt_phase0_audit_v1.json',report)
    lines=['# TrackVGGT source audit','',
        'Pinned inheritance861c8710; new branch research/trackvggt-identity-query-www27-20261015. Full prior reports read; old ROI/pose/sparsity experiments are retained as negative evidence rather than repeated as new ideas.','',
        '## Actual AG mechanism and adaptation','',
        '[VGGT-Det vggtdet.py](https://github.com/yangcaoai/VGGT-Det-CVPR2026/blob/73ea4cf94a05fa50279ee0b6095fb06f6d042dac/projects/VGGTDet/vggtdet/vggtdet.py): attention_guided_prob_fps starts with highest normalized attention; following points maximize normalized attention plus lambda times nearest selected-point distance. Original coordinates are3D points. We independently implement deterministic2D patch selection within original detection/context, exclude pad centers, cap context and add TRAIN-learned historical-query relevance. AG-A/B/C are distinct ablations; no random token sampling or3D detector is substituted.','',
        '[Pinned aggregator](https://github.com/yangcaoai/VGGT-Det-CVPR2026/blob/73ea4cf94a05fa50279ee0b6095fb06f6d042dac/vggt/models/aggregator.py) selects frame block0 attention and averages heads/query positions, retaining key saliency after patch_start_idx. This is incoming attention, not an identity probability. Our official-backbone hook calculates the same Q/K/RoPE statistic in chunks; max absolute error versus the actual pinned Det attention on real inputs is '+str(diag['incoming_attention_max_abs'])+'. It does not write backbone output tensors.','',
        '## Actual QD mechanism and adaptation','',
        '[TransformerDecoder_Multilevel](https://github.com/yangcaoai/VGGT-Det-CVPR2026/blob/73ea4cf94a05fa50279ee0b6095fb06f6d042dac/projects/VGGTDet/detr3_models/transformer.py): separate four-layer channel projections in get_box_features; output task-query token -> task_head -> Softmax over layers -> weighted token memory -> decoder interactions. We use independent128-dimensional projections and a two-layer/four-head decoder. Global learned See-Query is mandatory control; per-history-track base+MLP conditioned See-Query is the hypothesis. AG/QD credit remains with VGGT-Det; no eight-layer/3D detection head/PointNet/FPS CUDA or MMDetection3D import.','',
        '## Layer/attention/preprocessing contracts','',
        'Official VGGT '+reports[0]['source_commit']+' returns24 positions with uncached None values; Det '+reports[1]['source_commit']+' returns a compact list for4/11/17/23. explicit_layer_map requires the correct explicit scheme. Real images[1,2,3,518,518], RGB[0,1], exactly one official internal normalization; patch_start_idx5 excludes camera/register tokens. Each retained layer[1,2,1369,2048]; attention[1,2,1369]. PadMapping round trips, original-coordinate row-major patch orientation and zero/tiny/padded regions are tested. No dense maps/full attention are persisted; only32 diagnostic positions/view remain outside Git.','',
        '## Numerical and weight compatibility','',
        'Both independent aggregators strictly load the original official aggregator checkpoint keys. Their32 retained positions/view are exactly equal at all four layers under matched math SDPA/deterministic cuDNN. The entire four-layer official output is exactly equal with and without the attention hook, as is a further plain repeat. Default FP16 backend probes did not repeat exactly even without the hook; those failures, unpacking error and raw diagnostics remain archived. The bounded backend check was documented before execution in TRACKVGGT_NUMERICAL_DIAGNOSTIC.md. New teacher extraction scopes the verified backend locally; it changes neither GMT settings nor weights. This is numerical interface evidence on one group, not a speed/identity result.','',
        '## Sources and licenses','',
        'VGGT-Det repository root LICENSE is MIT (copyright2024 yangcaoai); upstream-derived layers also carry Meta/Apache notices. Official VGGT LICENSE.txt is the separate VGGT License v1 dated July29,2025, with attribution, redistribution and Acceptable Use conditions. The official weights remain under their applicable official terms; the Det MIT root is not a replacement license. We reuse authorized local original weights, no indoor3D detection checkpoints. No commercial permission or deployment claim is inferred. External sources are read-only, SHA-pinned in the machine audit; no wholesale source copy.','',
        '## Remaining gates','',
        'Actual hard-event collection, all adapter edge cases, AG ranking/size/crop experiments, QD/query-shuffle, temporal cycles, conditional KD and complete deployment remain required. Phase0 success is not TEACHER_SUCCESS/STUDENT_SUCCESS/SCIENTIFIC_SUCCESS/PAPER_READY.']
    (ROOT/'docs/TRACKVGGT_SOURCE_AUDIT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'docs/PHASE0_SOURCE_AND_INTERFACE_AUDIT.md').write_text('# Phase0 measured gates\n\nPASS: isolated branch/remote inheritance, actual paths/hardware, input lock, strict411-key GMT load and40-group raw OFF parity;11 CPU query/AG/QD/selection guards; real original/Det strict-weight two-view attention/layer compatibility.\n\nSee TRACKVGGT_SOURCE_AUDIT.md and results/trackvggt_phase0_audit_v1.json for every source/input/report hash, exact shape and numerical-failure boundary. No formal teacher training or complete new video result is claimed.\n')
    print(json.dumps(dict(status=report['status'],tests=11,cross_source_layer_comparisons=comparisons,source_artifacts=len(inputs))))


if __name__=='__main__':main()
