"""Preregistered G0--G5 training and closed-loop replay, protecting frozen B2."""
import argparse
from collections import Counter
import copy
import datetime as dt
import gzip
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from jev_phase6_common import ROOT, OUT, REPORTS, PHASE5, B2, PYTHON, sha, save, protect_anchor, new_output, load_controller


def binary_record(record):
    result = copy.deepcopy(record)
    legal = [a for a in record['legal_actions'] if a != 'REASSOCIATE']
    utilities = [float(record['action_outcomes'][a]['utility']) for a in legal]
    temperature = float(record.get('labeling', {}).get('temperature', 1.0))
    import math
    masses = [math.exp((u - max(utilities)) / temperature) for u in utilities]
    result.update(legal_actions=legal,
                  target_probs=[m / sum(masses) for m in masses],
                  best_actions=[a for a, u in zip(legal, utilities) if abs(u - max(utilities)) <= 1e-8],
                  action_outcomes={a: result['action_outcomes'][a] for a in legal})
    result['binary_action_contract'] = 'retained original utilities; no arbitrary reassociation-label folding'
    return result


def prepare():
    import numpy as np
    from jev_compact_dataset import build_compact_dataset, CompactJEVData
    protect_anchor()
    target = new_output(OUT / 'binary_match')
    source = ROOT / 'reports/WWW_JEV_PHASE5_20261007/match_training'
    inputs = []
    counts = Counter()
    for video in (6, 7):
        dest = target / f'video{video:02d}_binary_match.jsonl'
        with gzip.open(source / f'video{video:02d}_aligned_match.jsonl.gz', 'rt') as reader, dest.open('w') as writer:
            for line in reader:
                row = binary_record(json.loads(line))
                writer.write(json.dumps(row, sort_keys=True) + '\n')
                counts[video] += 1
        inputs.append(dest)
    build_compact_dataset(inputs, target / 'compact')
    original = CompactJEVData(PHASE5 / 'match_training/compact')
    derived = CompactJEVData(target / 'compact')
    contracts = {}
    for name in ('features', 'questions', 'sample_weight', 'sequence_ids', 'video_ids', 'frames', 'views', 'event_orders'):
        contracts[name] = bool(np.array_equal(getattr(original, name), getattr(derived, name)))
    if not all(contracts.values()) or counts != {6: 2600, 7: 1703}:
        raise RuntimeError('binary data altered the frozen non-action input contract')
    split = json.loads((PHASE5 / 'match_training/policy_split.json').read_text())
    split.update(source_files=[str(p) for p in inputs], source_sha256={str(p): 'sha256:' + sha(p) for p in inputs},
                 purpose='Phase VI binary action novelty control; no new selection sequences')
    save(target / 'policy_split.json', split)
    save(REPORTS / 'BINARY_MATCH_DATA_CONTRACT.json', {
        'status': 'PASS', 'records': sum(counts.values()), 'counts_by_video': dict(counts),
        'unchanged_input_arrays': contracts, 'source_manifest_sha256': sha(original.root / 'manifest.json'),
        'binary_manifest_sha256': sha(derived.root / 'manifest.json'), 'split_sha256': sha(target / 'policy_split.json'),
        'official_test_read': False, 'limitations_inherited': 'H8 prefix-reset and frozen OFF action-category continuation; same as B2',
    })
    print(json.dumps({'status': 'PREPARED', 'records': sum(counts.values())}))


def train(condition, device):
    import numpy as np
    import torch
    from gtr.modeling.jev_lifecycle_gates import MatchThresholdGate
    from gtr.modeling.jev_decision import JEVDecisionController
    from jev_compact_dataset import CompactJEVData
    from train_jev import load_policy_split
    from train_jev_compact import batch_forward, evaluate
    protect_anchor()
    output = new_output(OUT / 'gating_training' / condition)
    random.seed(20261003); np.random.seed(20261003); torch.manual_seed(20261003)
    data = CompactJEVData(OUT / 'binary_match/compact')
    splits = load_policy_split(OUT / 'binary_match/policy_split.json')
    train_ids = data.indices_for_sequences(splits['train'])
    val_ids = data.indices_for_sequences(splits['val'])
    hidden = {'G1': None, 'G2': 154, 'G3': 128}[condition]
    if condition == 'G3':
        model = JEVDecisionController(64, hidden_dim=128, question_dim=32, action_dim=32)
        name = 'jev'
    else:
        model = MatchThresholdGate(64, hidden)
        name = 'phase6_scalar_gate' if condition == 'G1' else 'phase6_dynamic_gate'
    model.to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if condition == 'G2' and abs(parameters / 34080 - 1) >= 0.01:
        raise RuntimeError('dynamic gate exceeds preregistered parameter difference')
    binding = {'condition': condition, 'model': name, 'hidden_dim': hidden, 'trainable_params': parameters,
               'source_commit': __import__('subprocess').check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
               'compact_manifest_sha256': sha(data.root / 'manifest.json'), 'split_sha256': sha(OUT / 'binary_match/policy_split.json'),
               'seed': 20261003, 'initial_threshold': 0.1, 'score_index': 56, 'epochs': 20, 'official_test_read': False}
    save(output / 'binding.json', binding)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rng = np.random.default_rng(20261003)
    history = []
    for epoch in range(20):
        model.train(); order = train_ids.copy(); rng.shuffle(order)
        loss_sum = weight_sum = 0.0
        for start in range(0, len(order), 128):
            ids = order[start:start+128]
            loss, _, _, weights = batch_forward(model, data, ids, torch.device(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            weight_sum += float(weights.sum()); loss_sum += float(loss.item()) * float(weights.sum())
        history.append({'epoch': epoch+1, 'train_loss': loss_sum/max(1,weight_sum),
                        'val': evaluate(model, data, val_ids, torch.device(device))})
        save(output / 'runtime_status.json', {'phase': 'TRAINING', 'condition': condition, 'completed_epochs': epoch+1, 'pid': os.getpid()})
    payload = {'model': model.cpu().state_dict(), 'model_name': name, 'state_dim': 64, 'hidden_dim': hidden,
               'question_dim': 32 if condition == 'G3' else None, 'action_dim': 32 if condition == 'G3' else None,
               'num_layers': 2 if condition == 'G3' else None, 'temperature': 1.0,
               'use_option_interaction': False, 'seed': 20261003, 'splits': splits,
               'phase6_binding': binding, 'initial_threshold': 0.1}
    torch.save(payload, output / 'model.pth')
    save(output / 'metrics.json', {'status':'COMPLETE', 'binding':binding, 'history':history, 'train_records':len(train_ids), 'val_records':len(val_ids)})
    # Validation-only temperature calibration; no checkpoint/threshold selection.
    from calibrate_jev_compact import predict, metrics
    model.eval(); raw = predict(model, data, val_ids)
    target = torch.as_tensor(np.asarray(data.target_probs[val_ids]), dtype=torch.float64)
    weights = torch.as_tensor(np.asarray(data.sample_weight[val_ids]), dtype=torch.float64)
    valid = torch.as_tensor(np.asarray(data.legal_actions[val_ids])) >= 0
    logits = raw.double().clamp_min(1e-8).log()
    parameter = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    calibration_optimizer = torch.optim.LBFGS([parameter], lr=0.25, max_iter=80, line_search_fn='strong_wolfe')
    def closure():
        calibration_optimizer.zero_grad()
        scaled = (logits / parameter.exp().clamp(0.05,20)).masked_fill(~valid, torch.finfo(torch.float64).min)
        loss = (-(target * torch.log_softmax(scaled, dim=-1).masked_fill(~valid,0)).sum(dim=-1)*weights).sum()/weights.sum()
        loss.backward(); return loss
    calibration_optimizer.step(closure)
    temperature = float(parameter.detach().exp().clamp(0.05,20))
    payload['calibration_temperature'] = temperature
    # The existing JEV loader uses temperature inside its semantic scorer.
    # The isolated scalar/dynamic loader wraps its logits instead.
    if condition == 'G3': payload['temperature'] = temperature
    torch.save(payload, output / 'model_calibrated.pth')
    calibration = {'status':'PASS','validation_only':True,'temperature':temperature,'weights_sha256_before':sha(output/'model.pth'),
                   'raw':metrics(raw,data,val_ids),'calibrated':metrics(raw,data,val_ids,temperature)}
    save(output / 'calibration.json', calibration)
    save(output / 'runtime_status.json', {'phase':'COMPLETE','checkpoint_sha256':sha(output/'model_calibrated.pth')})
    print(json.dumps({'condition':condition, 'status':'COMPLETE','trainable_params':parameters}))


def replay(condition, video, output, device, max_frame=None):
    protect_anchor()
    output = new_output(output)
    empty = OUT / 'empty_debug_inputs.jsonl'
    if not empty.exists(): empty.write_text('')
    os.environ.update(JEV_PILOT_ROOT=str(output), JEV_VIDEO_ID=str(video),
                      JEV_TRACE_PATH=str(empty), JEV_RECORDS_PATH=str(empty))
    import run_early_pilot_tracking as pilot
    from run_jev_phase5_ablation import now
    checkpoint = None if condition == 'G0' else B2 if condition in {'G4a','G4b','G5'} else OUT/'gating_training'/condition/'model_calibrated.pth'
    original_choose = pilot.choose
    counts = Counter(); original_counts = Counter()
    log = output / 'online_decisions.jsonl'
    last_update = 0.0
    with log.open('w') as writer:
        def choose(policy, feature, question, legal, off_action, context=None):
            nonlocal last_update
            binary = condition in {'G1','G2','G3'} and question == 'MATCH_DECISION'
            actual_legal = [a for a in legal if a != 'REASSOCIATE'] if binary else list(legal)
            threshold = None
            if condition == 'G0' or question != 'MATCH_DECISION':
                original_action = action = off_action
            elif condition in {'G1','G2'}:
                controller = policy.controller
                from gtr.modeling.jev_runtime import TemperatureScaledController
                if isinstance(controller, TemperatureScaledController): controller = controller.controller
                controller_feature = feature.to(next(controller.parameters()).device)
                action = original_action = controller.decide_strict(controller_feature, actual_legal)
                threshold = float(controller.threshold_for(controller_feature).detach()[0])
            else:
                original_action = original_choose(policy, feature, question, actual_legal, off_action, context=context)
                action = original_action
                if original_action == 'REASSOCIATE' and condition in {'G4a','G4b'}:
                    action = 'START_NEW' if condition == 'G4a' else 'ACCEPT_CURRENT'
            if action not in actual_legal:
                raise RuntimeError('illegal Phase VI action')
            counts[(question,action)] += 1; original_counts[(question,original_action)] += 1
            item = {'question':question,'action':action,'original_policy_action':original_action,'off_action':off_action,
                    'legal_actions':actual_legal,'feature_vector':feature.detach().cpu().tolist(),'context':context or {}}
            if threshold is not None: item['learned_threshold'] = threshold
            writer.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n')
            stamp = time.monotonic()
            if stamp-last_update > 10:
                writer.flush(); last_update=stamp
                save(output/'runtime_status.json', {'phase':'INFERENCE','condition':condition,'video':video,'frame':(context or {}).get('frame'),'view':(context or {}).get('view'),'pid':os.getpid(),'updated_utc':now()})
            return action
        pilot.choose = choose
        annotations, subset, lookup, by_key, records = pilot.load_inputs()
        names = ['build_formal_gmt_engine','MutableGMTState','FrozenPerceptionCache','JEVRuntimePolicy',
                 'build_controller_from_checkpoint','build_state_features','association_window_length',
                 'candidate_entropy','count_memory_observations','count_track_history','feature_names','legacy_acceptance_threshold']
        kwargs = dict(zip(names,pilot.load_runtime_modules())); kwargs.pop('feature_names')
        kwargs['build_controller_from_checkpoint'] = load_controller
        name = 'gmt_off' if condition == 'G0' else 'jev'
        result = pilot.run_method(name,checkpoint,image_lookup=lookup,by_key=by_key,records=records,
                                  feature_source_mode='runtime',parity_report={},device=device,max_frame=max_frame,**kwargs)
    save(output/'runtime_status.json', {'phase':'EVALUATION','pid':os.getpid(),'updated_utc':now()})
    dataset = pilot.prepare_eval_dataset(subset)
    prepared, evaluated = pilot.run_eval(name,Path(result['predictions']),dataset)
    metrics = pilot.extract_metrics(evaluated)
    value = {'status':'PASS','condition':condition,'video':video,'role':'DEVELOPMENT_DIAGNOSTIC' if video==1 else 'PREREGISTERED_CONTROLLER_TRAIN_HELDOUT',
             'metrics':metrics,'result':result,'checkpoint':str(checkpoint) if checkpoint else None,
             'checkpoint_sha256':sha(checkpoint) if checkpoint else None,'predictions_sha256':sha(result['predictions']),
             'decisions_sha256':sha(result['decisions']),'online_decisions_sha256':sha(log),
             'actions':{q:dict((a,n) for (qq,a),n in counts.items() if qq==q) for q in ('MATCH_DECISION','MEMORY_DECISION','REACTIVATION_DECISION')},
             'original_policy_actions':{q:dict((a,n) for (qq,a),n in original_counts.items() if qq==q) for q in ('MATCH_DECISION','MEMORY_DECISION','REACTIVATION_DECISION')},
             'official_test_read':False,'full24_authorized':False,'completed_utc':now(),'max_frame':max_frame,'prepared':str(prepared),'evaluation':str(evaluated)}
    if video == 1 and max_frame is None and condition in {'G0','G5'}:
        reference = json.loads((PHASE5/('ablations/A0/result.json' if condition=='G0' else 'minimal_tracking/B2/result.json')).read_text())
        value['frozen_reproduction'] = {'prediction_sha_identical':value['predictions_sha256']==reference['predictions_sha256'],
                                        'decisions_sha_identical':value['decisions_sha256']==reference['decisions_sha256'],
                                        'max_metric_error':max(abs(metrics[k]-reference['metrics'][k]) for k in metrics)}
        if not value['frozen_reproduction']['prediction_sha_identical'] or value['frozen_reproduction']['max_metric_error'] != 0:
            raise RuntimeError('permanent frozen baseline reproduction failed')
    save(output/'result.json',value)
    save(output/'runtime_status.json',{'phase':'COMPLETE','condition':condition,'video':video,'updated_utc':now()})
    protect_anchor()
    print(json.dumps({'condition':condition,'video':video,'metrics':metrics,'status':'PASS'}))


def aggregate():
    results = {c:json.loads((OUT/'gating_tracking'/c/'result.json').read_text()) for c in ('G0','G1','G2','G3','G4a','G4b','G5')}
    primary=('HOTA','AssA'); threshold=0.10
    def exceeds(a,b): return all(results[a]['metrics'][m]-results[b]['metrics'][m] >= threshold for m in primary)
    equivalent=all(abs(results[a]['metrics'][m]-results[b]['metrics'][m])<=threshold for a,b in (('G5','G3'),('G3','G2'),('G5','G2')) for m in primary)
    comparisons={f'{a}_vs_{b}':{m:results[a]['metrics'][m]-results[b]['metrics'][m] for m in results[a]['metrics']} for a,b in (('G5','G0'),('G5','G2'),('G5','G3'),('G3','G2'),('G5','G4a'),('G5','G4b'))}
    passed=exceeds('G5','G3') and exceeds('G3','G2') and exceeds('G5','G4a') and exceeds('G5','G4b')
    report={'status':'COMPLETE','conditions':results,'deltas':comparisons,'material_effect_pp':threshold,
            'STRUCTURED_MATCH_CLAIM':'PASS' if passed else 'NOT_ESTABLISHED','first_stop_condition_triggered':equivalent,
            'unified_training_eligible_from_match':passed,'binary_dynamic_three_action_descriptively_equivalent':equivalent,
            'what_did_we_learn':'inspect explicit metrics, frozen replacements and event evidence; action count alone does not establish structured novelty',
            'claim_scope':'single seed, reused TRAIN development video01; no heldout/generalization claim','official_test_read':False,'full24_authorized':False}
    save(REPORTS/'FANCY_GATING_AUDIT.json',report)
    print(json.dumps({c:r['metrics'] for c,r in results.items()}))
    print(json.dumps({'STRUCTURED_MATCH_CLAIM':report['STRUCTURED_MATCH_CLAIM'],'first_stop_condition_triggered':equivalent}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','train','replay','aggregate'))
    parser.add_argument('--condition',choices=('G0','G1','G2','G3','G4a','G4b','G5'))
    parser.add_argument('--video',type=int,default=1)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-frame',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='train':train(args.condition,args.device)
    elif args.mode=='replay':replay(args.condition,args.video,args.output,args.device,args.max_frame)
    else:aggregate()


if __name__=='__main__':
    try: main()
    except BaseException:
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])
            save(path/'failure.json',{'status':'FAILED','error':traceback.format_exc()})
        raise
