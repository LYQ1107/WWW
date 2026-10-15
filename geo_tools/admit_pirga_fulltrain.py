"""Independent complete input admission using sealed original traces and labels.

No video/model/teacher/GT matching is run. Native inputs and original actions
are checked before separately reusing offline existing-label certificates.
Actual descriptor versions without support certification remain UNKNOWN for
supervised identity targets, never removed from legal inference candidates.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from pirga_indexed_store import StoreReader,sha,dense_matrix
from audit_pirga_fulltrain import FIT,read_rows,checked


def jsha(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()


def validate_scene(scene,folder,scope,metadata_path,source_pins,label_stream):
    mp=folder/'manifest.json';m=json.loads(mp.read_text());sp=next(s for s in scope['scenes'] if s['scene']==scene)
    if m['status']!='COMPLETE_FULL_NATIVE_INPUTS_PARITY_PASS_PENDING_SUPERVISION' or m['groups']!=sp['groups'] or m['queries']!=sp['expected_queries']:raise ValueError('Incomplete full-scene source')
    base=Path(sp['original_audit_manifest']).parent
    for n,digest in [(sp['original_audit_manifest'],sp['original_audit_manifest_sha256']),(base/'actual_decisions.jsonl.gz',sp['original_decisions_sha256']),(base/'predictions.json',sp['original_predictions_sha256'])]:checked(n,digest)
    if sha(folder/'predictions.json')!=sp['original_predictions_sha256']:raise ValueError('Original full prediction bytes differ')
    for name in ['actual_decisions.jsonl.gz','detector_provenance.jsonl.gz']:
        old=list(read_rows(base/name));new=list(read_rows(folder/name))
        if old!=new:raise ValueError('Native source trace differs '+name)
    r=StoreReader(folder/'indexed_inputs',m['indexed_store_manifest_sha256'])
    if r.manifest['interface']!='pirga-indexed-native/v2' or r.manifest['GT_in_input_store'] is not False:raise ValueError('Wrong input schema/GT mixing')
    feat={row[0]:dict(scene=row[1],image=row[2],detection=row[3],frame=row[4],camera=row[5],variant=row[6]) for row in r.db.execute('SELECT id,scene,image,detection,frame,camera,variant FROM features')}
    image_map={x['image_id']:x for x in sp['images']};raw_by_source={}
    for fid,o in feat.items():
        im=image_map[o['image']]
        if o['scene']!=scene or o['frame']!=im['frame_id'] or o['camera']!=im['view_id']:raise ValueError('Physical camera/time source mismatch')
        if o['variant'] is None:raw_by_source[(o['image'],o['detection'])]=fid
    for start in range(0,r.manifest['features'],2048):
        x=np.asarray(r.vectors[start:start+2048],dtype=np.float32)
        if x.shape[1]!=1152 or not np.isfinite(x).all() or np.any(np.linalg.norm(x,axis=1)==0):raise ValueError('Native1152 vector contract failed')
    # Revalidate all original paired scores/thresholds plus full joint contexts.
    baseline={(c['stage'],c['method_call_index']):c for c in read_rows(base/'actual_decisions.jsonl.gz')}
    by_call=defaultdict(list)
    for row in read_rows(metadata_path):by_call_key=tuple(row['event_key']);by_call[by_call_key]=row
    consumed=set();keys=[];count=Counter();eligible=[];prediction=[];groups=set();stage_index=Counter();matrix_cells=references=0
    for c in r.calls():
        index=stage_index[c['stage']];stage_index[c['stage']]+=1;old=baseline[c['stage'],index]
        if c['query_keys']!=[[scene,*s] for s in old['queries']] or c['accepted_ids']!=old['accepted_track_ids']:raise ValueError('Original joint action/query rows changed')
        actual_refs=[dict(source=[feat[f]['image'],feat[f]['detection']],track=t) for f,t in zip(c['reference_features'],c['reference_tracks'])]
        if actual_refs!=old['references']:raise ValueError('Complete legal reference axis changed')
        mat=dense_matrix(c['matrix_shape'],c['matrix_entries']);cols={t:i for i,t in enumerate(c['candidate_ids'])}
        for p in old['hungarian_pairs']:
            col=cols[p['track']]
            if float(mat[p['row'],col])!=p['score'] or c['thresholds'][col]!=p['threshold']:raise ValueError('Original native score/threshold changed')
        references+=len(actual_refs);matrix_cells+=mat.size
        if c['stage']=='run_memory_tracker':continue
        rows=[by_call[tuple(k)] for k in c['query_keys']]
        # Candidate purity certificates describe the same full native call, so
        # known alias positives may also certify negatives for another row.
        people=[None]*len(cols)
        for label in rows:
            if label['candidate_ids_from_native_call']!=c['candidate_ids']:raise ValueError('GT-created or truncated candidate set')
            for i,p in enumerate(label['raw_source_candidate_persons']):
                if people[i] is not None and p is not None and people[i]!=p:raise ValueError('Conflicting source purity')
                if p is not None:people[i]=p
            if label['GT_person_existing'] is not None:
                for t in label['sealed_positive_tracks']:
                    i=cols[t];p=label['GT_person_existing']
                    if people[i] is not None and people[i]!=p:raise ValueError('Conflicting sealed alias certificate')
                    people[i]=p
        unsafe={cols[t] for fid,t in zip(c['reference_features'],c['reference_tracks']) if feat[fid]['variant'] is not None}
        for i in unsafe:people[i]=None
        for k,qfid,label in zip(c['query_keys'],c['query_features'],rows):
            key=tuple(k)
            if key in consumed or feat[qfid]['image']!=k[1] or feat[qfid]['detection']!=k[2]:raise ValueError('Duplicate/mismatched physical query')
            consumed.add(key);keys.append(k);person=label['GT_person_existing'];positive=[i for i,p in enumerate(people) if person is not None and p==person]
            known_negative=any(p is not None and p!=person for p in people) if person is not None else False
            qo=feat[qfid];past=[(f,t) for f,t in zip(c['reference_features'],c['reference_tracks']) if feat[f]['frame']<qo['frame']]
            rank=bool(positive and known_negative and past);cross=bool(positive) and any(cols[t] in positive and feat[f]['camera']!=qo['camera'] for f,t in past)
            count['queries']+=1;count['GT_known']+=person is not None;count['UNKNOWN_query']+=person is None;count['pure_positive_available_after_descriptor_mask']+=bool(positive)
            count['ranking_eligible']+=rank;count['crossview_prediction_eligible']+=cross;count['crossview_prediction_active_on_common_rank_domain']+=rank and cross;count['query_exposed_to_uncertified_descriptor_candidate']+=bool(unsafe)
            count['native_wrong_or_missed_rank_eligible']+=rank and label['actual_primary_state_existing'] in ('wrong_link','missed_known_positive')
            if person is not None:groups.add((scene,person))
            if rank:eligible.append(k)
            if cross:prediction.append(k)
            label_stream.write(json.dumps(dict(event_key=k,GT_person=person,candidate_persons=people,actual_primary_state=label['actual_primary_state_existing']),separators=(',',':'))+'\n')
    if len(keys)!=sp['expected_queries'] or jsha(keys)!=sp['query_keys_sha256']:raise ValueError('All actual original query coverage differs')
    # Old caches remain immutable and independently SHA-bound; only compare
    # native observations, never import teacher tensors or mining/training code.
    import torch
    oldcache=Path('/data1/liuyeqiang/WWW_trackvggt_runtime/selected_native_v3')/f'scene{scene}_full_cycle_v1/events.pt';checked(oldcache,source_pins[str(oldcache)])
    cache=torch.load(oldcache,map_location='cpu',weights_only=True);overlap=0
    for key,o in cache['observations'].items():
        fid=raw_by_source.get(tuple(key))
        if fid is None:raise ValueError('Old physical observation absent from complete input')
        if not np.array_equal(np.asarray(r.vectors[fid],dtype=np.float16),o['feature'].cpu().numpy().astype(np.float16)):raise ValueError('Old frozen native feature differs at '+str((scene,*key)))
        overlap+=1
    result=dict(scene=scene,counts=dict(count),native_calls=sum(stage_index.values()),matrix_cells=matrix_cells,reference_entries=references,
        source_ordered_query_sha256=jsha(keys),known_person_groups=len(groups),raw_observations=len(raw_by_source),actual_descriptor_versions=r.manifest['features']-len(raw_by_source),
        old_selected_observation_overlap=overlap,old_FP16_observations_bitwise_equal=True,original_predictions_and_all_recorded_native_actions_exact=True,
        input_store=dict(root=str((folder/'indexed_inputs').resolve()),manifest_sha256=m['indexed_store_manifest_sha256']))
    return result,eligible,prediction


def main():
    p=argparse.ArgumentParser();p.add_argument('--allocation',type=Path,required=True);p.add_argument('--capture-root',type=Path,required=True);p.add_argument('--metadata-root',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();started=time.monotonic()
    if a.out.exists():raise FileExistsError(a.out)
    allocation=json.loads(a.allocation.read_text());scope=json.loads(Path(allocation['scope_path']).read_text())
    if sha(allocation['scope_path'])!=allocation['scope_sha256']:raise ValueError('Frozen scope changed')
    ledger=json.loads((a.capture_root/'RESOURCE_LEDGER.json').read_text())
    if ledger['status']!='ALL11_CAPTURED_PENDING_CONSUMER_ADMISSION':raise ValueError('All11 complete sources required before admission')
    source=json.loads(a.source_manifest.read_text())['artifacts'];a.out.mkdir();results=[];eligible=[];prediction=[];totals=Counter()
    lp=a.out/'offline_supervision.jsonl.gz'
    try:
        with gzip.open(lp,'xt') as labels:
            for scene in FIT:
                result,e,pred=validate_scene(scene,a.capture_root/f'scene{scene}',scope,a.metadata_root/f'scene{scene}_metadata_only_supervision.jsonl.gz',source,labels)
                results.append(result);eligible.extend(e);prediction.extend(pred);totals.update(result['counts']);print(json.dumps(dict(scene=scene,counts=result['counts'])),flush=True)
        if totals['queries']!=154911:raise ValueError('Complete TRAIN count differs')
        ep=a.out/'ranking_eligible_keys.json';ep.write_text(json.dumps(eligible,separators=(',',':'))+'\n')
        pp=a.out/'prediction_eligible_keys.json';pp.write_text(json.dumps(prediction,separators=(',',':'))+'\n')
        stores={str(v['scene']):v['input_store'] for v in results};bindings={v['input_store']['root']:v['input_store']['manifest_sha256'] for v in results}
        report=dict(status='FULLTRAIN_DATA_CONTRACT_PASS',fit_scenes=list(FIT),development_excluded=[12],actual_queries=154911,counts=dict(totals),per_scene=results,stores=stores,source_store_manifests=bindings,
            supervision=dict(path=str(lp.resolve()),sha256=sha(lp)),ranking_eligibility=dict(path=str(ep.resolve()),sha256=sha(ep)),prediction_eligibility=dict(path=str(pp.resolve()),sha256=sha(pp)),
            labels_GT_physically_separate=True,UNKNOWN_not_negative=True,aliases_multi_positive=True,unverified_native_descriptor_identity_masked=True,legal_candidates_not_filtered=True,
            old_native_features_and_source_solver_records_reverified=True,training=False,source_GT_recomputed=False,wall_seconds=time.monotonic()-started,GPU_wall_hours=0,
            capture_allocation_sha256=sha(a.allocation),metadata_only_source_report_sha256=sha(a.metadata_root/'METADATA_SUPERVISION_COVERAGE.json'))
        (a.out/'ADMISSION.json').write_text(json.dumps(report,indent=2)+'\n')
    except BaseException as e:
        (a.out/'FAILED.json').write_text(json.dumps(dict(status='FAILED_NO_TRAINING_ADMISSION',reason=repr(e),completed_scenes=[r['scene'] for r in results]),indent=2)+'\n');raise

if __name__=='__main__':main()
