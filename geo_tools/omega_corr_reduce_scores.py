"""Reduce every raw local matrix into same-candidate row evidence, retaining missingness."""
import collections
import csv
import gzip
import hashlib
import json
import zlib
import numpy as np
from omega_corr_common import ROOT,RUNTIME,sha256,save_new,provenance
from omega_modules import geometry_module


def main():
    source=provenance();assert not source['source_status'].strip()
    index=json.loads((ROOT/'results/omega_corr_candidate_index_v1.json').read_text());native=json.loads(open(index['input']['path']).read())
    cert_index=json.loads((ROOT/'results/omega_corr_candidate_certification_v1.json').read_text());assert sha256(cert_index['input']['path'])==cert_index['input']['sha256']
    truth={tuple(r['query']):r for r in json.loads(open(cert_index['input']['path']).read())['rows']}
    cohort=json.loads((ROOT/'manifests/omega_corr_cohort_v1.json').read_text());pairs={p['pair_id']:p for p in cohort['pairs']}
    proposals=native['proposals'];query_map={tuple(q['query']):q for q in native['queries']}
    expected=collections.Counter()
    for q in native['queries']:expected[q['pair_id']]+=len(q['legal_synchronous_references'])
    P=geometry_module('omega_adapter').PixelMapping;regions_module=geometry_module('object_patch_regions')
    configurations={};states={};aggregate=collections.defaultdict(lambda:collections.Counter());inputs=[];checked=0;all_records=0
    slim_path=ROOT/'results/omega_corr_candidate_scores.csv'
    fields=['pair_id','query','reference','track','backend','variant','layer','mode','status','M','N','available_A','available_B','mutual_count','mutual_cosine','coverage_A','coverage_B','soft_cycle_A','soft_cycle_B','primary_score','cycle_score','ROI_align','observed_mutual_all_available_A','observed_mutual_all_available_B']
    csv_file=slim_path.open('x',newline='');writer=csv.DictWriter(csv_file,fieldnames=fields,lineterminator='\n');writer.writeheader()
    def config(backend,variant,layer,mode,kind,route,side):
        cid='|'.join(map(str,[route,backend,variant,layer,mode,kind,side]))
        configurations.setdefault(cid,dict(backend=backend,variant=variant,layer=layer,mode=mode,score_kind=kind,route=route,crop_side=side))
        return cid
    def accumulate(cid,row,value):
        key=tuple(row['query']);native_q=query_map[key];certificate=truth[key]
        rank={t:i for i,t in enumerate(native_q['original_native_ranked_tracks'])}
        identity=row['reference_track'];status=certificate['track_status'].get(str(identity),'UNKNOWN_REFERENCE_HISTORY')
        state=states.setdefault((cid,key),dict(best_score=None,best_track=None,valid_candidates=0,positive_best=None,negative_best=None,unknown_candidates=0))
        if value is None:return
        assert np.isfinite(value)
        state['valid_candidates']+=1;state['unknown_candidates']+=status not in ['CORRECT','WRONG']
        if state['best_score'] is None or value>state['best_score'] or (value==state['best_score'] and rank.get(identity,10**9)<rank.get(state['best_track'],10**9)):
            state['best_score']=value;state['best_track']=identity
        if status in ['CORRECT','WRONG']:
            field='positive_best' if status=='CORRECT' else 'negative_best'
            if state[field] is None or value>state[field]:state[field]=value
    for route,backend in [('FULL_IMAGE_MULTIVIEW','omega'),('FULL_IMAGE_MULTIVIEW','old_vggt'),('FULL_IMAGE_MULTIVIEW','dino'),('LOCAL_CROP_VISUAL','omega'),('LOCAL_CROP_VISUAL','dino')]:
        directory=RUNTIME/('extraction' if route=='FULL_IMAGE_MULTIVIEW' else 'local_crops')/backend
        manifest_path=directory/'manifest.json';manifest=json.loads(manifest_path.read_text());raw_path=manifest['output']['path'];assert sha256(raw_path)==manifest['output']['sha256']
        inputs.append(dict(route=route,backend=backend,manifest_sha256=sha256(manifest_path),raw=dict(path=raw_path,sha256=sha256(raw_path))))
        maps={r['pair_id']:r['mappings'] for r in manifest['pair_receipts']} if route=='FULL_IMAGE_MULTIVIEW' else {}
        geometry_cache={};count=0
        with gzip.open(raw_path,'rt') as stream:
            for line in stream:
                row=json.loads(line);count+=1;all_records+=1
                key=tuple(row['query']);q=query_map[key];assert q['pair_id']==row['pair_id']
                assert any(r['source']==row['reference'] and r['track']==row['reference_track'] for r in q['legal_synchronous_references'])
                C=np.asarray(row['raw']['cosine'],dtype=np.float32).reshape(row['metric']['valid_tokens_A'],row['metric']['valid_tokens_B'])
                m,n=C.shape;metric=row['metric'];mode=row['mode'];side=row.get('crop_side',0)
                assert len(set(row['patch_ids_A']))==m and len(set(row['patch_ids_B']))==n
                if route=='FULL_IMAGE_MULTIVIEW':
                    for label,physical in [('A',row['query']),('B',row.get('mapped_region_source',row['reference']))]:
                        prop_key=':'.join(map(str,physical));cache_key=(row['pair_id'],prop_key)
                        if cache_key not in geometry_cache:
                            image_index=next(i for i,v in enumerate(pairs[row['pair_id']]['images']) if v['image_id']==physical[0])
                            record=maps[row['pair_id']][image_index];mapping=P(**{k:record[k] for k in P.__dataclass_fields__})
                            geometry_cache[cache_key]=regions_module.object_regions(mapping,proposals[prop_key]['bbox_xyxy'],patch_size=16 if backend=='omega' else 14)
                        region=geometry_cache[cache_key]['regions'][mode]
                        assert region['patch_ids'].tolist()==row['patch_ids_'+label]
                        np.testing.assert_allclose(np.asarray(row['original_centers_'+label]).reshape(-1,2),region['original_centers'],atol=1e-10)
                    available_A=geometry_cache[row['pair_id'],':'.join(map(str,row['query']))]['regions'][mode]['available_count']
                    rsource=row.get('mapped_region_source',row['reference']);available_B=geometry_cache[row['pair_id'],':'.join(map(str,rsource))]['regions'][mode]['available_count']
                else:available_A=row['available_token_count_A'];available_B=row['available_token_count_B']
                assert m<=available_A and n<=available_B
                if min(m,n)<2:assert metric['primary_score'] is None and metric['hard_cycle_A'] is None
                else:
                    ab=C.argmax(1);ba=C.argmax(0);ii=np.flatnonzero(ba[ab]==np.arange(m));jj=ab[ii]
                    assert len(ii)==metric['mutual_count']
                    assert metric['mutual_coverage_A']==len(ii)/m and metric['mutual_coverage_B']==len(ii)/n
                    assert metric['hard_cycle_A']==metric['mutual_coverage_A']
                    assert abs(float(C[ii,jj].mean())-metric['mean_mutual_similarity'])<1e-6
                    checked+=1
                cid_base=(backend,row['variant'],row['layer'],mode)
                scores={'primary':metric.get('primary_score'),'cycle':metric.get('cycle_weighted_score')}
                if mode=='M0':scores.update(ROI_mean=metric.get('ROI_cosine'),ROI_align=metric.get('roi_align_cosine'))
                if min(m,n)>=2:
                    seed=zlib.crc32(json.dumps([row['pair_id'],row['query'],row['reference']],separators=(',',':')).encode())+20261016
                    rng=np.random.default_rng(seed);k=min(m,n);ia=rng.permutation(m)[:k];ib=rng.permutation(n)[:k]
                    scores['random_local_assignment']=float(C[ia,ib].mean())*min(k/m,k/n)
                else:scores['random_local_assignment']=None
                for kind,value in scores.items():accumulate(config(*cid_base,kind,route,side),row,value)
                for status in [truth[key]['track_status'].get(str(row['reference_track']),'UNKNOWN_REFERENCE_HISTORY')]:
                    g=aggregate[(route,backend,row['variant'],row['layer'],mode,side,status)]
                    g['candidate_pairs']+=1;g['selected_tokens_A']+=m;g['selected_tokens_B']+=n;g['available_tokens_A']+=available_A;g['available_tokens_B']+=available_B
                    g['mutual_pairs']+=metric['mutual_count'];g['point_available_pairs']+=min(m,n)>=2
                    if metric['mean_mutual_similarity'] is not None:g['mutual_similarity_sum']+=metric['mean_mutual_similarity']
                    if metric['soft_cycle_A'] is not None:g['soft_cycle_A_sum']+=metric['soft_cycle_A'];g['soft_cycle_B_sum']+=metric['soft_cycle_B']
                if route=='FULL_IMAGE_MULTIVIEW' and row['variant'] in ['F2','old_vggt','dino'] and mode=='M0':
                    writer.writerow(dict(pair_id=row['pair_id'],query=':'.join(map(str,row['query'])),reference=':'.join(map(str,row['reference'])),track=row['reference_track'],backend=backend,variant=row['variant'],layer=row['layer'],mode=mode,status=metric['status'],M=m,N=n,available_A=available_A,available_B=available_B,mutual_count=metric['mutual_count'],mutual_cosine=metric.get('mean_mutual_similarity'),coverage_A=metric.get('mutual_coverage_A'),coverage_B=metric.get('mutual_coverage_B'),soft_cycle_A=metric.get('soft_cycle_A'),soft_cycle_B=metric.get('soft_cycle_B'),primary_score=metric.get('primary_score'),cycle_score=metric.get('cycle_weighted_score'),ROI_align=metric.get('roi_align_cosine'),observed_mutual_all_available_A=metric['mutual_count']/available_A if available_A else None,observed_mutual_all_available_B=metric['mutual_count']/available_B if available_B else None))
        assert count==manifest['records'];print(route,backend,count,'reduced',flush=True)
    csv_file.close()
    result=dict(status='ALL_FROZEN_CONFIGURATIONS_REDUCED',source=source,inputs=inputs,raw_records=all_records,independent_mutual_matrix_checks=checked,
        configurations=configurations,rows=[dict(config_id=cid,query=list(q),**v) for (cid,q),v in states.items()],
        aggregate=[dict(route=k[0],backend=k[1],variant=k[2],layer=k[3],mode=k[4],crop_side=k[5],candidate_status=k[6],**v) for k,v in aggregate.items()],
        full_query_count=len(native['queries']),all_available_patch_denominators_retained=True)
    save_new(RUNTIME/'reduced_scores_v1.json',result)
    save_new(ROOT/'results/omega_corr_reduction_audit_v1.json',dict(status='PASS',raw_records=all_records,independent_mutual_checks=checked,
        configs=len(configurations),input_receipts=inputs,output=dict(path=str(RUNTIME/'reduced_scores_v1.json'),sha256=sha256(RUNTIME/'reduced_scores_v1.json')),
        candidate_CSV_scope='Fullimage M0 F2/oldVGGT/DINO, everyselected layer and allcandidatefailures. Complete allvariants/modes matrices remain hashed rawinputs; allquery config outputs inpairedcontrols.',candidate_CSV_sha256=sha256(slim_path)))
    print('PASS',all_records,checked,len(configurations),flush=True)

if __name__=='__main__':main()
