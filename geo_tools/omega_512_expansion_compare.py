"""All preregistered arms on conditional remaining72; uniform gate separate from TRAIN hard."""
import collections
import csv
import json
from pathlib import Path
import time
import numpy as np
from omega_common import ROOT, RUNTIME, NATIVE, jsonl, provenance, save_new, sha256
from omega_modules import geometry_module
from omega_background_audit import distinct_matches


def joint_support(rows, arm, method):
    groups = collections.defaultdict(dict)
    for row in rows:
        if row['arm']==arm and row['method']==method and row['order'] in ('AB','BA'):
            groups[row['pair_id']][row['order']]=row
    valid=[v['AB'] for v in groups.values() if set(v)=={'AB','BA'} and v['AB']['valid'] and v['BA']['valid']]
    scenes=sorted({r['scene'] for r in valid})
    return dict(joint_valid_pairs=len(valid),scenes=scenes,pair_ids=[r['pair_id'] for r in valid],
                passes_pilot_gate=len(valid)>=3 and len(scenes)>=2)


def main():
    import cv2
    cv2.setNumThreads(4)
    plan_path=ROOT/'manifests/omega512_independent_preregistration_v1.json'
    plan=json.loads(plan_path.read_text())
    assert sha256(plan_path)=='890fe639e67e844454330330d028217fb4530bf79901f4908d9e2bad0267cca7'
    for path,expected in plan['frozen_inputs'].items():assert sha256(ROOT/path)==expected,path
    source=provenance()
    assert not source['source_status'].strip(),'clean pinned source required'
    source['geometry_confidence_sha256']=sha256(ROOT/'gtr/geometry/geometry_confidence.py')
    out=RUNTIME/'omega512_independent_v1/expansion_comparison'
    out.mkdir(parents=True,exist_ok=False)
    cohort=json.loads((ROOT/'manifests/omega_fixed_pair_cohort_v1.json').read_text())
    pairs=[p for p in cohort['pairs'] if p['stage']!='initial24']
    assert len(pairs)==72
    cached={}
    for tag in ['omega512_remaining72_v1','old_vggt_remaining72_v1']:
        d=RUNTIME/'pose_audits'/tag
        m=json.loads((d/'manifest.json').read_text())
        assert m['pairs']==72 and sha256(m['outputs']['path'])==m['outputs']['sha256']
        cached[tag]={r['pair_id']:r for r in map(json.loads,Path(m['outputs']['path']).read_text().splitlines())}
    geom=geometry_module('omega_geometry'); confidence=geometry_module('geometry_confidence')
    adapter=geometry_module('omega_adapter')
    wanted={i['image_id'] for p in pairs for i in p['images']}
    boxes=collections.defaultdict(list);detector_sources=[]
    for scene in sorted({p['scene'] for p in pairs}):
        path=NATIVE/f'scene{scene}_full_v1/detector_provenance.jsonl.gz'
        receipt=next(r for r in cohort['native_source_receipts'] if Path(r['path']).parent.name==f'scene{scene}_full_v1')
        assert sha256(path)==receipt['artifacts']['detector_provenance.jsonl.gz']
        detector_sources.append(dict(path=str(path),sha256=sha256(path)))
        for row in jsonl(path):
            if row['source'][0] in wanted:
                x,y,w,h=row['detection']['bbox'];boxes[row['source'][0]].append([x,y,x+w,y+h])
    rows=[]
    pair_reports=[];started=time.perf_counter()
    with (out/'pair_results.jsonl').open('x') as stream:
        for arm in plan['arms']:
            for pair in pairs:
                begin=time.perf_counter();images=pair['images'];pair_id=pair['pair_id']
                assert all(sha256(i['path'])==i['sha256'] for i in images)
                gray=[cv2.imread(i['path'],cv2.IMREAD_GRAYSCALE) for i in images]
                assert all(a is not None for a in gray)
                sizes=[(i['original_width'],i['original_height']) for i in images]
                b=[np.asarray(boxes[i['image_id']],dtype=float).reshape(-1,4) for i in images]
                cv2.setRNGSeed(20261016)
                background=confidence.estimate_background(*gray,*b,method=arm['method'],mask_targets=True,
                                                          nfeatures=arm['nfeatures'],mutual=arm['mutual'])
                p,q=background['points1'],background['points2']
                raw=len(p);keep=np.ones(raw,dtype=bool)
                # Raw detector-excluded coordinates must also lie in model-covered regions.
                for physical,m in enumerate(cached['omega512_remaining72_v1'][pair_id]['outputs']['AB']['mappings']):
                    mapping=adapter.PixelMapping(**{k:m[k] for k in adapter.PixelMapping.__dataclass_fields__})
                    _,valid=mapping.forward((p,q)[physical]);keep&=valid
                p,q=distinct_matches(p[keep],q[keep])
                blocks=np.floor(p/np.array(sizes[0])*4).astype(int).clip(0,3)
                fit=(blocks[:,0]+blocks[:,1])%2==0
                pf,qf,pe,qe=p[fit],q[fit],p[~fit],q[~fit]
                corr=out/(arm['id']+'_'+pair_id+'.npz')
                np.savez_compressed(corr,fit_points_A=pf,fit_points_B=qf,eval_points_A=pe,eval_points_B=qe)
                Fref=None
                if len(pf)>=8:
                    cv2.setRNGSeed(20261016)
                    Fref,_=cv2.findFundamentalMat(pf,qf,cv2.FM_RANSAC,3.,.99)
                    if Fref is not None and Fref.shape!=(3,3):Fref=None
                models=[]
                for method,tag in [('omega512','omega512_remaining72_v1'),('old_vggt','old_vggt_remaining72_v1')]:
                    for order in ['AB','BA']:
                        output=cached[tag][pair_id]['outputs'][order]
                        raw_F=output.get('F_original') if method=='omega512' else output.get('F_in_physical_BA')
                        F=np.array(raw_F) if raw_F is not None else None
                        if F is not None and method=='omega512' and order=='BA':F=F.T
                        models.append((method,order,F))
                models.append(('traditional_RANSAC','AB',Fref))
                measured=[]
                for method,order,F in models:
                    m=geom.background_validity(F,pe,qe,sizes)
                    r=dict(arm=arm['id'],pair_id=pair_id,scene=pair['scene'],group=pair['group'],method=method,
                           order=order,valid=m['valid'],eval_matches=len(pe),fit_matches=len(pf),
                           median_original_px=m['median_residual'],p90_original_px=m['p90_residual'],
                           inlier_ratio=m['inlier_ratio'],original_area_coverage=m['coverage'],
                           reasons='|'.join(m['reasons']),source='NEW_MATCHER_REUSED_FROZEN_CAMERAS')
                    rows.append(r);measured.append(r)
                item=dict(arm=arm['id'],pair_id=pair_id,keypoints=[background['keypoints1'],background['keypoints2']],
                          ratio_test_matches=background['matches'],retained_raw_matches=raw,distinct_matches=len(p),
                          fit_matches=len(pf),eval_matches=len(pe),correspondences=dict(path=str(corr),sha256=sha256(corr)),
                          pair_seconds=time.perf_counter()-begin,metrics=measured)
                pair_reports.append(item);stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush()
                print(json.dumps(dict(arm=arm['id'],pair_id=pair_id,eval_matches=len(pe),Omega_valid=[r['valid'] for r in measured if r['method']=='omega512'])),flush=True)
    summaries={arm['id']:{method:joint_support(rows,arm['id'],method) for method in ['omega512','old_vggt']} for arm in plan['arms']}
    uniform_ids={p['pair_id'] for p in pairs if p['sampling']=='uniform_without_GT'}
    assert len(uniform_ids)==42
    uniform_summary=joint_support([r for r in rows if r['pair_id'] in uniform_ids],'C2_sift6000_mutual','omega512')
    admitted=['C2_sift6000_mutual'] if uniform_summary['passes_pilot_gate'] else []
    columns=['arm','pair_id','scene','group','method','order','valid','fit_matches','eval_matches','median_original_px',
             'p90_original_px','inlier_ratio','original_area_coverage','reasons','source']
    with (out/'comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    manifest=dict(schema='omega512-background-recovery/v1',status='UNIFORM_IDENTITY_ADMISSION_PASSED' if admitted else 'UNIFORM_GEOMETRY_GATE_FAILED',
                  role='INDEPENDENT_512_STUDY_NOT_416_REPRODUCTION',preregistration=dict(path=str(plan_path),sha256=sha256(plan_path)),
                  pairs_per_arm=72,new_matcher_arms=3,retained_baseline_arm=False,uniform_admission=uniform_summary,all_failure_pairs_retained=True,
                  unchanged_background_gate=plan['common_protocol']['background_gate'],summary=summaries,admitted_arms=admitted,
                  new_model_calls=0,GT_input=False,new_GMT_inference=False,detector_sources=detector_sources,
                  pair_rows=dict(path=str(out/'pair_results.jsonl'),sha256=sha256(out/'pair_results.jsonl')),
                  comparison=dict(path=str(out/'comparison.csv'),sha256=sha256(out/'comparison.csv')),
                  pair_reports=pair_reports,rows=rows,wall_after_setup_seconds=time.perf_counter()-started,
                  opencv=cv2.__version__,numpy=np.__version__,no_PER_pair_best_arm_selection=True,
                  expansion='CONDITIONAL_ADMITTED_ONLY' if admitted else 'NOT_EXECUTED_GATE_FAILED',
                  identity_probe='NOT_EXECUTED',online_fusion='NOT_EXECUTED',full_tracking='NOT_EXECUTED',
                  source=source)
    save_new(out/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],summary=summaries,uniform_admission=uniform_summary,admitted_arms=admitted)),flush=True)


if __name__=='__main__':main()
