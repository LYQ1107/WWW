"""Recompute and publish independent512 remaining72 all-arm heldout comparisons."""
import collections
import json
import math
from pathlib import Path
import shutil
import numpy as np
from omega_common import ROOT,RUNTIME,sha256,save_new
from omega_modules import geometry_module
from omega_512_background_recovery import joint_support

def main():
    run=RUNTIME/'omega512_independent_v1/expansion_comparison'
    manifest=json.loads((run/'manifest.json').read_text())
    receipt_path=RUNTIME/'omega512_independent_v1/logs/remaining72_compare_receipt.json'
    receipt=json.loads(receipt_path.read_text())
    assert receipt['exit_code']==0 and receipt['CUDA_VISIBLE_DEVICES']==''
    assert sha256(receipt['log']['path'])==receipt['log']['sha256']
    plan=json.loads((ROOT/'manifests/omega512_independent_preregistration_v1.json').read_text())
    for p,h in plan['frozen_inputs'].items():assert sha256(ROOT/p)==h
    cached={}
    model_receipts={}
    for tag,short in [('omega512_remaining72_v1','omega'),('old_vggt_remaining72_v1','old')]:
        d=RUNTIME/'pose_audits'/tag
        m=json.loads((d/'manifest.json').read_text())
        assert m['pairs']==72 and m['actual_model_calls_completed']==144
        assert sha256(m['outputs']['path'])==m['outputs']['sha256']
        cached[tag]={r['pair_id']:r for r in map(json.loads,Path(m['outputs']['path']).read_text().splitlines())}
        r=json.loads((RUNTIME/f'omega512_independent_v1/logs/remaining72_{short}_receipt.json').read_text())
        assert r['exit_code']==0 and r['CUDA_VISIBLE_DEVICES']=='4'
        assert sha256(r['log']['path'])==r['log']['sha256']
        model_receipts[tag]=dict(manifest=m,process_clock=r,rows=list(cached[tag].values()))
    assert sum(v['process_clock']['child_process_wall_seconds'] for v in model_receipts.values())<1800
    cohort=json.loads((ROOT/'manifests/omega_fixed_pair_cohort_v1.json').read_text())
    pairs={p['pair_id']:p for p in cohort['pairs'] if p['stage']!='initial24'}
    geom=geometry_module('omega_geometry')
    import cv2
    checks=0;correspondences={}
    assert len(manifest['pair_reports'])==216
    for report in manifest['pair_reports']:
        entry=report['correspondences']; assert sha256(entry['path'])==entry['sha256']
        with np.load(entry['path']) as z: a={k:z[k].copy() for k in z.files}
        correspondences[report['arm']+'/'+report['pair_id']]={k:v.tolist() for k,v in a.items()}
        p,q=a['eval_points_A'],a['eval_points_B'];pf,qf=a['fit_points_A'],a['fit_points_B']
        Fref=None
        if len(pf)>=8:
            cv2.setRNGSeed(20261016);Fref,_=cv2.findFundamentalMat(pf,qf,cv2.FM_RANSAC,3.,.99)
            if Fref is not None and Fref.shape!=(3,3):Fref=None
        for row in report['metrics']:
            F=Fref
            if row['method']!='traditional_RANSAC':
                tag='omega512_remaining72_v1' if row['method']=='omega512' else 'old_vggt_remaining72_v1'
                o=cached[tag][row['pair_id']]['outputs'][row['order']]
                raw_F=o.get('F_original') if row['method']=='omega512' else o.get('F_in_physical_BA')
                F=np.array(raw_F) if raw_F is not None else None
                if F is not None and row['method']=='omega512' and row['order']=='BA':F=F.T
            pair=pairs[row['pair_id']];sizes=[(i['original_width'],i['original_height']) for i in pair['images']]
            m=geom.background_validity(F,p,q,sizes)
            assert m['valid']==row['valid']
            for key,field in [('median_residual','median_original_px'),('p90_residual','p90_original_px'),('coverage','original_area_coverage'),('inlier_ratio','inlier_ratio')]:
                x,y=m[key],row[field]
                assert x==y or (isinstance(x,(int,float)) and isinstance(y,(int,float)) and math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-10))
            checks+=1
    assert checks==1080
    for arm in plan['arms']:
        for method in ['omega512','old_vggt']:
            assert joint_support(manifest['rows'],arm['id'],method)==manifest['summary'][arm['id']][method]
    uniform_ids={p['pair_id'] for p in pairs.values() if p['sampling']=='uniform_without_GT'}
    assert len(uniform_ids)==42
    assert joint_support([r for r in manifest['rows'] if r['pair_id'] in uniform_ids],'C2_sift6000_mutual','omega512')==manifest['uniform_admission']
    evidence=dict(manifest=manifest,process_clock=receipt,models=model_receipts,heldout_numeric_recomputations=checks,
                  all_new_correspondences=correspondences,source_tests=(RUNTIME/'omega512_independent_v1/logs/source_tests.log').read_text())
    dest=ROOT/'results/omega512_expansion_evidence_v1.json';save_new(dest,evidence)
    shutil.copyfile(run/'comparison.csv',ROOT/'results/omega512_expansion_background_v1.csv')
    print(json.dumps(dict(status=manifest['status'],summary=manifest['summary'],recomputed=checks)),flush=True)

if __name__=='__main__':main()
