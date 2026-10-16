"""Independent cached background-F compatibility; never gates visual token extraction."""
import csv
import gzip
import json
import numpy as np
from omega_corr_common import ROOT,RUNTIME,save_new,sha256,provenance
from omega_modules import geometry_module


def main():
    source=provenance();assert not source['source_status'].strip()
    pilot=json.loads((ROOT/'results/omega512_independent_evidence_v1.json').read_text());exp=json.loads((ROOT/'results/omega512_expansion_evidence_v1.json').read_text())
    valid=set(pilot['manifest']['summary']['C2_sift6000_mutual']['omega512']['pair_ids'])|set(exp['manifest']['summary']['C2_sift6000_mutual']['omega512']['pair_ids'])
    numeric=json.loads((ROOT/'results/omega_numeric_evidence_v1.json').read_text())
    cameras={r['pair_id']:r for r in numeric['runs']['omega512_initial24_v1']['rows']}
    cameras.update({r['pair_id']:r for r in exp['models']['omega512_remaining72_v1']['rows']})
    index=json.loads((ROOT/'results/omega_corr_candidate_index_v1.json').read_text());native=json.loads(open(index['input']['path']).read());meta=native['proposals']
    manifest=json.loads((RUNTIME/'extraction/omega/manifest.json').read_text());assert sha256(manifest['output']['path'])==manifest['output']['sha256']
    distance=geometry_module('geometry_alignment').symmetric_epipolar_distance
    output=RUNTIME/'geometry_support_v1.jsonl.gz';stats=dict(candidate_records=0,known_F_records=0,unknown_F_records=0,known_F_with_multipoint_match=0)
    with gzip.open(manifest['output']['path'],'rt') as f,gzip.open(output,'xt') as out:
        for line in f:
            row=json.loads(line)
            if row['variant']!='F2' or row['mode']!='M0':continue
            item=dict(pair_id=row['pair_id'],query=row['query'],reference=row['reference'],reference_track=row['reference_track'],layer=row['layer'],
                visual_evidence_type='VISUAL_CORRESPONDENCE_EVIDENCE',physical_same_surface_verification='NOT_ESTABLISHED',
                F_support='BACKGROUND_F_UNKNOWN',median_residual_original_px=None,inlier_fraction_12px=None,B10_score=None,mutual_count=row['metric']['mutual_count'])
            stats['candidate_records']+=1
            if row['pair_id'] in valid:
                stats['known_F_records']+=1;item['F_support']='HELDOUT_BACKGROUND_F_AVAILABLE'
                if row['metric']['status']=='LOCAL_VISUAL_MATCH_MEASURED':
                    stats['known_F_with_multipoint_match']+=1
                    ia=np.array(row['raw']['mutual_indices_A'],int);ib=np.array(row['raw']['mutual_indices_B'],int)
                    q=np.asarray(row['original_centers_A'])[ia];r=np.asarray(row['original_centers_B'])[ib]
                    camera=meta[':'.join(map(str,row['query']))]['camera'];pA,pB=(q,r) if camera==1 else (r,q)
                    old=cameras[row['pair_id']]['outputs'];F=np.asarray(old['AB']['F_original']);reverse=np.asarray(old['BA']['F_original']).T
                    error=np.maximum(distance(F,pA,pB),distance(reverse,pA,pB));assert np.isfinite(error).all()
                    fraction=float(np.mean(error<=12.));item.update(median_residual_original_px=float(np.median(error)),
                        inlier_fraction_12px=fraction,B10_score=row['metric']['cycle_weighted_score']*fraction,
                        F_support='EPIPOLAR_COMPATIBILITY_MEASURED',mutual_residuals_original_px=error.tolist())
            else:stats['unknown_F_records']+=1
            out.write(json.dumps(item,allow_nan=False,separators=(',',':'))+'\n')
    save_new(RUNTIME/'geometry_support_v1.json',dict(status='CACHED_INDEPENDENT_F_SUPPORT_MEASURED',source=source,stats=stats,
        reliable_original_pairs=sorted(valid),new_F_model_calls=0,no_visual_pairs_excluded=True,
        physical_same_body_surface_GT='NOT_AVAILABLE',interpretation='Background-gated predicted F supplies independent target-token epipolar compatibility, not proof of same3D point. Unreliable/absent F is UNKNOWN, never a visual failure.',
        B10_scope='Only reliable13 subset, compare B8/B9 on same exact subset; do not compare selectively gated13 with all160 as a model superiority claim.',
        output=dict(path=str(output),sha256=sha256(output))))
    print(stats,flush=True)

if __name__=='__main__':main()
