"""Complete registered placebo/background/ROI review on exact common candidates."""
import collections,csv,gzip,json
import numpy as np
from omega_corr_common import ROOT,RUNTIME,sha256,save_new,provenance
from omega_corr_eval_helpers import top_reference,cluster_interval


def main():
    source=provenance();assert not source['source_status'].strip()
    ev=json.loads((RUNTIME/'evaluation_v1.json').read_text());meta=json.loads((RUNTIME/'candidate_values_v1.json').read_text());assert sha256(meta['values']['path'])==meta['values']['sha256'];archive=np.load(meta['values']['path']);a={k:archive[k] for k in archive.files};archive.close();qs=json.loads((RUNTIME/'native_candidates_v1.json').read_text())['queries'];truth=json.loads((RUNTIME/'candidate_certifications_v1.json').read_text())['rows']
    names={c['config_id']:i for i,c in enumerate(meta['configurations'])};cache={};track={}
    def load(n):
        if n not in cache:
            ix=np.flatnonzero((a['config_index']==names[n])&a['score_valid']);d={}
            for k in ix:
                q,r=int(a['query_index'][k]),int(a['reference_source_index'][k]);d.setdefault(q,{})[r]=float(a['score_value'][k]);track[q,r]=int(a['reference_track'][k])
            cache[n]=d
        return cache[n]
    def status(i,t):return truth[i]['track_status'].get(str(t),'UNKNOWN')
    def choose(i,d):
        r=top_reference(d,{j:track[i,j] for j in d},qs[i]['original_native_ranked_tracks']);return track[i,r] if r is not None else None
    def name(v,l,m='M0',kind='primary',backend='omega'):return f'FULL_IMAGE_MULTIVIEW|{backend}|{v}|{l}|{m}|{kind}|0'
    results=[]
    for l in [11,17,23]:
        for kind in ['primary','cycle']:
            n=name('F2',l,kind=kind);left=load(n)
            controls=[name(v,l,kind=kind) for v in ['F0','F1','F3','N1','N2']]+[name('F2',l,m,kind) for m in ['M1','M2']]+[name('F2',l,kind=k) for k in ['ROI_mean','ROI_align','random_local_assignment']]+[name('old_vggt',l,kind=kind,backend='old_vggt'),name('dino',23,kind=kind,backend='dino')]
            if kind=='cycle':controls.append(name('F2',l))
            for other in controls:
                right=load(other)
                for role in ['FIT','MECHANISM_CHECK','EXPOSED_DIAGNOSTIC_ONLY']:
                    counts=collections.Counter();rows=[];details=[]
                    for i,q in enumerate(qs):
                        if q['role']!=role:continue
                        x,y=left.get(i,{}),right.get(i,{});common=set(x)&set(y);counts['scope_queries']+=1;counts['common_candidates']+=len(common);counts['union_candidates']+=len(set(x)|set(y))
                        if len(common)<2:counts['less_than_two_common_candidates']+=1;continue
                        statuses=[status(i,track[i,j]) for j in common]
                        if 'CORRECT' not in statuses or 'WRONG' not in statuses:counts['no_common_positive_and_negative']+=1;continue
                        tx,ty=choose(i,{j:x[j] for j in common}),choose(i,{j:y[j] for j in common});ox,oy=status(i,tx)=='CORRECT',status(i,ty)=='CORRECT'
                        counts['eligible_queries']+=1;counts['omega_correct']+=ox;counts['control_correct']+=oy;counts['omega_only_correct']+=ox and not oy;counts['control_only_correct']+=oy and not ox;counts['same_chosen_track']+=tx==ty
                        rows.append((q['scene'],truth[i]['query_gt_identity'],int(ox)-int(oy)));details.append(dict(query_index=i,omega_track=tx,control_track=ty,common_candidates=len(common)))
                    results.append(dict(omega=n,control=other,role=role,**counts,paired_interval=cluster_interval(rows),rows=details))
    # Patch bucket grouping uses original full-image physical coverage, never crop-created detail.
    buckets={tuple(map(int,r['source'].split(':'))):r['bucket'] for r in csv.DictReader((ROOT/'results/omega_corr_token_coverage.csv').open())};groups=collections.defaultdict(collections.Counter)
    with gzip.open(ev['query_output']['path'],'rt') as f:
        for row in csv.DictReader(f):
            c=ev['configurations'][int(row['configuration'])];i=int(row['query_index']);key=(int(row['configuration']),row['role'],buckets[tuple(qs[i]['query'])]);g=groups[key];g['queries']+=1;g['with_valid_evidence']+=int(row['valid_candidates'])>0;g['known_corrections']+=row['known_correction']=='True';g['known_harms']+=row['known_harm']=='True'
    save_new(RUNTIME/'control_review_v1.json',dict(status='ALL_REGISTERED_NEGATIVE_AND_SIMPLE_CONTROLS_REVIEWED',source=source,evaluation_sha256=sha256(RUNTIME/'evaluation_v1.json'),comparisons=results,patch_bucket_groups=[dict(configuration=k[0],role=k[1],original_object_patch_bucket=k[2],**v) for k,v in groups.items()],interpretation='Pairwise exact physical candidate intersections. Unknown/mixed retained, and require at least one certified positive and negative. No learned fusion and no GMT solver execution. Additional comparisons are diagnostics; no post-result layer selection.'))
    print('PASS control comparisons',len(results),'bucket groups',len(groups),flush=True)

if __name__=='__main__':main()
