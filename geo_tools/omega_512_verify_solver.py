"""Recompute native activation and actual Hungarian/threshold outcomes from small fork matrices."""
import collections
import itertools
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from omega_common import ROOT,RUNTIME,save_new,sha256


def main():
    run=RUNTIME/'omega512_independent_v1/solver_forks'
    manifest=json.loads((run/'manifest.json').read_text());assert sha256(manifest['output']['path'])==manifest['output']['sha256']
    calls=0;queries=0;memory_calls=0
    for line in open(manifest['output']['path']):
        row=json.loads(line)
        decisions=[d for d in row['decisions'] if d['stage']!='run_memory_tracker']
        memory_calls+=sum(d['stage']=='run_memory_tracker' for d in row['decisions'])
        assert len(decisions)==len(row['matrices'])
        for d,m in zip(decisions,row['matrices']):
            assert d['queries']==m['queries'] and d['references']==m['references']
            nq,nr=len(m['queries']),len(m['references'])
            raw=np.array(m['raw_logits'],dtype=np.float32).reshape(nq,nr)
            delta=np.array(m['delta'],dtype=np.float32).reshape(nq,nr)
            legal=np.array(m['legal'],dtype=bool).reshape(nq,nr)
            assert np.isfinite(raw).all() and np.isfinite(delta).all() and np.all(np.abs(delta)<=1.)
            assert np.all(delta[~legal]==0)
            if row['weight']==0:assert np.all(delta==0) and m['unchanged_object']
            ref_ids=np.array([r['track'] for r in m['references']]);unique=np.array(m['unique_tracks'])
            membership=(ref_ids[:,None]==unique[None,:]).astype(np.float32)
            blocks=[len(list(g)) for _,g in itertools.groupby(m['references'],key=lambda r:r['source'][0])]
            for values,field in [(raw,'before_track_scores'),(raw+delta,'after_track_scores')]:
                probs=[];start=0
                for n in blocks:
                    x=np.c_[values[:,start:start+n],np.zeros(nq,dtype=np.float32)]
                    x=np.exp(x-x.max(1,keepdims=True));x=x/x.sum(1,keepdims=True)
                    probs.append(x[:,:-1]);start+=n
                prob=np.concatenate(probs,axis=1) if probs else np.zeros((nq,0),np.float32)
                scores=prob@membership
                stored=np.array(m[field],dtype=np.float32).reshape(nq,len(unique))
                np.testing.assert_allclose(scores,stored,rtol=2e-5,atol=2e-6)
            scores=np.array(m['after_track_scores'],dtype=np.float32).reshape(nq,len(unique))
            ri,ci=linear_sum_assignment(-scores)
            expected=[(int(i),int(unique[j])) for i,j in zip(ri,ci)]
            observed=[(p['row'],p['track']) for p in d['hungarian_pairs']]
            assert expected==observed
            accepted=[-1]*nq
            for i,j,p in zip(ri,ci,d['hungarian_pairs']):
                assert float(scores[i,j])==p['score']
                passes=float(scores[i,j])>p['threshold'];assert passes==p['accepted']
                if passes:accepted[int(i)]=int(unique[j])
            assert accepted==d['accepted_track_ids']
            # Original tracker retains every primary acceptance after Bank/new-ID fallback.
            assert all(a<0 or a==f for a,f in zip(accepted,d['final_track_ids']))
            calls+=1;queries+=nq
    save_new(ROOT/'results/omega512_solver_verification_v1.json',dict(status='PASS',primary_calls=calls,queries=queries,
        memory_calls_observed=memory_calls,checks=['null-softmax per native image block','track membership aggregation',
        'stored exact float32 Hungarian assignment','strict selected-pair threshold','primary acceptance survives Bank/final-ID allocation',
        'bounded finite delta on legal references','zero branch exact object identity'],
        raw_evidence=dict(path=manifest['output']['path'],sha256=manifest['output']['sha256']),
        final_postprocessed_ID_check='separate identity adjudication binds each primary final ID to public source-index output',
        limits='Memory averaged-feature logits not persisted; actual Bank assignments and primary/final changes observed through original methods. No sustained fullvideo benefit or FPS claim.'))
    print('PASS',calls,queries,memory_calls)

if __name__=='__main__':main()
