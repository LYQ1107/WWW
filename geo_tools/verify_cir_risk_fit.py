"""Verify real saved FIT checkpoints and join GT-free native edge eligibility.

Score threshold crossings without an accepted native edge are not vetoes.
No new training, native future, development metric or tracking claim.
"""
import argparse
import gzip
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from common import sha256,save_json
from cir_risk_training import read_risk_training
from gtr.modeling.identity_revision.model import load_native_risk


def main():
    p=argparse.ArgumentParser();p.add_argument('--training-report',type=Path,required=True)
    p.add_argument('--events',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    began=time.monotonic();report=json.loads(a.training_report.read_text())
    if report['status']!='COMPLETE_FIRST_SEED_FIT12_RISK_ONLY_NATIVE_ONLINE_PENDING':raise ValueError('Actual complete first fit required')
    if sha256(a.events)!='106f5e7efce7b0ed08bb1f327d8aa30696ef65dbb5b29fadbdaeec88a44edc74':raise ValueError('Native policy event SHA drift')
    data=read_risk_training(Path(report['data']['path']),report['data']['sha256'],set(report['coverage']['all_source_video_ids']))
    eligible=[]
    with gzip.open(report['data']['path'],'rt') as risk,gzip.open(a.events,'rt') as policy:
        for r,p in itertools.zip_longest(risk,policy):
            if r is None or p is None:raise ValueError('Risk and native event coverage differs')
            r=json.loads(r);p=json.loads(p)
            if r['key']!=p['key'] or r['features']!=p['risk_features']:raise ValueError('Native input/source mismatch')
            eligible.append(p['C1_primary_id']>=1)
    eligible=np.asarray(eligible);known=np.zeros(len(eligible),dtype=bool);known[data['known_indices'].numpy()]=True
    if not eligible[known].all():raise ValueError('Known risk label attached to nonaccepted native edge')
    models={};normalization=None
    for name,fit in report['models'].items():
        before=torch.get_rng_state().clone();model,threshold=load_native_risk(fit['checkpoint']['path'],fit['checkpoint']['sha256'])
        if not torch.equal(before,torch.get_rng_state()):raise RuntimeError('Actual checkpoint load consumed RNG')
        if threshold!=fit['fitting_calibration']['threshold'] or sum(p.numel() for p in model.parameters())!=961:
            raise ValueError('Frozen checkpoint threshold or capacity mismatch')
        current=(model.feature_mean,model.feature_scale)
        if normalization is not None and any(not torch.equal(x,y) for x,y in zip(normalization,current)):raise ValueError('Normalization differs across heads')
        normalization=current;scores=[]
        with torch.inference_mode():
            for start in range(0,len(eligible),8192):scores.append(model(data['all_features'][start:start+8192]).numpy())
        scores=np.concatenate(scores);selected=scores>threshold;yp=data['labels'].numpy();pred=selected[known]
        counts=dict(TP=int((pred&(yp==1)).sum()),FP=int((pred&(yp==0)).sum()),FN=int((~pred&(yp==1)).sum()),TN=int((~pred&(yp==0)).sum()))
        if any(counts[k]!=fit['fitting_calibration'][k] for k in counts):raise RuntimeError('Saved checkpoint cannot reproduce actual fit classification')
        models[name]=dict(reproduced_fit_counts=counts,all_scores_above_threshold=int(selected.sum()),
            accepted_native_edges_above_threshold=int((selected&eligible).sum()),
            accepted_GT_UNKNOWN_edges_above_threshold=int((selected&eligible&~known).sum()),
            no_accepted_edge_above_threshold=int((selected&~eligible).sum()),
            actual_load_RNG_preserved=True,checkpoint_sha256=fit['checkpoint']['sha256'])
    result=dict(status='PASS_SAVED_FIT_CHECKPOINTS_AND_NATIVE_EDGE_ELIGIBILITY',models=models,
        accepted_native_queries=int(eligible.sum()),all_queries=len(eligible),normalization_identical=True,
        scope='same original-state cached query/edge flags, not actual mutated-state veto counts or tracking metrics',
        source_sha256=sha256(__file__),training_report_sha256=sha256(a.training_report),policy_events_sha256=sha256(a.events),
        CPU_wall_seconds=time.monotonic()-began,GPU_wall_hours=0,scientific_success=False)
    save_json(a.out,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
