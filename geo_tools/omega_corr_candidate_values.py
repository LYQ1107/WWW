"""Lossless per-candidate score index for exact common-valid-candidate comparisons."""
from array import array
import gzip
import json
import zlib
import numpy as np
from omega_corr_common import ROOT,RUNTIME,save_new,sha256,provenance


def main():
    source=provenance();assert not source['source_status'].strip()
    reduced_path=RUNTIME/'reduced_scores_v1.json';reduced=json.loads(reduced_path.read_text())
    cfgs=list(reduced['configurations']);cfg_id={k:i for i,k in enumerate(cfgs)}
    inputs=json.loads(open(json.loads((ROOT/'results/omega_corr_candidate_index_v1.json').read_text())['input']['path']).read())
    queries=[q['query'] for q in inputs['queries']];query_id={tuple(q):i for i,q in enumerate(queries)}
    refs=list(inputs['proposals']);ref_id={k:i for i,k in enumerate(refs)}
    configs=array('i');qids=array('i');rids=array('i');tracks=array('i');scores=array('d');valid=array('b')
    for input in reduced['inputs']:
        assert sha256(input['raw']['path'])==input['raw']['sha256']
        count=0
        with gzip.open(input['raw']['path'],'rt') as f:
            for line in f:
                row=json.loads(line);metric=row['metric'];mode=row['mode'];side=row.get('crop_side',0)
                values=dict(primary=metric.get('primary_score'),cycle=metric.get('cycle_weighted_score'),random_local_assignment=None)
                if mode=='M0':values.update(ROI_mean=metric.get('ROI_cosine'),ROI_align=metric.get('roi_align_cosine'))
                m,n=metric['valid_tokens_A'],metric['valid_tokens_B']
                if min(m,n)>=2:
                    C=np.asarray(row['raw']['cosine'],dtype=np.float32).reshape(m,n)
                    seed=zlib.crc32(json.dumps([row['pair_id'],row['query'],row['reference']],separators=(',',':')).encode())+20261016
                    rng=np.random.default_rng(seed);k=min(m,n);ia=rng.permutation(m)[:k];ib=rng.permutation(n)[:k]
                    values['random_local_assignment']=float(C[ia,ib].mean())*min(k/m,k/n)
                for kind,value in values.items():
                    name='|'.join(map(str,[input['route'],row['backend'],row['variant'],row['layer'],mode,kind,side]))
                    if name not in cfg_id:continue
                    configs.append(cfg_id[name]);qids.append(query_id[tuple(row['query'])]);rids.append(ref_id[':'.join(map(str,row['reference']))]);tracks.append(row['reference_track'])
                    scores.append(value if value is not None else 0.);valid.append(value is not None)
                count+=1
        print(input['route'],input['backend'],count,flush=True)
    arrays=dict(config_index=np.asarray(configs,dtype=np.int32),query_index=np.asarray(qids,dtype=np.int32),
        reference_source_index=np.asarray(rids,dtype=np.int32),reference_track=np.asarray(tracks,dtype=np.int32),
        score_value=np.asarray(scores,dtype=np.float64),score_valid=np.asarray(valid,dtype=bool))
    path=RUNTIME/'candidate_values_v1.npz';np.savez_compressed(path,**arrays)
    save_new(RUNTIME/'candidate_values_v1.json',dict(status='LOSSLESS_CANDIDATE_VALUES_READY',source=source,
        configurations=[dict(config_id=k,**reduced['configurations'][k]) for k in cfgs],queries=queries,reference_sources=refs,
        entries=len(configs),values=dict(path=str(path),sha256=sha256(path)),reduced_sha256=sha256(reduced_path),
        scalar_precision='float64 preserves original JSON/Python score, no new quantization',
        missing_storage='score_value0 with score_valid=False is absent evidence, never measured0; mask mandatory',
        comparison_requirement='Intersect valid physical reference-source sets per actual query across compared methods, then require certified-positive availability separately. Common query alone is insufficient. Report lost-query/candidate denominators and full native-fallback comparison too.'))
    print('PASS candidate entries',len(configs),flush=True)

if __name__=='__main__':main()
