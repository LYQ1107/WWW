"""Native appearance/GTA strong control reuses exact1152-dimensional proposal assets."""
import gzip
import json
import numpy as np
import torch
from omega_corr_common import ROOT,RUNTIME,sha256,save_new,provenance


def main():
    source=provenance();assert not source['source_status'].strip()
    native_manifest=json.loads((RUNTIME/'native_context_full_v1/manifest.json').read_text());assert native_manifest['status']=='MATCHED_NATIVE_CONTEXT_PARITY_PASS'
    assert sha256(native_manifest['proposal_cache']['path'])==native_manifest['proposal_cache']['sha256']
    cache=torch.load(native_manifest['proposal_cache']['path'],map_location='cpu',weights_only=True)
    from read_dev12_native_release import NativeRelease
    release=NativeRelease('/data1/liuyeqiang/WWW_trackvggt_runtime/cv_pil_dev12_native_release_v1','ef7f65d0bccb1e665f7125391f4a3c64a1449b1fe4d204c735371522386421d4')
    assert release.labels is None
    index=json.loads((ROOT/'results/omega_corr_candidate_index_v1.json').read_text());assert sha256(index['input']['path'])==index['input']['sha256']
    native=json.loads(open(index['input']['path']).read());output=RUNTIME/'native_appearance_candidates_v1.jsonl.gz';count=0
    def vector(source):
        if source[0] in cache:v=cache[source[0]]['fields']['reid_features'][source[1]].numpy()
        else:v=release.observations[tuple(source)]['feature'].numpy()
        assert v.shape==(1152,) and np.isfinite(v).all();return v
    with gzip.open(output,'xt') as f:
        for q in native['queries']:
            x=vector(q['query'])
            for r in q['legal_synchronous_references']:
                y=vector(r['source']);values={}
                for dim in [1024,1152]:
                    a,b=x[:dim],y[:dim];den=np.linalg.norm(a)*np.linalg.norm(b)
                    values[str(dim)]=float(a@b/den) if den>1e-12 else None
                f.write(json.dumps(dict(pair_id=q['pair_id'],query=q['query'],reference=r['source'],reference_track=r['track'],cosine=values),allow_nan=False)+'\n');count+=1
    save_new(RUNTIME/'native_appearance_v1.json',dict(status='NATIVE_APPEARANCE_CONTROL_READY',source=source,pairs=160,candidate_pairs=count,
        leading1024='OriginalROIappearance; sourceGTRROIHeads concatenates [reid_features,s_t_feature]',full1152='1024appearance plus128learned spatial-neighborhood context; not pureappearance',
        new_model_calls=0,no_GT_labels_input=True,exact_native_proposal_assets_reused=True,
        output=dict(path=str(output),sha256=sha256(output))))
    print('PASS native appearance candidatepairs',count,flush=True)

if __name__=='__main__':main()
