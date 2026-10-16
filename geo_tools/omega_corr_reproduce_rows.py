"""CPU-only independent row audit from published lossless scalars; no model/data path."""
import argparse,json
from pathlib import Path
import numpy as np
from omega_corr_eval_helpers import top_reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
    base=Path(__file__).resolve().parents[1]/'results';meta=json.loads((base/'omega_corr_candidate_values_index.json').read_text());native=json.loads((base/'omega_corr_native_query_index.json').read_text());truth=json.loads((base/'omega_corr_offline_certifications.json').read_text())['rows'];a=np.load(base/'omega_corr_candidate_values.npz');names={c['config_id']:i for i,c in enumerate(meta['configurations'])};cache={};tracks={}
    assert [q['query'] for q in native]==[q['query'] for q in truth]==meta['queries']
    def load(name):
        if name not in cache:
            d={}
            for k in np.flatnonzero((a['config_index']==names[name])&a['score_valid']):
                q,r=int(a['query_index'][k]),int(a['reference_source_index'][k]);d.setdefault(q,{})[r]=float(a['score_value'][k]);tracks[q,r]=int(a['reference_track'][k])
            cache[name]=d
        return cache[name]
    def status(i,t):return truth[i]['track_status'].get(str(t),'UNKNOWN')
    def choose(i,d):
        ref=top_reference(d,{j:tracks[i,j] for j in d},native[i]['original_native_ranked_tracks']);return tracks[i,ref] if ref is not None else truth[i]['native_global_top1']
    rows=[];nets={}
    for layer in [11,17,23]:
        n=f'FULL_IMAGE_MULTIVIEW|omega|F2|{layer}|M0|primary|0';d=load(n)
        for role in ['FIT','MECHANISM_CHECK','EXPOSED_DIAGNOSTIC_ONLY']:
            repairs=[];harms=[]
            for i,q in enumerate(native):
                if q['role']!=role:continue
                before,after=status(i,truth[i]['native_global_top1']),status(i,choose(i,d.get(i,{})))
                if before=='WRONG' and after=='CORRECT':repairs.append(i)
                if before=='CORRECT' and after=='WRONG':harms.append(i)
            rows.append(dict(layer=layer,role=role,known_row_corrections=repairs,known_row_harms=harms,net=len(repairs)-len(harms)))
            if role=='FIT':nets[layer]=len(repairs)-len(harms)
    selected=min([17,11,23],key=lambda l:(-nets[l],[17,11,23].index(l)));expected=json.loads((base/'omega_corr_experiment_manifest.json').read_text());assert selected==expected['selected_layer_fit_only'];assert {str(k):v for k,v in nets.items()}==expected['fit_layer_nets']
    result=dict(status='PUBLISHED_LOSSLESS_ROW_REPRODUCTION_PASS',selected_layer_fit_only=selected,fit_layer_nets=nets,rows=rows,solver_or_tracking_execution=False);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],selected_layer=selected,fit_nets=nets)))

if __name__=='__main__':main()
