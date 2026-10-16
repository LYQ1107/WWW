"""Retain full local metric denominators and independently verify soft returns."""
import collections,gzip,json
import numpy as np
from omega_corr_common import RUNTIME,save_new,sha256,provenance


def main():
    source=provenance();assert not source['source_status'].strip();reduced=json.loads((RUNTIME/'reduced_scores_v1.json').read_text());truth={tuple(r['query']):r for r in json.loads((RUNTIME/'candidate_certifications_v1.json').read_text())['rows']};agg=collections.defaultdict(collections.Counter);checks=0;measured=0;max_error=0.
    for file in reduced['inputs']:
        assert sha256(file['raw']['path'])==file['raw']['sha256']
        with gzip.open(file['raw']['path'],'rt') as f:
            for line in f:
                row=json.loads(line);m=row['metric'];t=truth[tuple(row['query'])];status=t['track_status'].get(str(row['reference_track']),'UNKNOWN');key=(file['route'],row['backend'],row['variant'],row['layer'],row['mode'],row.get('crop_side',0),t['role'],t['scene'],status);g=agg[key];g['candidate_pairs']+=1;g['tokens_A']+=m['valid_tokens_A'];g['tokens_B']+=m['valid_tokens_B'];g['mutual_count']+=m['mutual_count']
                if m['primary_score'] is None:g['point_unavailable_pairs']+=1;assert m['hard_cycle_A'] is None;continue
                g['measured_pairs']+=1;measured+=1
                for k in ['mean_mutual_similarity','mutual_coverage_A','mutual_coverage_B','soft_cycle_A','soft_cycle_B','best_second_margin','NN_similarity_A','NN_similarity_B','one_to_one_count','one_to_one_similarity']:
                    g[k+'_sum']+=m[k]
                if measured%257:continue
                C=np.asarray(row['raw']['cosine'],dtype=np.float64);M,N=C.shape
                def probabilities(z):
                    z=z/.07;z-=z.max(axis=1,keepdims=True);z=np.exp(z);return z/z.sum(axis=1,keepdims=True)
                p,q=probabilities(C),probabilities(C.T);sa=float(np.trace(p@q)/M);sb=float(np.trace(q@p)/N);err=max(abs(sa-m['soft_cycle_A']),abs(sb-m['soft_cycle_B']));assert err<2e-6;max_error=max(max_error,err)
                ia,ib=row['raw']['one_to_one_A'],row['raw']['one_to_one_B'];assert len(set(ia))==len(ia)==min(M,N)==len(set(ib));assert abs(float(C[ia,ib].mean())-m['one_to_one_similarity'])<1e-6;checks+=1
    keys=['route','backend','variant','layer','mode','crop_side','role','scene','candidate_status'];save_new(RUNTIME/'cycle_audit_v1.json',dict(status='SOFT_CYCLE_AND_LOCAL_ASSIGNMENT_AUDITED',source=source,all_valid_multipoint_pairs=measured,independent_stride257_checks=checks,maximum_float64_trace_softcycle_error=max_error,hard_cycle_is_mutual_coverage_identity=True,not_independent_geometry_truth=True,aggregate=[dict(zip(keys,k),**v) for k,v in agg.items()]))
    print('PASS',measured,checks,max_error,flush=True)

if __name__=='__main__':main()
