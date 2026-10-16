"""Read-only final artifact and independent-row consistency verification."""
import hashlib,json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1];r=root/'results';m=json.loads((r/'omega_corr_experiment_manifest.json').read_text())
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    for name,entry in m['artifacts'].items():assert (root/name).stat().st_size==entry['bytes'] and sha(root/name)==entry['sha256'],name
    for name,digest in m['documents'].items():assert sha(root/name)==digest,name
    p=json.loads((r/'omega_corr_independent_published_rows_v1.json').read_text());c=json.loads((r/'omega_corr_identity_corrections.json').read_text());configs=json.loads((r/'omega_corr_configuration_index.json').read_text());layer=m['selected_layer_fit_only'];name=f'FULL_IMAGE_MULTIVIEW|omega|F2|{layer}|M0|primary|0';cid=next(i for i,x in enumerate(configs) if x['config_id']==name)
    row=next(x for x in p['rows'] if x['layer']==layer and x['role']=='MECHANISM_CHECK');all_changes=c['changes'][str(cid)]
    for field in ['known_row_corrections','known_row_harms']:assert set(row[field])<=set(all_changes[field])
    cases=c['selected_mechanism_all_changed_rows'];assert set(row['known_row_corrections'])=={int(x['query_index']) for x in cases if x['known_correction']=='True'};assert set(row['known_row_harms'])=={int(x['query_index']) for x in cases if x['known_harm']=='True'}
    solver=json.loads((r/'omega_corr_solver_decisions.json').read_text());assert solver['status']=='NOT_EXECUTED' and solver['actual_final_ID_changes'] is None
    assert m['UNIQUE_OMEGA_EVIDENCE']=='NO_UNIQUE_OMEGA_CORRESPONDENCE_EVIDENCE';assert m['GPU_visible_child_hours']<=2 and m['cache_bytes']<2*1024**3
    print('PASS',len(m['artifacts']),'artifact hashes;',len(m['documents']),'report hashes; independent selected mechanism repair/harm cases',len(row['known_row_corrections']),len(row['known_row_harms']))

if __name__=='__main__':main()
