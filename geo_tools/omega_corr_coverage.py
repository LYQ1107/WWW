"""Official preprocessing and actual object/context patch counts, independent of feature values."""
import csv
import json
import torch
from omega_corr_common import ROOT,RUNTIME,OMEGA_RUNTIME,save_new,sha256
from omega_modules import geometry_module


def main():
    torch.set_num_threads(4)
    cohort_path=ROOT/'manifests/omega_corr_cohort_v1.json';cohort=json.loads(cohort_path.read_text())
    index=json.loads((ROOT/'results/omega_corr_candidate_index_v1.json').read_text())
    assert sha256(index['input']['path'])==index['input']['sha256']
    candidates=json.loads(open(index['input']['path']).read())
    preprocess=geometry_module('omega_adapter').preprocess_pair
    regions=geometry_module('object_patch_regions').object_regions
    by_image={}
    for proposal in candidates['proposals'].values():by_image.setdefault(proposal['image_id'],[]).append(proposal)
    rows=[];mappings={};region_records={}
    for pair in cohort['pairs']:
        images,maps=preprocess(OMEGA_RUNTIME/'official_vggt_omega',[i['path'] for i in pair['images']],512)
        assert tuple(images.shape[0:2])==(2,3)
        mappings[pair['pair_id']]=[m.record() for m in maps]
        for image,mapping in zip(pair['images'],maps):
            for proposal in by_image.get(image['image_id'],[]):
                r=regions(mapping,proposal['bbox_xyxy']);key=':'.join(map(str,proposal['source']))
                region_records[key]={name:dict(patch_ids=v['patch_ids'].tolist(),available_count=v['available_count'],
                    is_object_token=v['is_object_token'].tolist(),network_centers=v['network_centers'].tolist(),
                    original_centers=v['original_centers'].tolist()) for name,v in r['regions'].items()}
                rows.append(dict(pair_id=pair['pair_id'],role=pair['role'],scene=pair['scene'],source=key,
                    original_bbox_area=r['original_bbox_area'],network_bbox_width=float(r['network_bbox_size'][0]),network_bbox_height=float(r['network_bbox_size'][1]),
                    object_patch_count=r['valid_object_patch_count'],context_patch_count=r['valid_context_patch_count'],
                    background_ring_patch_count=r['regions']['M2']['available_count'],bucket=r['patch_bucket'],
                    selected_object_tokens=r['regions']['M0']['selected_count'],selected_context_tokens=r['regions']['M1']['selected_count'],
                    selected_ring_tokens=r['regions']['M2']['selected_count'],point_correspondence_available=r['regions']['M0']['point_correspondence_available']))
        print(pair['pair_id'],flush=True)
    path=ROOT/'results/omega_corr_token_coverage.csv'
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    n=len(rows);few=sum(r['object_patch_count']<2 for r in rows)
    save_new(RUNTIME/'regions_v1.json',dict(mappings=mappings,regions=region_records,cohort_sha256=sha256(cohort_path)))
    save_new(ROOT/'results/omega_corr_coverage_audit_v1.json',dict(status='ACTUAL_PATCH_COVERAGE_MEASURED',observations=n,less_than2_object_tokens=few,
        less_than2_fraction=few/n,required_local_crop_experiment=few/n>=.25,
        local_crop_pair_ids=cohort['local_crop_pair_ids'],buckets={b:sum(r['bucket']==b for r in rows) for b in ['0','1','2-3','4-8','9+']},
        coordinate_convention='continuous bbox edges via A_edges; centers at16*(index+.5)-.5, inverse via A; fullpatch inside real resized image',
        no_feature_values_used=True,no_new_model_calls=True,GT_input=False,
        regions=dict(path=str(RUNTIME/'regions_v1.json'),sha256=sha256(RUNTIME/'regions_v1.json')),
        coverage=dict(path=str(path),sha256=sha256(path))))
    print('coverage',n,'less than2',few,'localcrop required',few/n>=.25,flush=True)

if __name__=='__main__':main()
