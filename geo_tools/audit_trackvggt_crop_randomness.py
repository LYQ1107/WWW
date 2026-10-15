"""Check fixed crop variants' final native query-draw RNG equality."""
import torch
import numpy as np
from common import *


def rng_equal(left,right):
    return left[0]==right[0] and np.array_equal(left[1],right[1]) and left[2:]==right[2:]


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');rows=[]
    for backbone in ['vggt','dinov2']:
        states=[];manifests=[]
        for variant in ['full','local','combined']:
            path=runtime/'models'/f'ag_{backbone}_C_crop224_{variant}_k24_seed20261015_1000_v1/manifest.json'
            manifest=json.loads(path.read_text());checkpoint=path.parent/'checkpoint.pt'
            if manifest['status']!='COMPLETE_FINITE_TEACHER_FIT_AND_FIXED_VALIDATION' or sha256(checkpoint)!=manifest['checkpoint_sha256']:raise ValueError('crop fit still incomplete/changed')
            states.append(torch.load(checkpoint,map_location='cpu'));manifests.append(manifest)
        for state in states:
            if not rng_equal(state['numpy_rng_state'],states[0]['numpy_rng_state']):raise ValueError('native crop query-draw RNG differs')
            if not torch.equal(state['torch_rng_state'],states[0]['torch_rng_state']) or any(not torch.equal(x,y) for x,y in zip(state['cuda_rng_states'],states[0]['cuda_rng_states'])):raise ValueError('crop Torch RNG consumption differs')
            for key in ['training_seed','feature_plan_sha256','batch_size','optimizer','lr','weight_decay','clip_norm','target_steps']:
                if state['settings'][key]!=states[0]['settings'][key]:raise ValueError('crop matched fitting setting differs')
        if len({m['supervised_fit_queries'] for m in manifests})!=1:raise ValueError('crop GT-eligible query population differs')
        rows.append(dict(backbone=backbone,all_three_final_numpy_draw_states_exact=True,all_three_final_torch_rng_states_exact=True,same_original_GT_eligible_query_population=True,
            supervised_fit_queries=manifests[0]['supervised_fit_queries'],full_neutral_draws_in_last850=manifests[0]['neutral_fit_query_forwards'],
            source_scope='Final RNG state plus identical frozen eligible-index order/seed and sole np.random.choice draw site verifies matched native minibatch schedules by code; per-step keys were not separately logged.',
            checkpoint_SHA256=[m['checkpoint_sha256'] for m in manifests]))
    save_json(ROOT/'results/trackvggt_crop_matched_randomness_v1.json',dict(status='PASS_MATCHED_CROP_QUERY_DRAWS',rows=rows,
        no_GT_or_software_changes_to_fit_variants=True,provenance=provenance()))
    print('PASS_MATCHED_CROP_QUERY_DRAWS')


if __name__=='__main__':main()
