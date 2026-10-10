"""Read-only input identity checks; do not rewrite historical manifests."""
from jev_phase16_common import *
from jev_phase15_common import perception_provenance


def main():
    protect();storage_guard()
    old=read(REPORTS/'FROZEN_PRIOR_EVIDENCE.json')
    cache=[perception_provenance(v) for v in TRAIN+DEV]
    checkpoints=[]
    for path,digest in old['phase15_model_checkpoint_SHA256'].items():
        actual=ref(path);assert actual['SHA256']==digest,(path,'frozen checkpoint changed')
        checkpoints.append(actual)
    legacy=read(old['previous_full_332_checkpoint_hash_audit']['path'])
    descriptors=legacy['binding']['checkpoints']
    for item in descriptors:
        assert sha(item['path'])==item['SHA256'],('older checkpoint changed',item['path'])
    save(REPORTS/'INPUT_INTEGRITY.json',dict(status='PASS',binding=binding(
        checkpoints=checkpoints,inputs=[old['previous_full_332_checkpoint_hash_audit']],
        evaluator='fresh read-only byte hashes of all prior frozen weights and perception indices',
        scope='supplementary input check after P0 launch, not an invented initial recording'),
        checked_cache_indices=cache,older_checkpoint_count=len(descriptors),
        older_weight_descriptors=old['previous_full_332_checkpoint_hash_audit'],
        all_old_frozen_weights_bytes_unchanged=True,
        note='P0 cache reader independently verifies each actual loaded payload checksum. '
             'A cache index hash does not establish real image latency. Only TRAIN and reused DEV indices are read.',
        sealed_and_TEST_read=False))
    print('PHASE16_INPUT_INTEGRITY_PASS',len(descriptors),len(checkpoints),len(cache),flush=True)


if __name__=='__main__':main()
