"""CPU read-only hashes of approved images; no model or feature extraction."""
from gmt_jev_verifier_common import *
def main():
    records=read(ANNOTATIONS)['images'];cases=[]
    annotation=ref(ANNOTATIONS)
    for video in [12,13,14]:
        images=sorted((r for r in records if r['video_id']==video),key=lambda r:r['id'])
        h=hashlib.sha256()
        for image in images:h.update(f"{image['id']}:{sha(IMAGES/image['file_name'])}\n".encode())
        reference=legacy.read(legacy.REPORTS/f'TRAIN_MATCHED_original_VIDEO{video:02d}.json')['inference']
        assert h.hexdigest()==reference['input_images_SHA256']
        assert annotation['SHA256']==reference['annotation']['SHA256']
        cases.append(dict(video=video,images=len(images),current_input_images_SHA256=h.hexdigest(),reference=ref(legacy.REPORTS/f'TRAIN_MATCHED_original_VIDEO{video:02d}.json')))
        print('V2_INPUT_BYTES_PASS',video,len(images),flush=True)
    freeze=read(REPORTS/'BASELINE_FREEZE.json')
    assert sha(STAGE2)==freeze['GMT_checkpoint']['SHA256']
    assert sha(freeze['MLP_checkpoint']['path'])==freeze['MLP_checkpoint']['SHA256']
    config=ROOT/'configs/VISION_test.yaml'
    assert sha(config)==legacy.read(legacy.REPORTS/'BASELINE_PARITY.json')['source']['source_SHA256']['configs/VISION_test.yaml']
    save(REPORTS/'INPUT_PROVENANCE.json',dict(status='PASS_FRESH_APPROVED_IMAGE_ANNOTATION_CONFIG_CHECKPOINT_HASHES',
        annotation=annotation,cases=cases,config=ref(config),GMT_checkpoint=freeze['GMT_checkpoint'],MLP_checkpoint=freeze['MLP_checkpoint'],
        new_GPU_hours=0,files_copied=0,perception_features_extracted=0,other_video_images_opened=False,
        scope='Shared TRAIN annotation file is hashed/read; image bytes are opened only for approved TRAIN12/13/14. No TEST, sealed or new DEV image access.',source=binding()))
if __name__=='__main__':main()
