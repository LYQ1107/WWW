"""Audit real known training exposure; a filename never certifies generic pretraining."""
import collections
import torch
from jev_phase14_common import *
from gtr.modeling.jev_native_state import fingerprint

def ref(path):return dict(path=str(path),SHA256=sha(path))

def main():
    protect();torch.set_num_threads(1);source=binding();annotations=Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations');main=read(annotations/'train.json');stage1=read(annotations/'train_stage1.json')
    videos={v['id']:v['file_name'] for v in main['videos']};s1=sorted({i['video_id'] for i in stage1['images']});s2=sorted({i['video_id'] for i in main['images']})
    initializers=[];digests=[]
    for name in ['CH_FPN_1x.pth','CH_FPN_1x_key_adapted.pth']:
        path=Path('/data1/liuyeqiang/WWW/models')/name;checkpoint=torch.load(path,map_location='cpu');model=checkpoint['model']
        digests.append(collections.Counter(fingerprint(t) for t in model.values()))
        initializers.append(dict(**ref(path),bytes=path.stat().st_size,model_tensor_count=len(model),parameters_and_buffers=sum(t.numel() for t in model.values()),
            checkpoint_metadata={k:v for k,v in checkpoint.items() if k!='model'},
            association_parameter_keys=[k for k in model if 'asso' in k or 's_t_head' in k],
            verified_generic_training_dataset=None,author_download_provenance='repositoryREADME pretrained-backbone link, but no data manifest or public content checksum for these exact files',
            inferred_CrowdHuman_from_filename_only=True,inference_is_not_verification=True))
    directories={
      'server_single_camera_or_other_domain':['/data/DATASETS/TRACKING/JDE/SportsMOT','/data/DATASETS/TRACKING/JDE/KITTI','/data/DATASETS/TRACKING/JDE/Refer-DanceTrack','/data/DATASETS/TRACKING/JDE/AVIS'],
      'not_qualified_for_human_MCMOT':['/data1/liuyeqiang/3D-ZeF','/data1/liuyeqiang/dataset/Office31','/data1/liuyeqiang/dataset/OfficeHome']}
    protocol=REPORTS/'EXTERNAL_GENERALIZATION_PROTOCOL.json';metadata=OUT/'external_wildtrack_v1/RESULT_METADATA.json';m=read(metadata)
    result=dict(status='COMPLETE_KNOWN_EXPOSURE_AUDIT',binding=source,Stage1_annotation=ref(annotations/'train_stage1.json'),original_GMT_Stage2_annotation=ref(annotations/'train.json'),
        Stage1_images=len(stage1['images']),Stage1_actual_training_videos=[dict(id=v,scene=videos[v]) for v in s1],original_GMT_Stage2_actual_training_videos=s2,
        association_training=TRAIN,development=DEV,heldout=dict(videos=SEALED,status='SEALED'),
        Stage1_exposed_development=all(v in s1 for v in DEV),Stage1_exposed_heldout=all(v in s1 for v in SEALED),
        association_training_development_video_disjoint=not(set(TRAIN)&set(DEV)),
        scene_groups={'path':[12,13],'football':[14,15,16],'wood':[17],'park':[18,19]},
        identity_overlap='VisionTrack instanceID is annotation identity within sequences; numeric equality across independent videos is not proof of same biometric person; no inter-dataset biometric identity certificate',
        initializers=initializers,key_adaptation_tensor_multiset_bitwise_equal=digests[0]==digests[1],
        data_inventory={k:[dict(path=p,exists=Path(p).exists()) for p in paths] for k,paths in directories.items()},
        scheme_A=dict(status='QUALIFIED_FOR_FIXED_UNSEEN_DATASET_TRANSFER',protocol=ref(protocol),metadata=ref(metadata),
            persistent_person_ID=m['GT_metadata'],known_VisionTrack_training_image_paths_contain_WILDTRACK=False,
            acquisition_evidence=[dict(url='https://www.epfl.ch/labs/cvlab/data/data-wildtrack/',claim='fixed7cameraETHZurich public acquisition'),dict(url='https://arxiv.org/html/2407.01007v2',claim='VisionTrack distinct two-moving-UAV acquisition')],
            scope='fixed C1/C2 observed320-sample prefix; all model/config/hyperparameter choices frozen; independent dataset transfer under known training manifests; not official full7cam benchmark',
            absolute_no_pretraining_exposure='UNVERIFIED because exact initializer training manifest is absent'),
        scheme_B=dict(status='QUALIFICATION_FAILED',reason='no verified generic training manifest or public content checksum for available exact initialization weights; clean full-system Stage1/GMT/common-controller retraining has no separately frozen eligible budget and configuration',
            candidates=initializers,training_NOT_RUN=True,no_preexposed_Stage1_relabelled_clean=True),
        independent_absolute_full_system_claim='NO_GO_PENDING_INITIALIZER_PROVENANCE',
        read_official_TEST=False,Full24=False,sealed_predictions_or_scores_read=False)
    save(REPORTS/'PRETRAIN_EXPOSURE_AUDIT.json',result);print('PHASE14_EXPOSURE_AUDIT_COMPLETE',result['key_adaptation_tensor_multiset_bitwise_equal'],flush=True)

if __name__=='__main__':main()
