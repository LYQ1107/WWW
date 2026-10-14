"""Fit scalar cosine log odds only on frozen association-TRAIN pairs."""
import argparse,numpy as np
from common import *
from sklearn.linear_model import LogisticRegression

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--layer',type=int,default=23);p.add_argument('--pooling',default='roi_align');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.samples.read_text())
    if data['provenance'].get('split')!='train':raise ValueError('calibration must use association TRAIN')
    split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text());x=[];y=[]
    for sample in data['samples']:
        if sample['video_id'] not in split['association_train_video_ids']:raise ValueError('development leakage')
        arr=np.load(sample['path']);feat=np.load(a.features/Path(sample['path']).name)
        v0=feat[f'layer_{a.layer}/{a.pooling}/0'].astype(float);v1=feat[f'layer_{a.layer}/{a.pooling}/1'].astype(float)
        score=v1@v0.T;labels=(arr['labels_1'][:,None]==arr['labels_0'][None,:]);valid=(arr['labels_1'][:,None]>0)&(arr['labels_0'][None,:]>0)
        x.extend(score[valid]);y.extend(labels[valid])
    model=LogisticRegression(C=1.,class_weight='balanced',max_iter=200,random_state=20261014).fit(np.asarray(x)[:,None],y)
    report=dict(source_split='association_train',slope=float(model.coef_[0,0]),intercept=float(model.intercept_[0]),layer=a.layer,pooling=a.pooling,dim=256,
        sample_manifest_sha256=sha256(a.samples),source_video_ids=sorted({s['video_id'] for s in data['samples']}),pairs=len(y),positive_pairs=int(np.sum(y)),seed=20261014,weights='equal positive/negative classes, C=1 fixed',feature_artifact_sha256={str(a.features/Path(s['path']).name):sha256(a.features/Path(s['path']).name) for s in data['samples']})
    save_json(a.output,report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
