#!/usr/bin/env python3
"""Plot actual fit/VAL/final losses and legal TRAIN VAL native scores."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_worldobject_v3 import read,sha,write_json

METHODS={'FULL_WORLD_OBJECT_V3':('Full + Omega','#1f77b4'),
    'FULL_NO_GEOMETRY_V3':('Full no-geometry','#ff7f0e'),
    'HEAD_ONLY_NO_GEOMETRY_V3':('Head geometry off','#2ca02c')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=True,parents=True);fig,axes=plt.subplots(2,3,figsize=(15,9));proofs=[];raw={}
    for method,(name,color) in METHODS.items():
        path=a.root/'train'/method/'train_gate.json';gate=read(path);history=gate['selection_history'];epochs=[x['epoch'] for x in history]
        for col,key in enumerate(('identity_assignment','joint_birth')):
            axes[0,col].plot(epochs,[r['train'][key] for r in history],'o-',color=color,label=name+' fit18')
            axes[0,col].plot(epochs,[r['validation'][key] for r in history],'s--',color=color,label=name+' VAL6')
        axes[0,2].plot([r['epoch'] for r in gate['final_history']],
            [r['train']['total'] for r in gate['final_history']],'o-',color=color,label=name)
        raw[method]={'selection_history':history,'final_history':gate['final_history'],'final_checkpoint_sha256':gate['checkpoint_sha256']}
        proofs.append({'file':str(path),'sha256':sha(path),'checkpoint_sha256':gate['checkpoint_sha256']})
        if method=='FULL_NO_GEOMETRY_V3':continue
        points=[]
        for epoch in range(1,5):
            path=a.root/'diagnostics/retained_TRAIN_VAL'/method/f'epoch_{epoch:02d}/TRAIN_VAL_metrics.json';row=read(path)
            if row['legal_scored_scenes']!=[4,8,12] or not row['replay_exact_original_diagnostics']:raise ValueError('LEGAL_FULL_TRAIN_VAL_SCOPE_AND_EXACT_REPLAY_REQUIRED')
            points.append(row);proofs.append({'file':str(path),'sha256':sha(path),'checkpoint_sha256':row['checkpoint_sha256']})
        raw[method]['legal3_TRAIN_VAL_native_metrics']=points
        for col,key in enumerate(('HOTA','AssA','IDF1')):
            axes[1,col].plot([r['selection_epoch'] for r in points],[r['metrics'][key] for r in points],'o-',label=name,color=color)
    for axis,title in zip(axes.flat,('Assignment loss (fit18 / VAL6)','Joint-birth loss (fit18 / VAL6)',
        'Fresh all24 final fit: total loss','Native TRAIN VAL HOTA / legal3','Native TRAIN VAL AssA / legal3','Native TRAIN VAL IDF1 / legal3')):
        axis.set_title(title,fontsize=11);axis.grid(alpha=.25);axis.legend(fontsize=7)
        axis.set_xlabel('Completed epoch');axis.set_xticks([1,2,3,4])
    axes[0,2].set_xlabel('Registered final epoch (Full1 / NoGeo2 / Head3)')
    for axis in axes[1]:axis.set_ylim(0,100);axis.set_ylabel('Score (%)')
    fig.suptitle('Completed registered training: losses and retained-checkpoint TRAIN validation')
    fig.text(.025,.025,'TRAIN only. Native VAL scores use complete original-GT-legal scenes 4/8/12; 16/20/24 stay excluded. NoGeometry selection weights were overwritten, so its per-epoch native curve is unavailable.',fontsize=8)
    fig.text(.025,.008,'No TEST or budget reselection. Low TRAIN loss and this three-scene VAL subset do not establish generalization. Final fit uses all24 scenes after fixed-seed reinitialization.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.95));image=a.output/'fit_VAL_final_and_native_TRAIN_VAL.png';fig.savefig(image,dpi=140);plt.close(fig)
    write_json(a.output/'fit_VAL_final_and_native_TRAIN_VAL_manifest.json',{'image_sha256':sha(image),
        'source_artifacts':proofs,'raw_curve_data':raw,'TEST_read':False,'budget_reselected':False,
        'native_TRAIN_VAL_scenes':[4,8,12],'original_GT_modified':False,'analysis_sha256':sha(__file__)})
    print(json.dumps({'image':str(image),'bytes':image.stat().st_size}))


if __name__=='__main__':main()
