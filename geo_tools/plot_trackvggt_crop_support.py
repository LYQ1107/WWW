"""Actual token-coordinate diagnostic; no image copy or GT-based example choice."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import *


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');plan=json.loads((ROOT/'manifests/trackvggt_crop_comparison_v1.json').read_text())
    cohort=json.loads(Path(plan['base_cohort']).read_text());val={tuple(k) for k in plan['validation_keys']}
    full={}
    for name in cohort['feature_manifest_paths']['vggt']:
        m=json.loads(Path(name).read_text())
        for row in m['rows']:
            if tuple(row['key']) in val:full[tuple(row['key'])]=row
    key=next(key for key in sorted(val) if full[key]['raw_patch_centers']==0)
    crop_manifest=Path(plan['feature_manifests']['vggt']);crop=json.loads(crop_manifest.read_text())
    local=next(row for row in crop['rows'] if tuple(row['key'])==key);native=full[key]
    for row in [native,local]:
        if sha256(row['file'])!=row['sha256']:raise ValueError('actual example token file changed')
    arrays=[]
    for row in [native,local]:
        with np.load(row['file']) as a:arrays.append({k:a[k].copy() for k in ('xy','inside')})
    box=np.asarray(local['original_box']);x1,y1,x2,y2=local['crop_rect'];fig,axes=plt.subplots(1,2,figsize=(8,4),constrained_layout=True)
    for ax,row,a,title in zip(axes,[native,local],arrays,['Full image, pad518','Local magnification, pad224']):
        inside=a['inside'];xy=a['xy'];ax.scatter(xy[~inside,0],xy[~inside,1],s=22,c='#8c96a0',label='Context center')
        ax.scatter(xy[inside,0],xy[inside,1],s=24,c='#e17736',label='Box center')
        ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,color='#167b77',lw=1.5,label='Native detection'))
        ax.set_xlim(x1,x2);ax.set_ylim(y2,y1);ax.set_aspect('equal');ax.set_xlabel('Original image x (pixels)');ax.set_ylabel('Original image y (pixels)')
        ax.set_title(f"{title}\n{row['raw_patch_centers']} valid box centers; {len(xy)} retained tokens",fontsize=10)
        ax.legend(loc='lower left',fontsize=7)
    fig.suptitle(f'Actual validation query {list(key)}; one physical observation, no GT selection',fontsize=10)
    directory=ROOT/'results/plots';directory.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(directory/f'trackvggt_crop_support_v1.{suffix}',dpi=160)
    plt.close(fig)
    save_json(ROOT/'results/trackvggt_crop_support_figure_v1.json',dict(status='ACTUAL_COORDINATE_DIAGNOSTIC',key=list(key),
        example_selection='First original validation key with observed full-grid zero centers, independent of GT/teacher output. Does not select or remove any evaluation query.',
        full_native_patch_centers=native['raw_patch_centers'],crop_native_patch_centers=local['raw_patch_centers'],
        full_token_file_sha256=native['sha256'],crop_token_file_sha256=local['sha256'],crop_manifest_sha256=sha256(crop_manifest),
        raw_images_copied=False,GT_read=False,meaning='Token support/coordinates only, not identity correctness or teacher benefit.',
        artifacts={p.name:sha256(p) for p in directory.glob('trackvggt_crop_support_v1.*')}))
    print(json.dumps(dict(key=list(key),full_centers=native['raw_patch_centers'],crop_centers=local['raw_patch_centers'])))


if __name__=='__main__':main()
