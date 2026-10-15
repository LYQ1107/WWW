"""Static scientific figures from immutable metrics; no invented error bars."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from common import *


def main():
    out = ROOT/'results/plots'
    out.mkdir(exist_ok=True)
    sources = {}
    baseline_path = ROOT/'results/B0_strict_suite_v1_metrics.json'
    baseline = json.loads(baseline_path.read_text())['metrics']
    sources[str(baseline_path)] = sha256(baseline_path)
    methods = [
        ('DINO ROI, k10, scalar .25', 'B1_dinov2_strict_suite_v1_metrics.json'),
        ('DINO ROI, k1, scalar .5', 'B1_dinov2_dense_strict_suite_v1_metrics.json'),
        ('VGGT l17, k10, scalar .5', 'B2_native_absolute_strict_suite_v1_metrics.json'),
        ('VGGT l17, k1, scalar .5', 'B2_native_absolute_dense_strict_suite_v1_metrics.json'),
        ('VGGT 4 layers, k10, scalar .5', 'B3_multilayer_strict_suite_v1_metrics.json'),
        ('VGGT l17, k10, scalar 2', 'B4_native_scalar_w2_k10_strict_v1_suite_metrics.json'),
        ('GMT learned, k10, weight 2', 'C_lowrank_gmt_seed14_k10_strict_v1_suite_metrics.json'),
        ('DINO learned, k10, weight 2', 'C_lowrank_dinov2_seed14_k10_strict_v1_suite_metrics.json'),
        ('VGGT learned, k10, weight 2', 'B5_lowrank_vggt_seed14_k10_strict_v1_suite_metrics.json'),
        ('VGGT camera only, k10, .5', 'B6_vggt_camera_strict_suite_v1_metrics.json'),
        ('VGGT learned + camera, k10', 'B7_lowrank_vggt_camera_seed14_k10_strict_v1_suite_metrics.json'),
    ]
    labels, values = [], []
    for label, name in methods:
        path = ROOT/'results'/name
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if not report['full_video'] or report['causal_online'] or report['video_ids'] != [1,10,11,18,19]:
            raise ValueError('offline strict suite required')
        labels.append(label)
        values.append(report['metrics'])
        sources[str(path)] = sha256(path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
    for axis, metric in zip(axes, ['HOTA', 'AssA', 'CVIDF1']):
        delta = [row[metric]-baseline[metric] for row in values]
        axis.barh(np.arange(len(labels)), delta, color=['#427aab' if d >= 0 else '#a65d50' for d in delta])
        axis.axvline(0, color='black', linewidth=.8)
        axis.set_title(metric)
        axis.set_xlabel('Change from GMT (percentage points)')
        axis.grid(axis='x', alpha=.2)
        axis.xaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.suptitle('Complete strict development: native TrackEval combination, offline protocol', fontsize=12)
    fig.text(.5, .012, 'Fixed TRAIN settings; single primary head seed; no repeatability/error-bar claim. Small positive control changes do not establish VGGT benefit.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.96))
    fig.savefig(out/'strict_identity_deltas.png', dpi=180)
    fig.savefig(out/'strict_identity_deltas.pdf')
    plt.close(fig)
    attribution_path=ROOT/'results/causal_dense_scene1_backbone_controls_partial_v1.json'
    if attribution_path.exists():
        attribution=json.loads(attribution_path.read_text())
        if attribution['status']=='COMPLETE_SCENE1_ATTRIBUTION_NOT_FULL_DEV':
            source_rows=attribution['rows']
            by_tag={row['tag']:row for row in source_rows}
            baseline_scene=by_tag['online_B0_full_v1_source44_v1']['metrics']
            specs=[('GMT head','C_gmt_head_dense_causal_v1_v1'),
                   ('DINO head','C_dino_head_dense_causal_v1_v1'),
                   ('VGGT head','B5_vggt_head_dense_causal_v1_v1'),
                   ('VGGT camera','B6_vggt_camera_dense_causal_v1_v1'),
                   ('VGGT head + camera','causal_B7_fixed_k1_strict_suite_v1_v1')]
            for label,name in [('Native-head constant score','constant_native_head_scene1_accuracy_v1.json'),('Head-free fixed prior','native_prior_scene1_accuracy_v1.json')]:
                path=ROOT/'results'/name
                if path.exists():
                    evidence=json.loads(path.read_text());row=evidence['rows'][0];tag=Path(row['run']).name
                    by_tag[tag]=dict(tag=tag,metrics=row['metrics'],camera_image_fps=row['camera_image_fps'],speed_eligible=row.get('speed_eligible',True))
                    specs.append((label,tag));sources[str(path)]=sha256(path)
            fig,axes=plt.subplots(1,2,figsize=(9,4.5))
            for axis,metric in zip(axes,['CVIDF1','AssA']):
                delta=[by_tag[tag]['metrics'][metric]-baseline_scene[metric] for _,tag in specs]
                axis.barh(range(len(specs)),delta,color=['#687e93','#407c8e','#7d6093','#b18758','#7d6093','#4a785d','#4a785d'][:len(specs)])
                axis.set_yticks(range(len(specs)),[label for label,_ in specs]);axis.invert_yaxis()
                axis.axvline(0,color='black',linewidth=.8)
                axis.set_xlabel('Change (percentage points)')
                axis.set_title(metric);axis.grid(axis='x',alpha=.2)
            fig.suptitle('Scene1 complete causal feature and calibration controls',fontsize=11)
            fig.text(.5,.015,'One full scene / inference seed. Native fixed prior exceeds VGGT here; no independent geometry output effect. Full-dev attribution/stability incomplete.',ha='center',fontsize=7.5)
            fig.tight_layout(rect=(0,.06,1,.94))
            fig.savefig(out/'causal_scene1_attribution.png',dpi=180)
            fig.savefig(out/'causal_scene1_attribution.pdf')
            plt.close(fig)
            sources[str(attribution_path)]=sha256(attribution_path)
            fig,axis=plt.subplots(figsize=(8,4.5))
            runtime_specs=[('GMT baseline','online_B0_full_v1_source44_v1'),*specs]
            for label,tag in runtime_specs:
                row=by_tag[tag]
                if row.get('speed_eligible',True):
                    axis.scatter(row['camera_image_fps'],row['metrics']['CVIDF1'],s=40)
                    offset=(4,22) if label=='VGGT head + camera' else (4,6)
                    axis.annotate(label,(row['camera_image_fps'],row['metrics']['CVIDF1']),xytext=offset,textcoords='offset points',fontsize=8)
            axis.set_xlabel('Complete camera images/s');axis.set_ylabel('CVIDF1 (%)');axis.grid(alpha=.2)
            axis.margins(x=.18,y=.2)
            fig.suptitle('Scene1 causal identity and runtime: descriptive measurements',fontsize=11)
            fig.text(.5,.015,'Single V100 runs; no repeated timing or Pareto claim. Static GMT setup excluded; extra encoder cold cost charged.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.06,1,.94))
            fig.savefig(out/'causal_scene1_identity_runtime.png',dpi=180);fig.savefig(out/'causal_scene1_identity_runtime.pdf')
            plt.close(fig)
    audit_path = ROOT/'results/causal_fixed_joint_random_total80_full_v18_v1.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        rows = audit['rows']
        labels = ['Fixed20', 'Joint', 'Random']
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        axes[0].bar(labels, [row['camera_image_fps'] for row in rows], color='#427aab')
        axes[0].set_ylabel('Camera images/s (complete causal pipeline)')
        axes[1].bar(labels, [1000*row['input_to_identity_p95'] for row in rows], color='#a65d50')
        axes[1].set_ylabel('Input-to-identity P95 (ms)')
        fig.suptitle('Scene18: 80 actual VGGT calls each; predictions exactly equal to causal GMT OFF', fontsize=10)
        fig.text(.5,.015,'Actual V100 hardware; single descriptive runs, no repeated speed or policy superiority claim.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.92))
        fig.savefig(out/'causal_total80_runtime.png', dpi=180)
        fig.savefig(out/'causal_total80_runtime.pdf')
        plt.close(fig)
        sources[str(audit_path)] = sha256(audit_path)
    retrieval_runs = []
    scope_path=ROOT/'results/www_scope_evidence_v1.json'
    cohort_rows=json.loads(scope_path.read_text())['production_fixed_scene18_rows'] if scope_path.exists() else []
    if cohort_rows:sources[str(scope_path)]=sha256(scope_path)
    for cohort in cohort_rows:
        path=Path(cohort['metrics'])
        manifest_path = path.parent/'manifest.json'
        if sha256(path)!=cohort['metrics_sha256'] or sha256(manifest_path)!=cohort['manifest_sha256']:
            raise ValueError('changed fixed-query retrieval artifact')
        if not cohort['exact_detector_query_cohort'] or not cohort['exact_full_scene18_detector_outputs']:
            raise ValueError('fixed detector cohort required')
        manifest = json.loads(manifest_path.read_text())
        if not manifest['full_video'] or not manifest.get('speed_eligible',True):
            continue
        retrieval_runs.append((manifest, json.loads(path.read_text())))
        sources[str(path)] = sha256(path)
        sources[str(manifest_path)] = sha256(manifest_path)
    if retrieval_runs:
        tracking_labels={'gmt_off':'GMT','native_fixed_prior':'Native prior','vggt_dense':'Dense VGGT'}
        labels = [f"{tracking_labels.get(manifest.get('tracking_method','gmt_off'),manifest.get('tracking_method'))}; rank {'native ReID' if manifest['backbone']=='none' else manifest['backbone']}; {manifest['camera_streams']} cameras"+
                  (f"; k{manifest['interval']}" if manifest['backbone']!='none' else '') for manifest, _ in retrieval_runs]
        fig, axes = plt.subplots(1, 3, figsize=(15, max(4.5,.4*len(labels))),sharey=True)
        for axis, metric in zip(axes, ['Recall@1', 'MRR', 'Complete images/s']):
            values = [(report['recall_at_k']['1'] if metric == 'Recall@1' else report['mrr']) if metric != 'Complete images/s' else manifest['camera_image_fps'] for manifest, report in retrieval_runs]
            axis.barh(np.arange(len(values)), values, color='#427aab')
            axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
            axis.set_title(metric)
            if metric != 'Complete images/s':
                axis.set_xlim(0, 1.05)
        axes[0].invert_yaxis()
        fig.suptitle('Complete retrieval: fixed scene18 detector queries across galleries', fontsize=12)
        fig.text(.5,.01,'Quality conditional on each run\'s eligible gallery; throughput charges the whole producer. Local replay; no stable timing or Web deployment claim.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.93))
        fig.savefig(out/'causal_retrieval_results.png', dpi=180)
        fig.savefig(out/'causal_retrieval_results.pdf')
        plt.close(fig)
    budget_path=ROOT/'results/native_prior_vs_vggt_full_dev_budget_v1.json'
    if budget_path.exists():
        budget=json.loads(budget_path.read_text());sources[str(budget_path)]=sha256(budget_path)
        labels=['Head-free fixed prior','Dense VGGT','Sparse VGGT, k20/cap80']
        fig,axes=plt.subplots(1,2,figsize=(10,4.5))
        for method,label,color in zip(budget['methods'],labels,['#448b78','#986799','#b98549']):
            name=Path(method['source']).stem.removesuffix('_sources')
            metric_path=ROOT/'results'/f'{name}_metrics.json';metric=json.loads(metric_path.read_text())
            if metric['video_ids']!=[1,10,11,18,19] or not metric['causal_online'] or metric['images']!=method['total_images']:
                raise ValueError('matched complete strict5 causal figure required')
            sources[str(metric_path)]=sha256(metric_path)
            for axis,x in zip(axes,[method['cumulative_camera_image_fps'],method['total_actual_vggt_calls']]):
                axis.scatter(x,metric['metrics']['CVIDF1'],s=55,color=color,label=label)
                axis.grid(alpha=.2);axis.set_ylabel('CVIDF1 (%)')
        axes[0].set_xlabel('Complete camera images/s');axes[1].set_xlabel('Actual VGGT calls (all five scenes)')
        axes[0].legend(loc='lower right',fontsize=8)
        fig.suptitle('Complete strict development: identity and compute',fontsize=12)
        fig.text(.5,.015,'Single V100 measurements; repeated stability/attribution incomplete; no Pareto or real-time claim.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.055,1,.94))
        fig.savefig(out/'causal_full_dev_identity_compute.png',dpi=180)
        fig.savefig(out/'causal_full_dev_identity_compute.pdf');plt.close(fig)
    interval_path=ROOT/'results/causal_complete_interval_identity_budget_v1.json'
    if interval_path.exists():
        measured=json.loads(interval_path.read_text());sources[str(interval_path)]=sha256(interval_path)
        base_path=ROOT/'results/causal_B0_strict_suite_v1_metrics.json';base=json.loads(base_path.read_text())['metrics']['CVIDF1']
        prior_path=ROOT/'results/native_prior_seed20261014_strict_suite_v1_metrics.json'
        prior=json.loads(prior_path.read_text())['metrics']['CVIDF1'];sources[str(base_path)]=sha256(base_path);sources[str(prior_path)]=sha256(prior_path)
        fixed=[];identity=None
        for method in measured['methods']:
            name=Path(method['source']).stem.removesuffix('_sources');path=ROOT/'results'/f'{name}_metrics.json'
            result=json.loads(path.read_text())
            if result['video_ids']!=[1,10,11,18,19] or not result['causal_online'] or result['images']!=method['total_images']:
                raise ValueError('complete matched interval curve required')
            sources[str(path)]=sha256(path)
            point=dict(calls=method['total_actual_vggt_calls'],fps=method['cumulative_camera_image_fps'],delta=result['metrics']['CVIDF1']-base)
            if '_fixed_k' in name:point['label']='k'+name.split('_fixed_k')[1].split('_')[0];fixed.append(point)
            else:point['label']='identity only';identity=point
        fig,axes=plt.subplots(1,2,figsize=(11,4.7))
        for axis,key,title in zip(axes,['calls','fps'],['Actual VGGT calls (all five scenes)','Complete camera images/s']):
            ordered=sorted(fixed,key=lambda row:row[key]);axis.plot([row[key] for row in ordered],[row['delta'] for row in ordered],'o-',color='#6e5d8f',label='Fixed period')
            for row in ordered:axis.annotate(row['label'],(row[key],row['delta']),xytext=(4,5),textcoords='offset points',fontsize=8)
            if identity:
                axis.scatter(identity[key],identity['delta'],marker='^',s=60,color='#c48045',label='Identity only')
            axis.axhline(0,color='#666666',linestyle=':',label='Causal GMT reference')
            axis.axhline(prior-base,color='#448b78',linestyle='--',label='Native prior accuracy reference')
            axis.set_xlabel(title);axis.set_ylabel('CVIDF1 change (percentage points)');axis.grid(alpha=.2)
        axes[0].set_xscale('log');axes[0].legend(fontsize=7,loc='upper left')
        fig.suptitle('Complete causal development: keyframe budget and identity quality',fontsize=12)
        fig.text(.5,.015,'Primary head seed; single V100 timings. Equal CVIDF1 does not imply identical full predictions or stable Pareto superiority.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.055,1,.94));fig.savefig(out/'causal_complete_interval_curve.png',dpi=180);fig.savefig(out/'causal_complete_interval_curve.pdf');plt.close(fig)
    attribution_path=ROOT/'results/causal_dense_geometry_full_strict_attribution_v1.json'
    if attribution_path.exists():
        attribution=json.loads(attribution_path.read_text())
        if attribution['status']!='EXACT_FEATURE_JOINT_FULL_STRICT_OUTPUTS' or not attribution['full_strict_suite']:
            raise ValueError('full feature/joint attribution required for current figure')
        sources[str(attribution_path)]=sha256(attribution_path)
        base_path=ROOT/'results/causal_B0_strict_suite_v1_metrics.json';base=json.loads(base_path.read_text())['metrics'];sources[str(base_path)]=sha256(base_path)
        cases=[('GMT learned head','C_gmt_head_dense_causal_strict_suite_v1'),('DINO learned head','C_dino_head_dense_causal_strict_suite_v1'),
            ('VGGT feature only','causal_B5_dense_seed20261014_strict_suite_v1'),('VGGT + geometry','causal_B7_fixed_k1_strict_suite_v1'),
            ('Head-free fixed prior','native_prior_seed20261014_strict_suite_v1')]
        labels=[];values=[]
        for label,name in cases:
            path=ROOT/'results'/f'{name}_metrics.json';result=json.loads(path.read_text())
            if result['status']!='PASS' or result['video_ids']!=[1,10,11,18,19] or not result['causal_online'] or not result['full_video']:
                raise ValueError('complete matched causal attribution suite required')
            sources[str(path)]=sha256(path);labels.append(label);values.append(result['metrics'])
        colors=['#777777','#bd8a45','#806399','#806399','#448b78']
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),sharey=True)
        for axis,delta,title in [(axes[0],[value['CVIDF1']-base['CVIDF1'] for value in values],'CVIDF1 change (percentage points)'),
                                  (axes[1],[base['IDSW']-value['IDSW'] for value in values],'ID switches avoided')]:
            axis.barh(np.arange(len(labels)),delta,color=colors);axis.axvline(0,color='black',linewidth=.8);axis.grid(axis='x',alpha=.2);axis.set_xlabel(title)
            axis.margins(x=.2)
            for index,value in enumerate(delta):axis.annotate(f'{value:+.6f}' if axis is axes[0] else str(value),(value,index),xytext=(4,0),textcoords='offset points',va='center',fontsize=8)
        axes[0].set_yticks(np.arange(len(labels)),labels);axes[0].invert_yaxis()
        fig.suptitle('Complete causal strict5: attribution against frozen GMT',fontsize=12)
        fig.text(.5,.012,'Primary TRAIN-head seed; inference seed20261014. Feature/joint raw outputs are exact; no stable architecture or speed claim.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(out/'causal_full_dev_attribution.png',dpi=180);fig.savefig(out/'causal_full_dev_attribution.pdf');plt.close(fig)
    save_json(out/'manifest.json', dict(status='IN_PROGRESS_FIGURES', source_artifacts=sources,
              outputs={path.name:sha256(path) for path in out.iterdir() if path.suffix in {'.png','.pdf'}},
              provenance=provenance(), limitations='incomplete research matrix; no replicate confidence intervals or accuracy/runtime Pareto superiority inferred'))


if __name__ == '__main__':
    main()
