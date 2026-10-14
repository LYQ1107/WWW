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
    for path in sorted((RUNTIME/'retrieval').glob('*/retrieval.metrics.json')):
        manifest_path = path.parent/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if not manifest['full_video'] or not manifest.get('speed_eligible',True):
            continue
        retrieval_runs.append((manifest, json.loads(path.read_text())))
        sources[str(path)] = sha256(path)
        sources[str(manifest_path)] = sha256(manifest_path)
    if retrieval_runs:
        labels = [f"{manifest['backbone']}, {manifest['camera_streams']} cameras, k{manifest['interval']}" for manifest, _ in retrieval_runs]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, metric in zip(axes, ['Recall@1', 'MRR', 'Complete images/s']):
            values = [(report['recall_at_k']['1'] if metric == 'Recall@1' else report['mrr']) if metric != 'Complete images/s' else manifest['camera_image_fps'] for manifest, report in retrieval_runs]
            axis.bar(np.arange(len(values)), values, color='#427aab')
            axis.set_xticks(np.arange(len(labels)), labels, rotation=20, ha='right', fontsize=8)
            axis.set_title(metric)
            if metric != 'Complete images/s':
                axis.set_ylim(0, 1.05)
        fig.suptitle('Complete causal target-to-clip retrieval: local dataset replay', fontsize=12)
        fig.text(.5,.01,'GT only for evaluation; quality conditional on matched queries with positive gallery clips. No Web deployment or stable timing claim.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.93))
        fig.savefig(out/'causal_retrieval_results.png', dpi=180)
        fig.savefig(out/'causal_retrieval_results.pdf')
        plt.close(fig)
    save_json(out/'manifest.json', dict(status='IN_PROGRESS_FIGURES', source_artifacts=sources,
              outputs={path.name:sha256(path) for path in out.iterdir() if path.suffix in {'.png','.pdf'}},
              provenance=provenance(), limitations='incomplete research matrix; no replicate confidence intervals or accuracy/runtime Pareto superiority inferred'))


if __name__ == '__main__':
    main()
