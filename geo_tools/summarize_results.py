"""Generate status and tables directly from metric artifacts, never enter numbers by hand."""
import csv,json
from common import *

def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def main():
    metric_rows=[];runtime_rows=[];experiments=[]
    for path in sorted((RUNTIME/'baseline').glob('*/*.json')):
        if path.name.endswith('_manifest.json') and 'aborted' not in path.name:
            value=json.loads(path.read_text());runtime_rows.append(dict(method=path.parent.name,video_id=value['video_id'],full_video=value['full_video'],causal_online=value['causal_online'],speed_eligible=value.get('speed_eligible',True),
                camera_image_fps=value['camera_image_fps'],synchronized_group_fps=value['synchronized_group_fps'],total_seconds=value['total_seconds'],inference_seconds=value['inference_seconds'],read_preprocess_seconds=value['read_preprocess_seconds'],peak_gpu_bytes=value['peak_gpu_bytes'],
                geo_calls=value.get('geo_calls',0),geo_seconds=value.get('geo_seconds',0),detection_p50=value['detection_latency_p50'],detection_p95=value['detection_latency_p95'],end_to_end_p50=None,end_to_end_p95=None,
                latency_note=('association trace synchronizes tensors; diagnostic timing only' if not value.get('speed_eligible',True) else 'shared GPU contention detected; no fair speed claim' if path.parent.name=='B2_l17_sym_w025_smoke60' else 'full-video offline; detector latency is not online identity latency'),source=str(path)))
            experiments.append(dict(path=str(path),sha256=sha256(path),**value))
        elif path.name.endswith('.metrics.json'):
            value=json.loads(path.read_text());metric_rows.append(dict(method=path.parent.name,video_id=value['video_id'],full_video=value['full_video'],causal_online=value['causal_online'],
                raw_gt_diagnostic=value['allow_duplicate_gt'],**value['metrics'],source=str(path),sha256=sha256(path)))
    # Online runs have a separate manifest and real input-to-identity latency.
    for path in sorted((RUNTIME/'online').glob('*/manifest.json')):
        value=json.loads(path.read_text())
        runtime_rows.append(dict(method=path.parent.name,video_id=value['video_id'],full_video=value['full_video'],causal_online=True,speed_eligible=value.get('speed_eligible',True),
            camera_image_fps=value['camera_image_fps'],synchronized_group_fps=value['synchronized_group_fps'],total_seconds=value['seconds'],
            inference_seconds=None,read_preprocess_seconds=None,peak_gpu_bytes=value['peak_gpu_bytes'],geo_calls=value.get('geo_calls',0),
            geo_seconds=None,detection_p50=None,detection_p95=None,end_to_end_p50=value['input_to_identity_p50'],end_to_end_p95=value['input_to_identity_p95'],
            latency_note='current synchronized group image reading to CPU identity output; static model setup excluded; cold lazy geometry included',source=str(path)))
        experiments.append(dict(path=str(path),sha256=sha256(path),**value))
        for metric_path in path.parent.glob('*.metrics.json'):
            report=json.loads(metric_path.read_text())
            metric_rows.append(dict(method=path.parent.name,video_id=report['video_id'],full_video=report['full_video'],causal_online=report['causal_online'],raw_gt_diagnostic=report['allow_duplicate_gt'],**report['metrics'],source=str(metric_path),sha256=sha256(metric_path)))
    for suite_path in sorted((ROOT/'results').glob('*_suite*_metrics.json')):
        value=json.loads(suite_path.read_text())
        if value.get('status')!='PASS':continue
        metric_rows.append(dict(method=suite_path.stem.removesuffix('_metrics'),video_id='ALL_STRICT',full_video=True,causal_online=value['causal_online'],raw_gt_diagnostic=False,**value['metrics'],source=str(suite_path),sha256=sha256(suite_path)))
    metrics=['method','video_id','full_video','causal_online','raw_gt_diagnostic','HOTA','DetA','AssA','IDF1','CVIDF1','CVMA','CVMA_sequential','MOTA','IDSW','source','sha256']
    runtime=['method','video_id','full_video','causal_online','speed_eligible','camera_image_fps','synchronized_group_fps','total_seconds','inference_seconds','read_preprocess_seconds','peak_gpu_bytes','geo_calls','geo_seconds','detection_p50','detection_p95','end_to_end_p50','end_to_end_p95','latency_note','source']
    write_csv(ROOT/'results/ablation_results.csv',metric_rows,metrics);write_csv(ROOT/'results/runtime_results.csv',runtime_rows,runtime)
    save_json(ROOT/'results/experiment_manifest.json',dict(status='IN_PROGRESS',experiments=experiments,raw_gt_policy='unchanged, diagnostic and strict protocols separate',split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text())))
    baseline=[r for r in metric_rows if r['method'].startswith('B0') and isinstance(r['video_id'],int) and r['full_video'] and not r['raw_gt_diagnostic']]
    summary=dict(status='IN_PROGRESS',scientific_success='NOT_ESTABLISHED',engineering_complete=False,paper_ready=False,
        complete_strict_baseline_videos=sorted({r['video_id'] for r in baseline}),metrics=metric_rows,feature_screening_complete=(ROOT/'results/feature_screening_manifest.json').exists(),
        off_zero_parity=json.loads((ROOT/'results/off_zero_parity.json').read_text()) if (ROOT/'results/off_zero_parity.json').exists() else None,
        limitations=['original GMT offline full-video postprocessing','conflicting raw GT identities in several original TRAIN sequences','no paired gain established','WWW application evidence pending'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save_json(ROOT/'results/final_summary.json',summary)
    lines=['# VGGT+GMT report (automatically generated; research in progress)','',
        'SCIENTIFIC_SUCCESS: not established. ENGINEERING_COMPLETE: false. PAPER_READY: false.','',
        'Tables below are read from actual evaluation files. Raw-GT diagnostic rows cannot substitute for strict evaluation. All input frames are retained. Original GMT full-video postprocessing is offline; no real-time identity latency claimed.','',
        '| Method | Video | Strict GT | HOTA | AssA | IDF1 | CVIDF1 | IDSW |','|---|---:|---|---:|---:|---:|---:|---:|']
    for row in metric_rows:
        lines.append(f"| {row['method']} | {row['video_id']} | {'no: raw diagnostic' if row['raw_gt_diagnostic'] else 'yes'} | {row['HOTA']:.4f} | {row['AssA']:.4f} | {row['IDF1']:.4f} | {row['CVIDF1']:.4f} | {row['IDSW']} |")
    lines+=['','Required remaining work: dense-update residual paired full-video tests and counterfactual activity diagnosis; geometry failure recovery; actual adaptive/async causal integration with separate baseline; matched-call-budget scheduler experiments; repeated stability checks and WWW retrieval evidence. Completed sparse feature tests, source/pose diagnostics and causal OFF results are evidence with limited scope, not completion of the requested experiment matrix.','',
        'See results/experiment_manifest.json for exact commands, seeds, checkpoint SHA, environment, cache status and external artifact paths. See docs/GEO_STREAM_RESEARCH_LOG.md for commits and failed runs.','']
    (ROOT/'docs/GEO_STREAM_FINAL_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(strict_baseline_videos=summary['complete_strict_baseline_videos'],metric_rows=len(metric_rows),runtime_rows=len(runtime_rows)),indent=2))
if __name__=='__main__':main()
