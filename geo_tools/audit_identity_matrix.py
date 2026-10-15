"""Complete per-ablation observable identity relations with explicit scope."""
import argparse
import collections
import csv
from common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate-missing',action='store_true');args=parser.parse_args()
    plan_path=ROOT/'manifests/identity_relations_matrix_v2.json';plan=json.loads(plan_path.read_text())
    strict=sorted(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
    existing={}
    for path in (ROOT/'results').glob('*identity*pair*audit*.json'):
        audit=json.loads(path.read_text())
        if audit.get('status')=='PASS' and audit.get('full_strict_suite') and audit.get('sources_sha256'):
            existing[tuple(audit['sources_sha256'])]=path
    split_path=ROOT/'manifests/geo_split_v1.json'
    rows=[];pending=[];artifacts={str(plan_path):sha256(plan_path),str(split_path):sha256(split_path)}
    for entry in plan['methods']:
        metric=ROOT/'results'/f"{entry['suite']}_metrics.json"
        if not metric.exists():pending.append(dict(**entry,reason='complete native suite metric not yet available'));continue
        measured=json.loads(metric.read_text())
        if measured.get('status')!='PASS' or not measured.get('full_video') or measured['video_ids']!=strict:
            raise ValueError('invalid complete suite: '+entry['suite'])
        causal=entry['protocol']=='causal'
        if measured['causal_online']!=causal:raise ValueError('wrong declared protocol')
        sources=[ROOT/'manifests'/name for name in (entry['baseline_sources'],entry['method_sources'])]
        digests=tuple(sha256(source) for source in sources)
        report=existing.get(digests,ROOT/'results'/f"matrix_{entry['suite']}_identity_pair_audit_v2.json")
        if not report.exists() and args.generate_missing:
            log=RUNTIME/'logs'/f"matrix_{entry['suite']}_identity_pair_audit_v2.log"
            with log.open('x') as stream:
                subprocess.run([sys.executable,str(ROOT/'geo_tools/audit_tracking_identity_pairs.py'),
                    '--baseline-sources',str(sources[0]),'--method-sources',str(sources[1]),'--tag',report.stem],
                    cwd=ROOT,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT)
        if not report.exists():pending.append(dict(**entry,reason='identity relation audit missing'));continue
        audit=json.loads(report.read_text())
        if audit['status']!='PASS' or not audit['full_strict_suite'] or tuple(audit['sources_sha256'])!=digests:
            raise ValueError('unpaired/incomplete relation audit')
        specs=[json.loads(source.read_text())['video_predictions'] for source in sources]
        if any(sorted(map(int,spec))!=strict for spec in specs) or sorted(row['video_id'] for row in audit['rows'])!=strict:
            raise ValueError('missing strict scenes')
        for row in audit['rows']:
            video=row['video_id'];paths=[Path(spec[str(video)]) for spec in specs]
            if [row['baseline_prediction'],row['method_prediction']]!=list(map(str,paths)):
                raise ValueError('different prediction scope')
            if [sha256(path) for path in paths]!=row['prediction_sha256']:
                raise ValueError('prediction artifact changed')
            runpath=paths[1].with_name(paths[1].stem+'_manifest.json')
            if not runpath.exists():runpath=paths[1].parent/'manifest.json'
            run=json.loads(runpath.read_text());capture=measured['run_manifests'][str(video)]
            if str(runpath)!=capture['path'] or sha256(runpath)!=capture['sha256'] or run['causal_online']!=causal:
                raise ValueError('metric and relation inference differ')
        for path,digest in audit['raw_gt_sha256'].items():
            if sha256(path)!=digest:raise ValueError('raw GT changed')
        summary=dict(method=entry['method'],suite=entry['suite'],protocol=entry['protocol'],full_strict_suite=True,
            matched_detections=sum(row['all_time_identity_pairs']['detections'] for row in audit['rows']),
            added_detections=sum(row['added_detections'] for row in audit['rows']),removed_detections=sum(row['removed_detections'] for row in audit['rows']),
            raw_baseline_detections=sum(row['baseline_detections'] for row in audit['rows']),raw_method_detections=sum(row['method_detections'] for row in audit['rows']),
            CVIDF1=measured['metrics']['CVIDF1'],IDSW=measured['metrics']['IDSW'],source=str(report),source_sha256=sha256(report))
        for label,key in [('all','all_time_identity_pairs'),('cross_camera','cross_camera_all_time_identity_pairs')]:
            counts=collections.Counter()
            for row in audit['rows']:
                value=row[key]
                counts['pairs']+=value['unordered_pairs'];counts['positive_pairs']+=value['gt_positive_pairs']
                counts['baseline_wrong_method_correct']+=value['fixes_false_negative_pairs']+value['fixes_false_positive_pairs']
                counts['baseline_correct_method_wrong']+=value['harms_true_positive_pairs']+value['harms_true_negative_pairs']
            summary.update({label+'_'+name:count for name,count in counts.items()})
        rows.append(summary)
        artifacts.update({str(path):sha256(path) for path in [*sources,metric,report]})
        print(entry['suite'],'verified',flush=True)
    fields=list(rows[0]) if rows else []
    with (ROOT/'results/identity_relations_matrix_v2.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    text=['# Observable identity relations across complete ablations','',
        'Every row is independently paired to its same-protocol locked baseline over all five strict scenes. Counts classify unordered pairs of common, one-to-one IoU≥0.5 GT-matched detector outputs; pairs never cross scenes. Long tracks contribute quadratically. These are output relations, not numbers of Hungarian decisions. Added/removed and unmatched detections are separately excluded from correctness counts. GT enters only post-run analysis.','',
        'Offline GMT includes its original full-video postprocessing; causal GMT uses the separate streaming baseline. Values across those protocols are not paired comparisons. Missing complete suites remain pending. Actual pre-bank Hungarian decision recording is a separate scene1 diagnostic.','',
        '| Method | Protocol | Matched detections | Added/removed output detections | Cross-camera fixes | Cross-camera harms | CVIDF1 | IDSW |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['method']} / {row['suite']} | {row['protocol']} | {row['matched_detections']} | {row['added_detections']}/{row['removed_detections']} | {row['cross_camera_baseline_wrong_method_correct']} | {row['cross_camera_baseline_correct_method_wrong']} | {row['CVIDF1']:.6f} | {row['IDSW']} |")
    text+=['','Pending complete suites: '+', '.join(row['suite'] for row in pending)+'.','']
    (ROOT/'docs/GEO_STREAM_IDENTITY_RELATIONS.md').write_text('\n'.join(text))
    save_json(ROOT/'results/identity_relations_matrix_v2.json',dict(status='COMPLETE_DECLARED_MATRIX' if not pending else 'AVAILABLE_COMPLETE_SUITES_WITH_PENDING',
        rows=rows,pending=pending,source_artifacts=artifacts,scope=text[2],limitations=text[4],provenance=provenance()))


if __name__=='__main__':main()
