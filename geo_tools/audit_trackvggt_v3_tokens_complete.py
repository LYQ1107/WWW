"""Complete the existing frozen-feature queue receipt; never re-extract."""
from common import *
from trackvggt_data import TrackEventData, NEW
from evaluate_teacher_pairs import evaluate


def main():
    controls=json.loads((ROOT/'results/trackvggt_v2_loader_native_control_v2.json').read_text())
    cohort=NEW/'cohorts/hard_events_expanded_v3/cohort.json';value=json.loads(cohort.read_text());summaries=[];queue_rows=[]
    for gpu in [0,4,5,6]:
        path=NEW/'queues'/f'trackvggt_v3_tokens_v1_gpu{gpu}/manifest.json';queue=json.loads(path.read_text())
        if queue['status']!='FINISHED' or any(s['status']!='PASS' for s in queue['stages']):raise ValueError('frozen queue incomplete')
        queue_rows.append(dict(gpu=gpu,manifest=str(path),sha256=sha256(path),outer_process_seconds=sum(s['seconds'] for s in queue['stages'])))
    for backbone in ['vggt','dinov2']:
        data=TrackEventData(backbone,cohort_path=cohort)
        expected=next(r for r in controls['rows'] if r['backbone']==backbone)
        native=evaluate(None,data,data.validation,scorer='native')['metrics'];prior=evaluate(None,data,data.validation,scorer='prior')['metrics']
        if native!=expected['native'] or prior!=expected['prior']:raise ValueError('fixed original300 validation/native/prior changed')
        new_manifests=[Path(p) for p in value['feature_manifest_paths'][backbone] if p.endswith('_new_v3/manifest.json')]
        events=0;newbytes=0;encoder_calls=0;manifest_rows=[]
        for path in new_manifests:
            m=json.loads(path.read_text())
            if m['cohort_sha256']!=sha256(cohort):raise ValueError('frozen cohort drift')
            for name,digest in m['source_artifacts'].items():
                if sha256(name)!=digest:raise ValueError('frozen encoder input changed')
            events+=m['events'];newbytes+=m['token_bytes'];encoder_calls+=m['encoder_calls']
            manifest_rows.append(dict(path=str(path),sha256=sha256(path),targets=m['events'],token_bytes=m['token_bytes']))
        if events!=2428 or len(data.rows)!=3300 or len(data.target_rows)!=3728:raise ValueError('physical pool completeness')
        summaries.append(dict(backbone=backbone,fit=len(data.fit),validation=len(data.validation),primary_targets=len(data.rows),
            total_main_and_auxiliary_targets=len(data.target_rows),new_physical_targets=events,reused_V2_physical_targets=1300,
            new_token_bytes=newbytes,new_encoder_calls=encoder_calls,new_manifests=manifest_rows,
            validation_native=native,validation_prior=prior,validation300_exact_previous_control=True))
    total=sum(p.stat().st_size for p in (NEW/'tokens').rglob('*.npz'))
    if total>2*(1<<30):raise ValueError('shared target-cache cap violated')
    report=dict(status='PASS_V3_FROZEN_FEATURES_COMPLETE',cohort_sha256=sha256(cohort),rows=summaries,queues=queue_rows,
        shared_all_NPZ_bytes_including_existing_probes=total,shared_all_NPZ_cap_bytes=2*(1<<30),
        frozen_feature_outer_gpu_wall_hours=sum(r['outer_process_seconds'] for r in queue_rows)/3600,
        no_new_encoder_calls_for_this_audit=True,teacher_fitting_started=False,scientific_success=False,
        independent_from_DEV12_native_asset_release=True,provenance=provenance())
    save_json(ROOT/'results/trackvggt_v3_tokens_complete_v1.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('provenance','rows','queues')}))


if __name__=='__main__':main()
