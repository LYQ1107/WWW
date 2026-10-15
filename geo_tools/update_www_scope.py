"""Generate the scope note from verified CFP facts and actual retrieval runs."""
from common import *


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    cfp_path=ROOT/'manifests/www_cfp_verified_v1.json';cfp=json.loads(cfp_path.read_text())
    reference=RUNTIME/'retrieval/retrieval_gmt_v18_full_v1'
    original_queries=[row['query'] for row in lines(reference/'queries.jsonl') if row['query']['scene']==18]
    fields=('scene','camera','round_id','image_id','frame_id','bbox')
    def cohort(rows):return [{key:row[key] for key in fields} for row in rows]
    def detector(rows):return [{key:value for key,value in row.items() if key!='track_id'} for row in rows]
    reference_detections=detector(json.loads((reference/'video_18.json').read_text()))
    sources={str(cfp_path):sha256(cfp_path)};rows=[]
    for path in sorted((RUNTIME/'retrieval').glob('*/manifest.json')):
        run=json.loads(path.read_text())
        if not run.get('full_video') or 18 not in run['video_ids'] or not run.get('speed_eligible',True):continue
        metric=path.parent/('retrieval.scene_18.metrics.json' if len(run['video_ids'])>1 else 'retrieval.metrics.json')
        if not metric.exists():continue
        result=json.loads(metric.read_text())
        queries=[row['query'] for row in lines(path.parent/'queries.jsonl') if row['query']['scene']==18]
        if len(queries)!=result['requests']:raise ValueError('scene18 query/metric scope mismatch')
        for name in ('queries.jsonl','video_18.json'):
            if sha256(path.parent/name)!=run['artifacts'][name]:raise ValueError('changed retrieval artifact')
        same=cohort(queries)==cohort(original_queries)
        same_det=detector(json.loads((path.parent/'video_18.json').read_text()))==reference_detections
        rows.append(dict(method=path.parent.name,tracking_method=run.get('tracking_method','gmt_off'),camera_streams=run['camera_streams'],backbone=run['backbone'],interval=run['interval'],
                         requests=result['requests'],eligible_queries=result['gallery_eligible_queries'],recall1=result['recall_at_k']['1'],mrr=result['mrr'],
                         exact_detector_query_cohort=same,exact_full_scene18_detector_outputs=same_det,metrics=str(metric),metrics_sha256=sha256(metric),manifest_sha256=sha256(path)))
        sources[str(metric)]=sha256(metric);sources[str(path)]=sha256(path)
    text=['# WWW scope and deadlines','',
          f"[Official Research Track CFP]({cfp['url']}) verified {cfp['checked_utc'][:10]}: abstract {cfp['abstract_deadline_aoe']} AoE, full paper {cfp['full_paper_deadline_aoe']} AoE. These end at 20:00 Asia/Shanghai on the following calendar day. Target track: {cfp['target_track']}.",'',
          cfp['relevance_summary']+f" [Source]({cfp['url']})",'',
          'The executable task queries a current detector target against clips from other cameras. Actual image reads, frozen causal GMT, feature publication, bounded index maintenance and query responses run online in sequence; labels enter post-run evaluation only. Separate scenes interleave by replay round and retain independent tracker banks/RNG. Frozen encoder weights are already shared serially across scene extractors; no cross-scene tracking/geometry state is shared.', '',
          'Completed production tracker modes in this table: '+', '.join(sorted({row['tracking_method'] for row in rows}))+'. Original experiments keep tracking at causal GMT OFF and vary retrieval features/scoring. Optional fixed-prior and dense B7 tracking-to-retrieval controls are separately gated; only completed metric artifacts enter this table. Local replay does not establish Internet-origin streams, live Web deployment, real user-query traffic or an independent Web scientific contribution.', '',
          'Production retrieval results below use scene18 queries at every stream count. Exact query image/camera/frame/box cohorts and complete scene18 detector outputs are checked against the two-camera native reference. Recall@1/MRR are conditional on matched queries with positive gallery clips; coverage and all-request success remain in the metric artifacts.', '',
          '| Method | Tracking | Cameras | Queries | Eligible | Recall@1 | MRR | Exact cohort/detections |',
          '|---|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        text.append(f"| {row['method']} | {row['tracking_method']} | {row['camera_streams']} | {row['requests']} | {row['eligible_queries']} | {100*row['recall1']:.4f}% | {row['mrr']:.6f} | {row['exact_detector_query_cohort']}/{row['exact_full_scene18_detector_outputs']} |")
    outcome_path=ROOT/'results/retrieval_paired_query_outcomes_v1.json'
    if outcome_path.exists():
        outcomes=json.loads(outcome_path.read_text());sources[str(outcome_path)]=sha256(outcome_path)
        candidate=next((row for row in outcomes['rows'] if row['method']=='VGGT2'),None)
        if candidate:
            text+=['',f"The completed dense-B7 two-camera tracker feeding the native ranker has {candidate['common_eligible_rank1_fixes']} Recall1 query corrections and {candidate['common_eligible_rank1_harms']} harms on {candidate['common_eligible']} common eligible scene18 queries, with eligible gained/lost {candidate['eligible_gained']}/{candidate['eligible_lost']}. Same detector query cohort and GT identities are independently verified. This is a small primary-head replay signal; completed matched-head controls are reported individually below and remaining comparisons stay pending. See GEO_STREAM_RETRIEVAL_QUERY_AUDIT.md; no unique VGGT or stable speed claim follows.",'']
        for row in outcomes['rows']:
            if row['method'].startswith(('GMT-head','DINO-head')):
                text += [f"Completed matched publication control {row['method']}, fixed scene{row['query_scene']}: {row['requests']} exact detector queries/{row['common_eligible']} common eligible; R1 corrections/harms {row['common_eligible_rank1_fixes']}/{row['common_eligible_rank1_harms']}, eligibility gained/lost {row['eligible_gained']}/{row['eligible_lost']}, MRR delta {row['common_eligible_mrr_delta']:+.6f}, first-positive rank or eligibility changes {row['changed_rank_or_eligibility_queries']}. This is one primary-head local replay, not a completed stability or unique-foundation attribution result.",'']
            if row['method']=='VGGT4':
                text += [f"The complete four-camera dense-B7 replay, fixed scene{row['query_scene']}: {row['requests']} exact detector queries/{row['common_eligible']} common eligible, R1 corrections/harms {row['common_eligible_rank1_fixes']}/{row['common_eligible_rank1_harms']}, eligibility gained/lost {row['eligible_gained']}/{row['eligible_lost']}, MRR delta {row['common_eligible_mrr_delta']:+.6f}. Scene18 shares the two-camera corrected query; it is not an independent query replication. Per-scene cohorts are kept separate.",'']
        for row in outcomes.get('shared_control_corrections',[]):
            if row['vggt_rank1_fixes']:
                text += [f"Same ordered-query audit of {row['vggt_method']} versus {row['control_method']}, scene{row['query_scene']}: VGGT/control R1 fixes {row['vggt_rank1_fixes']}/{row['control_rank1_fixes']}, shared fixes {row['shared_rank1_fixes']}, VGGT fixes not reproduced by this control {row['vggt_fixes_not_reproduced_by_this_control']}. Equal aggregate gains are insufficient to infer the same corrected query; different corrections in a single replay do not establish stable unique VGGT benefit.",'']
    text += ['',
        'Same-gallery diagnostics retain the full native positive-gallery denominator. The sampled-observation native scorer is stronger than transferred association-trained projection/cosine scores in the completed scene18 tests; positive candidate coverage is 100%. These diagnose scoring/representation transfer and do not show all raw foundation features are useless. Their five-ranker producer timings are diagnostic cost, not production throughput. See retrieval_scoring_diagnostics.csv and the source metrics.', '',
        'Query latency measures local scoring/response serialization after representations are available. Full replay-round timing separately charges reads, tracking, optional encoder work, publication, queries and log I/O. It is not HTTP/network or camera-capture-to-output latency. Source acquisition timing and a deadline-paced live camera service have not been verified. Index payload caps do not bound original GMT history or total process/GPU memory.', '',
        'Remaining scope evidence: complete the planned stream/keyframe matrix and candidate tracking-to-retrieval evaluation, preserve fixed-query comparisons, and establish an actual Web scientific problem/application rather than changing a tracking title. Current evidence does not justify PAPER_READY; this fit risk does not stop the technical research.', '']
    (ROOT/'docs/WWW_SCOPE_NOTE.md').write_text('\n'.join(text))
    save_json(ROOT/'results/www_scope_evidence_v1.json',dict(status='IN_PROGRESS_SCOPE_EVIDENCE',production_fixed_scene18_rows=rows,
        all_query_cohorts_exact=all(row['exact_detector_query_cohort'] for row in rows),all_scene18_detector_outputs_exact=all(row['exact_full_scene18_detector_outputs'] for row in rows),
        source_artifacts=sources,paper_ready=False,provenance=provenance()))


if __name__=='__main__':main()
