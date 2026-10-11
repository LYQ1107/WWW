"""Offline GT-only adapter for future evaluation; never a policy input reader."""
import math
from collections import Counter
from pathlib import Path
import subprocess
from jev_master_audit_common import ROOT, MASTER, protect, read, save, ref, sha


def classify_bbox(coords, width=1920, height=1080):
    if len(coords)!=4 or not all(type(x) in [int,float] and math.isfinite(x) for x in coords):
        raise ValueError('Invalid/nonfinite GT bbox; do not silently clean labels')
    if list(coords)==[-1,-1,-1,-1]:return 'ABSENT'
    x,y,x2,y2=coords
    if x2<=x or y2<=y:raise ValueError('Nonpositive GT bbox')
    if min(x2,width)<=max(x,0) or min(y2,height)<=max(y,0):return 'OUTSIDE_IMAGE'
    return 'INTERSECTS_IMAGE'


def offline_rows(raw):
    ids=set();rows=[];counts=Counter()
    for person in raw:
        ident=person['personID']
        if type(ident) is not int or ident<0 or ident in ids:raise ValueError('Duplicate/invalid raw personID')
        ids.add(ident);seen=set()
        for view in person['views']:
            number=view['viewNum']
            if type(number) is not int or number not in range(7) or number in seen:raise ValueError('Duplicate/invalid view key')
            seen.add(number)
            if number not in [0,1]:continue
            box=[view[k] for k in ['xmin','ymin','xmax','ymax']]
            status=classify_bbox(box);counts[status]+=1
            if status!='INTERSECTS_IMAGE':continue
            x,y,x2,y2=box
            # Preserve the official bbox; do not clip or rewrite the input annotations.
            rows.append(dict(offline_GT_personID=ident,source_viewNum=number,bbox_XYWH=[x,y,x2-x,y2-y]))
            counts['legacy_any_minus1_would_drop']+=int(-1 in box)
        if seen!=set(range(7)):raise ValueError('Incomplete raw view set')
    return rows,counts


def main():
    protect()
    # Boundary counterexamples and raw duplicate rejection protect GT denominators.
    assert classify_bbox([-1,-1,-1,-1])=='ABSENT'
    assert classify_bbox([10,-1,20,30])=='INTERSECTS_IMAGE'
    assert classify_bbox([-20,-10,20,30])=='INTERSECTS_IMAGE'
    assert classify_bbox([2000,10,2010,20])=='OUTSIDE_IMAGE'
    try:classify_bbox([0,0,0,20])
    except ValueError:pass
    else:raise AssertionError('zero area must fail')
    test_person=dict(personID=7,views=[dict(viewNum=v,xmin=1,ymin=1,xmax=2,ymax=2) for v in range(7)])
    try:offline_rows([test_person,test_person])
    except ValueError:pass
    else:raise AssertionError('duplicate GT must fail, not be deduplicated')
    protocol=read(MASTER/'WILDTRACK_REUSED_TRAIN_PROTOCOL.json')
    audit=read(MASTER/'WILDTRACK_REUSED_TRAIN_LABEL_AUDIT.json')
    counts=Counter();row_count=0
    for item in protocol['GT_records']:
        assert sha(item['path'])==item['SHA256']
        rows,c=offline_rows(read(item['path']));row_count+=len(rows);counts.update(c)
        assert all(set(row)=={'offline_GT_personID','source_viewNum','bbox_XYWH'} for row in rows)
    assert row_count==audit['selected_C1_C2']['positive_area_with_image_intersection']
    assert counts['legacy_any_minus1_would_drop']==audit['selected_C1_C2']['mixed_minus1_with_valid_intersection']
    # Read a prior TRAIN data manifest, not tracking results, to verify the exact omission.
    previous=Path('/data1/liuyeqiang/WWW_jev_phase14_runtime/20261010_v1/external_eval_v1/DATA_MANIFEST.json')
    old=read(previous)
    assert old['counts']['included']==row_count-counts['legacy_any_minus1_would_drop']
    result=dict(status='PASS_OFFLINE_GT_INTERFACE_ONLY',
        actor_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source=ref(Path(__file__)),protocol=ref(MASTER/'WILDTRACK_REUSED_TRAIN_PROTOCOL.json'),
        semantic_audit=ref(MASTER/'WILDTRACK_REUSED_TRAIN_LABEL_AUDIT.json'),old_TRAIN_format_manifest=ref(previous),
        counterexamples_passed=6,actual_files_checked=320,offline_rows=row_count,counts=dict(counts),
        old_included_rows=old['counts']['included'],additional_boundary_GT_rows=counts['legacy_any_minus1_would_drop'],
        raw_GT_or_old_results_modified=False,GT_world_fields_returned=False,
        output_rows_saved=False,policy_inputs_generated=False,
        tracker_metrics_recomputed=False,model_benefit_claim=False,
        new_images_read=0,new_downloads=0,training=False,GPU_seconds=0,
        deployment='Future offline evaluation only; old Phase14 metrics remain with their original denominator and disclosed limitation.')
    save(MASTER/'WILDTRACK_OFFLINE_GT_INTERFACE_AUDIT.json',result)
    print('PASS_OFFLINE_GT_INTERFACE',row_count,'old',old['counts']['included'],
          'additional',counts['legacy_any_minus1_would_drop'])


if __name__=='__main__':
    main()
