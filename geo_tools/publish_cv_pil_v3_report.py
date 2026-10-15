"""Preserve raw local reports; publish nonduplicated review evidence only."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from cv_pil_native_handoff import sha256


def publish(run, dest, mechanism_name):
    dest.mkdir(parents=True,exist_ok=False)
    raw=json.loads((run/'RESULTS.json').read_text())
    summary={k:v for k,v in raw.items() if k not in ('by_scene','by_person_camera','by_person_camera_time50')}
    summary['full_local_results']=dict(path=str(run/'RESULTS.json'),sha256=sha256(run/'RESULTS.json'))
    (dest/'RESULTS.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for name in ('QUERY_RESULTS_OFFLINE_ONLY.csv','INPUT_PINS.json'):
        shutil.copyfile(run/name,dest/name)
    shutil.copyfile(run/mechanism_name,dest/'ORIGINAL_ERROR_MECHANISM_AUDIT.json')
    fieldnames=['grouping','group','method',*raw['methods']['C0'].keys()]
    with (dest/'GROUP_METHOD_METRICS_OFFLINE_ONLY.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fieldnames,lineterminator='\n');w.writeheader()
        for grouping in ('by_scene','by_person','by_person_camera','by_person_camera_time50'):
            for group,record in raw[grouping].items():
                for arm,metrics in record['methods'].items():w.writerow(dict(grouping=grouping,group=group,method=arm,**metrics))
    pair_fields=['grouping','group','comparison','kind','certified_fixes','certified_harms','unique_fix_person_groups',
                 'unique_harm_person_groups','net_positive_person_groups','correct_to_unknown','unknown_to_correct']
    with (dest/'GROUP_PAIR_METRICS_OFFLINE_ONLY.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=pair_fields,lineterminator='\n');w.writeheader()
        for grouping in ('by_scene','by_person','by_person_camera','by_person_camera_time50'):
            for group,record in raw[grouping].items():
                for comparison in ('C3_vs_C0','C3_vs_C2','C3_vs_C4'):
                    for kind,pair in record['paired'][comparison].items():
                        if kind not in ('rank','primary'):continue
                        w.writerow(dict(grouping=grouping,group=group,comparison=comparison,kind=kind,
                                        **{k:pair[k] for k in pair_fields[4:]}))
    provenance=dict(raw_local_files={str(p):dict(sha256=sha256(p),bytes=p.stat().st_size) for p in run.iterdir() if p.is_file()},
                    public_files={p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in dest.iterdir() if p.is_file()},
                    reduction='Full local raw JSON preserved; duplicated camera/time nested JSON published as equivalent metric tables. Exact full-query CSV retained.',
                    weights_features_videos_uploaded=False)
    (dest/'PUBLICATION_MANIFEST.json').write_text(json.dumps(provenance,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,required=True)
    p.add_argument('--mechanism-name',default='ORIGINAL_ERROR_MECHANISM_AUDIT.json');a=p.parse_args();publish(a.run,a.dest,a.mechanism_name)


if __name__=='__main__':main()
