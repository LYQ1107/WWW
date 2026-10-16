"""Import a terminal CIR cost record once, preserving every old stage and file."""
import argparse
import copy
import json
from pathlib import Path
from common import sha256
from run_cir_guarded import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--child',type=Path,required=True);a=p.parse_args()
    parent=json.loads(a.parent.read_text());child=json.loads(a.child.read_text());digest=sha256(a.child)
    if child['parent_root']!=str(a.parent.resolve()) or not child['parent_existing_credit_only'] or child['new_GPU_credit']!=0:
        raise ValueError('Only reserved existing CIR credit may be imported')
    if any(s['status'].startswith('RUNNING') for s in parent['stages']+child['stages']):
        raise ValueError('Both resource writers must be terminal before merging')
    if not child['stages']:raise ValueError('No actual stage to import')
    if any(r['path']==str(a.child.resolve()) for r in parent.get('imported_subledgers',[])):
        raise ValueError('Cost record already imported; never count twice')
    names={s['stage'] for s in parent['stages']}
    if any(s['stage'] in names for s in child['stages']):raise ValueError('Duplicate existing stage identity')
    used=child['measured_GPU_wall_hours']
    if used<0 or used>child['allocated_total_GPU_wall_hours'] or parent['measured_GPU_wall_hours']+used>parent['allocated_total_GPU_wall_hours']:
        raise ValueError('Recorded cost exceeds existing registered credit')
    evidence=dict(path=str(a.child.resolve()),sha256=digest,GPU_wall_hours=used,new_GPU_credit=0)
    for stage in copy.deepcopy(child['stages']):
        stage['cost_record']=evidence;parent['stages'].append(stage)
    parent['measured_GPU_wall_hours']+=used;parent.setdefault('imported_subledgers',[]).append(evidence)
    if sha256(a.child)!=digest:raise ValueError('Cost record changed during read')
    save(a.parent,parent);print(json.dumps(evidence),flush=True)


if __name__=='__main__':main()
