#!/usr/bin/env python3
"""Compare real object caches before/after separating bbox support from masks."""
import argparse
import json
from pathlib import Path
import numpy as np
from worldmot.models.person_detector import sha256_file


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();records=[];timestamps=0;observations=0
    for folder in (a.before,a.after):
        if json.loads((folder/'manifest.json').read_text())['status']!='COMPLETE':
            raise ValueError('Both real cache attempts must be complete')
    paths=sorted(a.before.glob('*.npz'))
    if not paths or [x.name for x in paths]!=[x.name for x in sorted(a.after.glob('*.npz'))]:
        raise ValueError('Nonempty identical shard coverage required')
    for path in paths:
        other=a.after/path.name;fields={}
        with np.load(path) as x,np.load(other) as y:
            if set(x.files)!=set(y.files):raise AssertionError('Array names changed')
            for key in x.files:
                if x[key].dtype!=y[key].dtype or not np.array_equal(x[key],y[key]):
                    raise AssertionError(f'Real cache changed: {path.name}/{key}')
                fields[key]={'shape':list(x[key].shape),'dtype':str(x[key].dtype),'bit_exact':True}
            timestamps+=len(x['timestamps']);observations+=len(x['observations'])
        records.append({'shard':path.name,'before_sha256':sha256_file(path),
            'after_sha256':sha256_file(other),'arrays':fields})
    report={'status':'PASS_REAL_ALL_ARRAYS_BIT_EXACT_AFTER_MASK_SEMANTICS_FIX',
        'before':str(a.before),'after':str(a.after),'timestamps':timestamps,'observations':observations,
        'before_manifest_sha256':sha256_file(a.before/'manifest.json'),
        'after_manifest_sha256':sha256_file(a.after/'manifest.json'),'shards':records,
        'scope':'Same 32 TRAIN Scene 8 timestamps. Exact object/geometry/identity/projection arrays; not a full-cache rerun or an accuracy gain.'}
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='shards'}),flush=True)


if __name__=='__main__':main()
