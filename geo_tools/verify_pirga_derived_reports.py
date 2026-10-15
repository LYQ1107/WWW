"""Independently reconcile camera/error-type tables with sealed endpoint totals."""
import csv,datetime,json
from pathlib import Path
from pirga_indexed_store import sha


def main():
    repo=Path(__file__).resolve().parents[1];root=repo/'results/PIRGA_FULLTRAIN'
    registry=root/'ORIGINAL_ERROR_REPAIRS.json';camera=root/'PER_CAMERA_RESULTS.csv';transitions=root/'ORIGINAL_ERROR_ACTION_TRANSITIONS.csv'
    with camera.open() as f:cam=list(csv.DictReader(f))
    with transitions.open() as f:err=list(csv.DictReader(f))
    groups={};checks=0
    for row in cam:groups.setdefault((row['family'],row['endpoint'],row['method']),[]).append(row)
    seen=set()
    for entry in json.loads(registry.read_text())['entries']:
        path=repo/entry['path']
        if sha(path)!=entry['sha256']:raise ValueError('Endpoint source changed')
        data=json.loads(path.read_text());family=entry['family'];endpoint=path.parent.name
        for arm,method in data['methods'].items():
            key=(family,endpoint,arm);seen.add(key);values=groups[key]
            if len({r['camera'] for r in values})!=len(values):raise ValueError('Duplicate camera stratum')
            for field in ['queries','positive_covered','primary_known_correct','primary_known_wrong','primary_unknown']:
                if sum(int(r[field]) for r in values)!=method[field]:raise ValueError('Camera total does not reconcile '+str((key,field)))
                checks+=1
            covered=method['positive_covered']
            for field in ['top1','mrr']:
                mean=sum(float(r[field])*int(r['positive_covered']) for r in values if int(r['positive_covered']))/covered if covered else None
                if (mean is None)!=(method[field] is None) or (mean is not None and abs(mean-method[field])>1e-10):raise ValueError('Camera metric does not reconcile '+str((key,field)))
                checks+=1
            if 'original_error_methods' in data:native=data['original_error_methods'][arm]
            else:native=data.get('mechanism',data.get('original_error_mechanism'))['methods'][arm]
            for kind in ['wrong_link','missed_known_positive']:
                rows=[r for r in err if (r['family'],r['endpoint'],r['method'],r['original_error_type'])==(family,endpoint,arm,kind)]
                if len(rows)!=1:raise ValueError('Error-type axis missing or duplicated')
                r=rows[0];partition=['accepted_known_correct','accepted_known_wrong','accepted_unknown','rejected_with_pure_positive','rejected_without_pure_positive']
                if sum(int(r[k]) for k in partition)!=int(r['original_errors']):raise ValueError('Error outcomes not a partition')
                if int(r['original_errors'])!={'wrong_link':6,'missed_known_positive':20}[kind]:raise ValueError('Frozen original errors changed')
                checks+=1
            repairs=native.get('original_error_repairs',native.get('certified_reachable_original_primary_errors_repaired'))
            harms=native.get('known_harms_to_original_correct',native.get('originally_correct_primary_known_harms'))
            if sum(int(r['original_error_repairs']) for r in values)!=repairs or sum(int(r['known_harms']) for r in values)!=harms:raise ValueError('Camera causal counts do not reconcile')
            checks+=2
    if set(groups)!=seen:raise ValueError('Unexpected family/endpoint/method')
    result=dict(status='PASS_REAL_DERIVED_REPORT_RECONCILIATION',method_endpoint_groups=len(seen),camera_strata=len(cam),checks=checks,
        input_sha256={str(p.relative_to(repo)):sha(p) for p in [registry,camera,transitions]},observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scientific_success=False,qualification='Reconciles exact same-state DEV12 reporting totals and weighted metrics; camera strata share people and are not independent trials. No online metric or independent validation certified.')
    (root/'DERIVED_REPORT_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
