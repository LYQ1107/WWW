"""Read-only apparent storage footprint; external symlink targets are excluded."""
from collections import defaultdict
from common import *


GROUPS={
    'frozen_model_weights':{'weights'},
    'environment_under_runtime':{'venv'},
    'source_snapshots':{'pinned_sources','sources'},
    'compact_features_heads_and_samples':{'features','hard_features','hard_events','history_train','heads','samples'},
    'inference_retrieval_and_evaluation_artifacts':{'baseline','online','suites','retrieval','retrieval_outcomes','matrix_reports'},
    'logs_controls_and_diagnostics':{'camera_order','correctness','encoder_microbench','geometry','handoffs','logs','preflight','priority_windows','queues','resource_samples'},
}


def main():
    entries=sorted(RUNTIME.iterdir())
    command=['du','--bytes','--summarize','--',*map(str,entries)]
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    output=subprocess.check_output(command,text=True)
    measured={}
    for line in output.splitlines():
        value,path=line.split('\t',1);measured[path]=int(value)
    if set(measured)!=set(map(str,entries)) or any(value<0 for value in measured.values()):
        raise ValueError('incomplete storage accounting; no missing values filled with zero')
    categories=defaultdict(int);rows=[]
    for path in entries:
        matches=[group for group,names in GROUPS.items() if path.name in names]
        if len(matches)>1:raise ValueError('overlapping storage categories')
        category=matches[0] if matches else 'uncategorized'
        size=measured[str(path)];categories[category]+=size
        rows.append(dict(path=str(path),category=category,apparent_bytes=size,is_symlink=path.is_symlink()))
    result=dict(status='MEASURED_CURRENT_STORAGE_SNAPSHOT',started_utc=started,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,
        total_apparent_bytes=sum(measured.values()),category_apparent_bytes=dict(categories),rows=rows,
        scope='current project-owned runtime entries including hidden entries, GNU du apparent bytes; symlink targets are not followed; derived repository reports are outside this runtime root',
        limitations='not an atomic filesystem snapshot, filesystem-allocated bytes or a lifetime storage peak; live artifacts may grow; referenced external dataset/GMT checkpoint/environment storage is not included; source snapshots and frozen weights are separate from compact feature/training artifacts; no dense-token or offline-throughput claim follows from directory sizes',
        provenance=provenance())
    save_json(ROOT/'results/storage_budget_v1.json',result)
    text=['# Project storage budget snapshot','',result['scope']+'.','',result['limitations']+'.','',
          '| Category | Apparent bytes | GiB |','|---|---:|---:|']
    for group,size in sorted(categories.items()):text.append(f'| {group} | {size} | {size/(1<<30):.6f} |')
    text+=['',f"Total apparent bytes: {result['total_apparent_bytes']}. Exact command, timestamps and all per-entry values are in results/storage_budget_v1.json. No cleanup is performed; the command writes its derived reports and stdout.",'']
    (ROOT/'docs/GEO_STREAM_STORAGE_BUDGET.md').write_text('\n'.join(text))
    print(json.dumps(dict(total_apparent_bytes=result['total_apparent_bytes'],category_apparent_bytes=dict(categories))))


if __name__=='__main__':main()
