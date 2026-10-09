"""Verify compact report references and frozen assets before publishing results."""
import collections
from jev_phase14_common import *

REQUIRED=['PREREGISTRATION','PHASE13_FROZEN_EVIDENCE','ERROR_ATTRIBUTION','UNKNOWN_CHOICE_FORENSICS','FALSE_MERGE_SPLIT_DIAGNOSTICS','QUESTION_SUPERVISION_ELIGIBILITY','AVAILABILITY_INTERVENTION_DATA','STRUCTURED_ACTION_LOSS_ABLATION','IDENTITY_MEMORY_FACTORIAL','ON_POLICY_STABILITY','FAIR_BASELINE_RESULTS','ONLINE_VALIDATION','OFFICIAL_MATLAB_RESULTS','LATENCY_ATTRIBUTION','PRETRAIN_EXPOSURE_AUDIT','FINAL_GO_NO_GO','EXTERNAL_GENERALIZATION_RESULTS','FULL_VIDEO_CACHE_PARITY','PAIRED_NATIVE_FUTURE_RESULTS','SOURCE_INTERFACE_MANIFEST','PRIMARY_RESULTS_SNAPSHOT','EXTERNAL_RUNTIME_DIAGNOSTICS']
def main():
    protect();source=binding();assert read(REPORTS/'FINAL_GO_NO_GO.json')['execution_deliveries']=='COMPLETE'
    checked={};missing=[];mismatch=[];references=[]
    def visit(value,report,where=''):
        if isinstance(value,dict):
            if 'path' in value and 'SHA256' in value and isinstance(value['path'],str) and isinstance(value['SHA256'],str):
                p=Path(value['path']);expected=value['SHA256']
                if not p.is_absolute():p=ROOT/p
                if not p.is_file():missing.append(dict(report=report,field=where,path=str(p),SHA256=expected))
                else:
                    actual=checked.setdefault(str(p),None)
                    if actual is None:actual=sha(p);checked[str(p)]=actual
                    if actual!=expected:mismatch.append(dict(report=report,field=where,path=str(p),expected=expected,actual=actual))
                    else:references.append(dict(report=report,path=str(p),SHA256=actual))
            for k,v in value.items():visit(v,report,where+'/'+k)
        elif isinstance(value,list):
            for i,v in enumerate(value):visit(v,report,where+f'/{i}')
    reports=[]
    for name in REQUIRED:
        p=REPORTS/(name+'.json');assert p.exists(),str(p);d=read(p);reports.append(dict(name=name,status=d['status'],path=str(p),SHA256=sha(p)));visit(d,name)
    historical=read(REPORTS/'HISTORICAL_CHECKPOINT_AUDIT.json')
    assert historical['status']=='PASS'
    native=read(REPORTS/'FULL_VIDEO_CACHE_PARITY.json');assert len(native['cases'])==9 and all(all(x['equal'].values()) for x in native['cases'])
    actual=read(REPORTS/'SOURCE_INTERFACE_MANIFEST.json');configs={}
    for item in actual['sources']:
        for rel in ['configs/VISION_test.yaml','configs/VISION_stage1.yaml']:
            p=Path(item['path'])/rel;assert sha(p)==item['files'][rel];configs[str(p)]=item['files'][rel]
    metadata=read(OUT/'diagnostic_queue_v1/SCHEDULER_EXPANSION.json');state=Path('/proc')/str(metadata['PID'])/'status'
    if state.exists():assert 'State:\tT' not in state.read_text(),'original scheduler must be resumed'
    report=dict(status='PASS' if not missing and not mismatch else 'FAIL_REFERENCE_AUDIT',binding=source,required_reports=reports,verified_file_count=len(checked),reference_count=len(references),missing=missing,mismatch=mismatch,
        actual_config_SHA256=configs,actual_source_and_checkpoint_manifest=dict(path=str(REPORTS/'SOURCE_INTERFACE_MANIFEST.json'),SHA256=sha(REPORTS/'SOURCE_INTERFACE_MANIFEST.json')),
        all_protected_Phase13_report_SHA_unchanged=True,historical_checkpoint_audit=dict(path=str(REPORTS/'HISTORICAL_CHECKPOINT_AUDIT.json'),SHA256=sha(REPORTS/'HISTORICAL_CHECKPOINT_AUDIT.json')),
        scientific_failure_is_not_execution_failure=True,original_scheduler_resumed=True,large_files_to_upload=False,heldout='SEALED',Full24=False,official_TEST=False)
    save(REPORTS/'DELIVERY_AUDIT.json',report);print('PHASE14_DELIVERY_AUDIT',report['status'],len(checked),len(missing),len(mismatch),flush=True)
    assert not missing and not mismatch

if __name__=='__main__':main()
