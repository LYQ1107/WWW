"""Direct frozen input hash recheck, separate from scientific completion."""
import argparse
import re
from common import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--detail-output',type=Path,required=True)
    args=parser.parse_args()
    if args.detail_output.exists():raise FileExistsError(args.detail_output)
    names=['trackvggt_phase0_audit_v1.json','trackvggt_full_train_completion_audit_v1.json',
        'trackvggt_full_train_event_inventory_v1.json','trackvggt_v3_native_capture_complete_v1.json',
        'trackvggt_v3_tokens_complete_v1.json','trackvggt_crop_inputs_complete_v1.json',
        'trackvggt_real_cycle_features_v1.json']
    references={};reports={}
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ['source_artifacts','feature_manifests','token_manifests'] and isinstance(item,dict):
                    for name,digest in item.items():
                        if isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) and Path(name).is_absolute():
                            if name in references and references[name]!=digest:raise ValueError('conflicting frozen input hashes: '+name)
                            references[name]=digest
                visit(item)
        elif isinstance(value,list):
            for row in value:visit(row)
    for name in names:
        path=ROOT/'results'/name;reports[str(path)]=sha256(path);visit(json.loads(path.read_text()))
    checked=[];mismatches=[]
    for path,digest in sorted(references.items()):
        actual=sha256(path) if Path(path).is_file() else None
        row=dict(path=path,expected_sha256=digest,actual_sha256=actual)
        checked.append(row)
        if digest!=actual:mismatches.append(row)
    proof=dict(status='PASS_DIRECT_FROZEN_INPUT_HASH_RECHECK' if not mismatches else 'FAIL_DIRECT_FROZEN_INPUT_HASH_RECHECK',
        reports=reports,direct_inputs=len(checked),mismatches=mismatches,
        scope='Recheck direct absolute source_artifacts/feature_manifests/token_manifests references from seven frozen phase/input audits. Not recursive validation of all internal files, repeated tracker inference, or a scientific-success/completion audit.',
        provenance=provenance())
    save_json(args.detail_output,dict(proof,checked=checked))
    proof.update(detail_report=str(args.detail_output),detail_report_sha256=sha256(args.detail_output))
    save_json(ROOT/'results/trackvggt_frozen_direct_input_hash_recheck_v1.json',proof)
    print(json.dumps(dict(status=proof['status'],inputs=len(checked),mismatches=mismatches)))
    if mismatches:raise SystemExit(1)


if __name__=='__main__':main()
