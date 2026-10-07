"""Map native bank-local rows to immutable full-frame cache rows by box.

Native candidate values and decisions are preserved. No GT is read and no
nearest-box fallback is allowed: a native box must match one unique cache box.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT, ROOT / 'third_party/CenterNet2']:sys.path.insert(0,str(path))
import torch
from gtr.modeling.jev_perception_cache import FrozenPerceptionCache


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--native-trace',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise RuntimeError('refusing to overwrite canonical trace')
    cache=FrozenPerceptionCache(args.cache);payloads=OrderedDict();mapped=[];events=[]
    for line in args.native_trace.open():
        event=json.loads(line);c=event.get('context',{})
        if event.get('question')=='REACTIVATION_DECISION':
            key=(int(c['video_id']),int(c['frame']),int(c['view']))
            if key not in payloads:
                payloads[key]=cache.load(*key)
                if len(payloads)>64:payloads.popitem(last=False)
            boxes=torch.as_tensor(payloads[key]['pred_boxes'],dtype=torch.float32)
            box=torch.as_tensor(c['bbox_xyxy'],dtype=torch.float32)
            errors=(boxes-box).abs().amax(dim=1)
            matches=torch.nonzero(errors<=2e-5).flatten().tolist()
            if len(matches)!=1:raise RuntimeError(f'non-unique exact native/cache box mapping {key}: {matches}')
            local=int(c['detection_index']);global_row=int(matches[0])
            c['native_bank_detection_index']=local;c['detection_index']=global_row
            mapped.append({'key':key,'native_bank_row':local,'cache_row':global_row,'max_box_error':float(errors[global_row])})
        events.append(event)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w') as f:
        for event in events:f.write(json.dumps(event,sort_keys=True)+'\n')
    report={'status':'PASS','native_trace':str(args.native_trace),'native_trace_sha256':sha(args.native_trace),
            'canonical_trace':str(args.output),'canonical_trace_sha256':sha(args.output),
            'cache_index_sha256':sha(args.cache/'index.jsonl'),'records':len(events),'mapped_reactivation_events':len(mapped),
            'native_candidate_values_and_actions_unchanged':True,'mapping_rule':'unique exact box within 2e-5',
            'changed_rows':sum(m['native_bank_row']!=m['cache_row'] for m in mapped),'mapping':mapped}
    Path(str(args.output)+'.row_mapping.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='mapping'}))


if __name__=='__main__':main()
