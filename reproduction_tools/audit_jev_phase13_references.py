"""Read exact source blobs and license, bind callable signatures and mechanisms."""
import ast,re
from jev_phase13_common import *
FILES=[['code/vdm/models/vdm_model.py','code/vdm/training/train.py'],['valen/modeling/dual_encoder/model.py','valen/modeling/dual_encoder/layers.py','valen/training/rlcd.py'],['mso/head.py','mso/infer.py'],['qev/engine.py','qev/mm_engine.py','train/mm.py'],['scripts/train_toy_decisions.py'],['src/vision_jev/model.py'],['vev/model.py'],['src/openjev/vision/public_model.py','src/openjev/vision/model.py'],['model/s1/model.py'],['src/vision_decision/scoring.py','src/vision_decision/smolvlm_backend.py'],['cameltrack/camel.py','cameltrack/architecture/gaffe.py','cameltrack/architecture/temporal_encoder.py'],['models/motip/id_decoder.py','models/runtime_tracker.py'],['models/query_updater.py','models/memotr.py'],['3. Tracker/trackers/tracker.py'],['src/models/hiclnet.py','src/tracker/hicl_tracker.py'],['tracker/bot_sort.py'],['yolox/tracker/byte_tracker.py'],['trackers/ocsort_tracker/ocsort.py'],['qdtrack/models/trackers/quasi_dense_embed_tracker.py','qdtrack/models/roi_heads/track_heads/quasi_dense_embed_head.py'],['trackers/ocsort_tracker/ocsort.py']]
def blob(root,path):return subprocess.check_output(['git','show','HEAD:'+path],cwd=root,text=True)
def main():
    pins=json.loads((OUT/'REFERENCE_PINS.json').read_text());records=[]
    for index,r in enumerate(pins['repositories']):
        if r['status']=='UNAVAILABLE':records.append(r);continue
        root=Path(r['checkout']);evidence=root/'audited_sources';sources=[]
        for path in FILES[index]:
            try:s=blob(root,path)
            except subprocess.CalledProcessError:sources.append({'path':path,'status':'ABSENT_DO_NOT_CLAIM_AUDITED'});continue
            p=evidence/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
            tree=ast.parse(s);functions=[{'name':n.name,'line':n.lineno,'end':n.end_lineno,'args':[a.arg for a in n.args.args]} for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
            sources.append({'path':path,'SHA256':sha(p),'functions':functions,'status':'READ_SOURCE_BLOB','URL':r['URL']+'/blob/'+r['commit']+'/'+path})
        license_path=r['license_paths'][0] if r['license_paths'] else None;license_text=blob(root,license_path) if license_path else '';license_file=evidence/'LICENSE_CAPTURE';license_file.parent.mkdir(parents=True,exist_ok=True);license_file.write_text(license_text)
        label='Apache-2.0' if 'Apache License' in license_text else 'MIT' if 'Permission is hereby granted' in license_text else 'AGPL' if 'AFFERO' in license_text.upper() else 'GPL' if 'GNU GENERAL PUBLIC LICENSE' in license_text else 'CUSTOM_READ_FULL_LICENSE' if license_text else 'NO_LICENSE_DO_NOT_COPY'
        records.append({**{k:v for k,v in r.items() if k!='license_paths'},'status':'SOURCE_READ_PENDING_MECHANISM_NOTES','sources':sources,'license':{'path':license_path,'SHA256':sha(license_file),'label':label},'no_import_or_vendor_copy':True})
        print('SOURCE_READ',r['repository'],label,[(s['path'],len(s.get('functions',[]))) for s in sources],flush=True)
    save(OUT/'EXTERNAL_SOURCE_READS.json',{'status':'SOURCE_BLOBS_READ','repositories':records})
if __name__=='__main__':main()
