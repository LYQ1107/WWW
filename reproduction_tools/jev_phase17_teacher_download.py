"""Pin and stream only necessary licensed teacher files, outside frozen GMT."""
import os,json,hashlib,urllib.request,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
OUT=Path('/home/liuyeqiang/WWW_jev_phase17_runtime/20261011_v1');repo=Path(__file__).resolve().parents[1]
proxies=urllib.request.ProxyHandler({'https':'http://127.0.0.1:17891','http':'http://127.0.0.1:17891'});opener=urllib.request.build_opener(proxies)
def get(url):
 with opener.open(url,timeout=90) as r:return r.read()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 owner=json.loads((repo/'reports/JEV_PHASE17/OWNER_IDENTIFIABILITY.json').read_text());assert owner['teacher_group_count']>0
 output=[]
 for model in ['Qwen/Qwen3-VL-4B-Instruct','guanxuyu/visual-jev-4b-answer-sft']:
  meta=json.loads(get('https://huggingface.co/api/models/'+model+'?blobs=true'));revision=meta['sha'];folder=OUT/'teacher_models'/model.replace('/','--');folder.mkdir(parents=True,exist_ok=True);(folder/'MODEL_METADATA.json').write_text(json.dumps(meta,indent=2))
  chosen=[]
  for f in meta['siblings']:
   name=f['rfilename']
   if name in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer.json','tokenizer_config.json','chat_template.json','chat_template.jinja','vocab.json','merges.txt','model.safetensors.index.json','adapter_config.json','README.md','LICENSE'] or name.endswith('.safetensors'):chosen.append(f)
  total=sum(f.get('size',f.get('lfs',{}).get('size',0)) for f in chosen);assert total<=12*2**30
  def download(f):
   p=folder/f['rfilename'];p.parent.mkdir(parents=True,exist_ok=True);expected=f.get('lfs',{}).get('sha256');size=f.get('size',f.get('lfs',{}).get('size'))
   if not p.exists():
    temp=p.with_name(p.name+'.partial')
    with opener.open('https://huggingface.co/'+model+'/resolve/'+revision+'/'+f['rfilename']+'?download=true',timeout=180) as r,temp.open('wb') as stream:
     while True:
      b=r.read(4*1024*1024)
      if not b:break
      stream.write(b)
    if size:assert temp.stat().st_size==size,(p,size,temp.stat().st_size)
    temp.replace(p)
   digest=sha(p)
   if expected:assert digest==expected,p
   print('TEACHER_FILE',model,p.name,p.stat().st_size,flush=True);return dict(path=str(p),SHA256=digest,bytes=p.stat().st_size)
  with ThreadPoolExecutor(max_workers=3) as pool:files=list(pool.map(download,chosen))
  output.append(dict(model=model,revision=revision,license=meta.get('cardData',{}).get('license'),files=files,folder=str(folder)))
  (OUT/'teacher_models/MANIFEST.json').write_text(json.dumps(output,indent=2)+'\n')
 print('TEACHER_DOWNLOAD_COMPLETE',flush=True)
if __name__=='__main__':main()
