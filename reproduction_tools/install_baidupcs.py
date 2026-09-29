import os, subprocess, json, zipfile, hashlib
from pathlib import Path
r=Path('/data3/liuyeqiang/GMT_VisionTrack_repro')
j=json.loads((r/'manifests/baidupcs_release.json').read_text())
a=next(a for a in j['assets'] if a['name'].endswith('linux-amd64.zip'))
archive=r/'downloads'/a['name']
args=['curl','-fL','--connect-timeout','15','--max-time','180','--retry','1','-o',str(archive),a['browser_download_url']]
env={k:v for k,v in os.environ.items() if k.lower() not in ['http_proxy','https_proxy','all_proxy']}
# github.com direct connection already failed in the mandatory network test.
with (r/'logs/network_direct_test.log').open('a') as f:f.write('github.com release asset DIRECT_FAILED -> PROXY_FALLBACK (prior github.com direct timeout)\n')
p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(r/'logs/install_baidupcs.log').write_text(p.stdout)
p.check_returncode()
dest=r/'tools/BaiduPCS-Go';dest.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  if Path(name).name=='BaiduPCS-Go':
   b=dest/'BaiduPCS-Go';b.write_bytes(z.read(name));b.chmod(0o755)
   break
 else:raise RuntimeError('binary missing')
with (r/'manifests/tools.sha256').open('w') as f:
 for path in [archive,b]:f.write(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path)+'\n')
p=subprocess.run([str(b),'version'],env=env,capture_output=True,text=True)
print(p.stdout,p.stderr);p.check_returncode()
with (r/'logs/install_baidupcs.log').open('a') as f:f.write(p.stdout+p.stderr)
