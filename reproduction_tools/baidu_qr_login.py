#!/usr/bin/env python3
"""Direct official Baidu QR session; credentials stay outside project/logs."""
import json, os, time
from pathlib import Path
from urllib.parse import urlparse
import requests
os.umask(0o077)
private = Path.home()/'.local/state/gmt-baidu-auth'
private.mkdir(parents=True, exist_ok=True);private.chmod(0o700)
public = Path(os.environ.get('GMT_BAIDU_QR_PATH', '/data1/liuyeqiang/WWW/outputs/baidu-login-qr.png'))
state = private/'status.json'
def status(value):
 state.write_text(json.dumps({'status':value,'updated':time.time(),'pid':os.getpid()}))
 print(value,flush=True)
def get(session,url,**kwargs):
 host=urlparse(url).hostname or ''
 if not (host=='baidu.com' or host.endswith('.baidu.com')):raise ValueError('Non-Baidu host rejected')
 r=session.get(url,timeout=30,**kwargs);r.raise_for_status();return r
try:
 s=requests.Session();s.trust_env=False
 s.headers['User-Agent']='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36'
 j=get(s,'https://passport.baidu.com/v2/api/getqrcode',params={'lp':'pc','tpl':'netdisk','apiver':'v3'}).json()
 if j.get('errno')!=0:raise ValueError('QR request rejected')
 url=j['imgurl'];url=url if url.startswith('https://') else 'https://'+url.removeprefix('http://').lstrip('/')
 public.write_bytes(get(s,url).content)
 status('waiting_for_scan')
 deadline=time.monotonic()+300
 while time.monotonic()<deadline:
  try:
   raw=get(s,'https://passport.baidu.com/channel/unicast',params={'channel_id':j['sign'],'callback':''}).text
   data=json.loads(raw.strip().strip('();'))
   channel=data.get('channel_v')
   if channel:
    v=json.loads(channel) if isinstance(channel,str) else channel
    if v.get('status')==0 and v.get('v'):
     get(s,'https://passport.baidu.com/v3/login/main/qrbdusslogin',params={'bduss':v['v'],'tpl':'netdisk'},allow_redirects=False)
     cookies={c.name:c.value for c in s.cookies}
     if not cookies.get('BDUSS'):
      status('confirmed_but_cookie_exchange_failed');break
     (private/'cookies.json').write_text(json.dumps(cookies))
     status('authenticated_pending_cli_import');break
    status('scanned_waiting_confirmation')
  except requests.exceptions.RequestException:
   pass
  time.sleep(2)
 else:status('expired')
except Exception as e:
 status('error_'+type(e).__name__)
