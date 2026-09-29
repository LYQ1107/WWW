import json,os,subprocess
from pathlib import Path
os.umask(0o077)
private=Path.home()/'.local/state/gmt-baidu-auth'
c=json.loads((private/'cookies.json').read_text())
b='/data3/liuyeqiang/GMT_VisionTrack_repro/tools/baidupcs-direct'
# Official fallback for unavailable Tieba account-profile lookup. This alias is
# local metadata only; authenticated quota/list requests still validate access.
p=subprocess.run([b,'config','set','-force_login_username=GMT_QR_SESSION'],capture_output=True,timeout=10)
cmd='login -cookies="'+'; '.join(k+'='+v for k,v in c.items())+';"\nquit\n'
p=subprocess.run([b],input=cmd,text=True,capture_output=True,timeout=30)
print('CLI login success marker:', '登录成功' in p.stdout or '登陆成功' in p.stdout)
config=Path.home()/'.config/BaiduPCS-Go/pcs_config.json';config.chmod(0o600)
for arg in [['who'],['quota'],['ls','/']]:
 try:
  p=subprocess.run([b]+arg,text=True,capture_output=True,timeout=25)
  if arg[0]=='ls':
   print('ls finished; authentication error:',any(t in p.stdout for t in ['未登录','登录失败','获取文件列表失败']));print('output lines:',len(p.stdout.splitlines()))
  else: print(p.stdout)
 except subprocess.TimeoutExpired:print(arg[0], 'timeout')
