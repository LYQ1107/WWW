"""Finite compact publication refresh; no neural job restarts or metric fabrication."""
from jev_recovery_common import *
from jev_recovery_publish import publish
def main():
 last=None
 while True:
  files=list(REPORTS.glob('TRAIN_R*.json'))+list(REPORTS.glob('R*/ONLINE_*.json'))+list(REPORTS.glob('R*/MATLAB_*.json'))+list(REPORTS.glob('R*/EFFICIENCY_*.json'))
  digest=hashlib.sha256(''.join(str(p)+sha(p) for p in sorted(files)).encode()).hexdigest()
  if digest!=last:
   publish('Refresh exact completed recovery evidence and active master status');last=digest
  if (OUT/'finish/READY.json').exists():return
  time.sleep(30)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;failure('compact_publication_refresh',traceback.format_exc());raise
