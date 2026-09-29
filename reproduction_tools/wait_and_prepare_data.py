from pathlib import Path
import json,time,subprocess,sys
r=Path('/data3/liuyeqiang/GMT_VisionTrack_repro')
while True:
 state=json.loads((r/'manifests/archive_intake_state.json').read_text())
 if state['status']=='complete_pending_dataset_integrity':break
 p=Path('/proc')/str(state['pid'])/'stat'
 if state['status']=='failed' or not p.exists() or p.read_text().split()[2]=='Z':
  raise RuntimeError('Archive intake not successful; inspect logs before data preparation')
 time.sleep(15)
subprocess.run([sys.executable,str(r/'tools/prepare_visiontrack.py')],check=True)
