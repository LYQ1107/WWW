"""Route legacy automatic entry points without changing historical evidence."""
import json
from pathlib import Path

def current():
 root=Path(__file__).resolve().parents[1];p=root/'reports/CURRENT_RESEARCH_GOAL.json'
 return json.loads(p.read_text()) if p.exists() else None

def redirect_if_superseded(expected):
 d=current()
 if d and d['master_goal']!=expected:
  print(json.dumps(dict(status='SUPERSEDED_CONTROLLER_NO_OLD_JOB_LAUNCHED',current_goal=d),indent=2))
  return True
 return False
