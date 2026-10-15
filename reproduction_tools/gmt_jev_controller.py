"""Current authoritative research status; never launches superseded queues."""
from gmt_jev_common import *
if __name__=='__main__':
 pointer=read(ROOT/'reports/CURRENT_RESEARCH_GOAL.json')
 goal_path=ROOT/pointer['master_goal']
 print(json.dumps(dict(goal=read(goal_path),registry=read(goal_path.parent/'EXPERIMENT_REGISTRY.json')),indent=2,ensure_ascii=False))
