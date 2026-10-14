"""Current authoritative research status; never launches superseded queues."""
from gmt_jev_common import *
if __name__=='__main__':
 print(json.dumps(dict(goal=read(REPORTS/'MASTER_GOAL.json'),registry=read(REPORTS/'EXPERIMENT_REGISTRY.json')),indent=2,ensure_ascii=False))
