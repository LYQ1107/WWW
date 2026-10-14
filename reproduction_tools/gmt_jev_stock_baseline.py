"""Evaluate actual unchanged stock GMT predictions retained after full-image timing."""
import gzip
from gmt_jev_common import *
from gmt_jev_evaluate import evaluate,duration

def main():
 speed=read(REPORTS/'SPEED_original.json');assert speed['status']=='COMPLETE'
 p=speed['stock_predictions'];assert sha(p['path'])==p['SHA256']
 with gzip.open(p['path'],'rt') as f:pred=json.load(f)
 result=evaluate(pred,OUT/'metrics/GMT_stock')
 result.update(status='COMPLETE_NATIVE_STOCK_ORIGINAL_GMT',predictions=p,
  convention='UNCHANGED_NATIVE_MIN_TRACK_LEN_50_AND_POSTPROCESS',
  identity=duration(pred),source=binding(),independent_validation=False,
  full_image_FPS=2*speed['mean_full_scene_FPS'],full_scene_FPS=speed['mean_full_scene_FPS'],primary_evaluation=False)
 save(REPORTS/'ORIGINAL_GMT_STOCK_METRICS.json',result)
 print('GMT_STOCK_BASELINE',result['metrics'],result['CVIDF1'],flush=True)

if __name__=='__main__':main()
