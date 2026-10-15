"""Standalone scientific figure from completed measurements; no GPU work."""
from gmt_jev_common import *

def main():
 reserve();os.environ['MPLCONFIGDIR']=str(OUT/'figures/mplconfig')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 from matplotlib.patches import Patch
 base_path=REPORTS/'ORIGINAL_GMT_METRICS.json';base=read(base_path)
 baseline_speed=read(REPORTS/'SPEED_original.json');mean=baseline_speed['mean_full_scene_FPS']
 points=[]
 labels={'original':'GMT','threshold':'Threshold','rule':'Rule','rule_cached':'Rule + cache',
  'confidence':'Confidence','mlp_learned':'MLP','attention_learned':'Attention',
  'cic_learned':'CIC','mlp_learned_cached':'MLP + exact cache'}
 offsets={'original':(7,-9),'threshold':(-56,-17),'rule':(3,12),'rule_cached':(-74,-20),
  'confidence':(-65,10),'mlp_learned':(-13,12),'attention_learned':(7,10),
  'cic_learned':(7,8),'mlp_learned_cached':(7,-19)}
 colors={'original':'#666666','threshold':'#56B4E9','rule':'#CC79A7','rule_cached':'#CC79A7',
  'confidence':'#009E73','mlp_learned':'#0072B2','attention_learned':'#D55E00',
  'cic_learned':'#D55E00','mlp_learned_cached':'#0072B2'}
 for kind,label in labels.items():
  speed_path=REPORTS/f'SPEED_{kind}.json'
  if not speed_path.exists():continue
  s=read(speed_path);assert s['status']=='COMPLETE' and len(s['trials'])==3
  assert s['GPU_UUID']==baseline_speed['GPU_UUID']
  raw_kind=kind.replace('_cached','')
  quality_path=base_path if kind=='original' else REPORTS/f'ONLINE_{raw_kind}.json'
  case=read(quality_path);quality=case if kind=='original' else case['primary_raw']
  if kind=='rule_cached':
   assert read(REPORTS/'FEATURE_CACHE_PARITY.json')['status']=='PASS_EXACT_FULL_NATIVE_RULE'
   assert s['stock_prediction_SHA']==read(REPORTS/'SPEED_rule.json')['stock_prediction_SHA']
  if kind=='mlp_learned_cached':
   proof=read(REPORTS/'LEARNED_MLP_FEATURE_CACHE_PARITY.json')
   assert proof['status']=='PASS_EXACT_FULL_NATIVE_MLP_LEARNED' and proof['score_feature_pairs']==2067
   assert s['trained_checkpoint']['SHA256']==case['trained_checkpoint']['SHA256']
   assert s['stock_prediction_SHA']==read(REPORTS/'SPEED_mlp_learned.json')['stock_prediction_SHA']
  m=quality['metrics'];b=base['metrics'];identity=base['identity'] if kind=='original' else case['identity']
  stable=(m['AssA']>=b['AssA'] and quality['CVIDF1']>=base['CVIDF1'] and m['IDSW']<=1.05*b['IDSW']
   and m['Frag']<=1.05*b['Frag'] and identity['identity_inconsistency_duration_camera_frames']<=1.05*base['identity']['identity_inconsistency_duration_camera_frames'])
  trials=[t['full_scene_FPS']/mean for t in s['trials']]
  points.append(dict(kind=kind,label=label,FPS_ratio=s['mean_full_scene_FPS']/mean,
   timing_range=[min(trials),max(trials)],delta_HOTA=m['HOTA']-b['HOTA'],
   measured_identity_side_gates_pass=stable,speed=ref(speed_path),quality=ref(quality_path)))
 fig,ax=plt.subplots(figsize=(10,5.8));fig.subplots_adjust(left=.10,right=.97,bottom=.23,top=.88)
 xmin=min(.65,min(p['timing_range'][0] for p in points)-.025);xmax=max(1.055,max(p['timing_range'][1] for p in points)+.025)
 ymin=min(-.90,min(p['delta_HOTA'] for p in points)-.15);ymax=max(.85,max(p['delta_HOTA'] for p in points)+.15)
 ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='Full-image FPS / GMT (same GPU)',ylabel='HOTA gain over GMT (points)')
 ax.fill_between([.95,xmax],.5,ymax,color='#DDEEDF',alpha=.7,zorder=0)
 ax.axvline(.95,color='#555555',linestyle='--',linewidth=1);ax.axhline(.5,color='#555555',linestyle='--',linewidth=1)
 ax.axhline(0,color='#BBBBBB',linewidth=.8);ax.grid(alpha=.18)
 for p in points:
  x,y=p['FPS_ratio'],p['delta_HOTA'];lo,hi=p['timing_range'];kind=p['kind'];color=colors[kind]
  ax.errorbar(x,y,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='none',ecolor=color,capsize=3,alpha=.75,zorder=2)
  marker='o' if p['measured_identity_side_gates_pass'] else 'X'
  ax.scatter(x,y,s=70,marker=marker,color=color,edgecolors='white',linewidths=.6,zorder=3)
  ax.annotate(p['label'],(x,y),xytext=offsets[kind],textcoords='offset points',fontsize=9,color=color)
 handles=[Line2D([],[],marker='o',color='none',markerfacecolor='#555555',label='Measured identity side gates pass'),
  Line2D([],[],marker='X',color='none',markerfacecolor='#555555',label='A measured identity side gate fails'),
  Patch(facecolor='#DDEEDF',label='HOTA / FPS target region only')]
 ax.legend(handles=handles,loc='upper left',fontsize=8.5,framealpha=.9)
 fig.suptitle('GMT-JEV: native-raw development accuracy / efficiency',fontsize=13)
 fig.text(.10,.10,'TRAIN16 is exposed development; fixed training seed 20261009. Bars show the range of three timing trials, not confidence intervals.',fontsize=8)
 fig.text(.10,.055,'Serial shared-host blocks: timing drift is not isolated. Causal error propagation, training-seed replication and independence remain unproven.',fontsize=8)
 png=DOCS/'GMT_JEV_ACCURACY_EFFICIENCY.png';pdf=DOCS/'GMT_JEV_ACCURACY_EFFICIENCY.pdf'
 fig.savefig(png,dpi=170);fig.savefig(pdf,metadata={'CreationDate':None,'ModDate':None,'Creator':'GMT-JEV measured-artifact renderer'});plt.close(fig)
 assert png.stat().st_size<2**20 and pdf.stat().st_size<2**20
 save(REPORTS/'ACCURACY_EFFICIENCY_FIGURE.json',dict(status='COMPLETED_MEASUREMENTS_ONLY',points=points,
  PNG=ref(png),PDF=ref(pdf),baseline=ref(base_path),baseline_speed=ref(REPORTS/'SPEED_original.json'),
  errorbars='Observed three-trial speed range, not confidence intervals or independent seeds',
  marker_scope='AssA/CVIDF1/IDSW/Frag and retrospective-duration side gates; not causal propagation or scientific success',
  scientific_success=False,independent_validation=False))
 print('TRADEOFF_FIGURE_BUILT',len(points),png.stat().st_size,pdf.stat().st_size,flush=True)

if __name__=='__main__':main()
