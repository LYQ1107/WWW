"""Exportable scientific figure; pooled comparisons without invented CIs."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from jev_phase7_common import *

p=json.loads((REPORTS/'PHASE7_REASSOCIATION_ABLATION.json').read_text())['pooled_metrics']
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(12.8,4.6),gridspec_kw={'width_ratios':[1,1.5]})
colors=['#0072B2','#D55E00'];methods=['FIXED','BINARY','B2','MLP_LN'];labels=['Fixed rule','Binary JEV','Frozen B2','Ordinary LN MLP'];x=np.arange(len(methods));width=.34
for offset,metric,color in zip([-.5,.5],['HOTA','AssA'],colors):
 values=[p[m][metric]-p['GMT'][metric]for m in methods]
 bars=axes[0].bar(x+offset*width,values,width,color=color,label=metric)
 axes[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
axes[0].set_xticks(x,labels,rotation=15,ha='right');axes[0].set_ylim(0,1.02);axes[0].set_ylabel('True pooled gain over GMT (points)')
axes[0].set_title('(a) Raw primary controls');axes[0].legend(frameon=False);axes[0].axhline(0,color='.3',lw=.8)
comparisons=[('B2','FIXED','Fixed / raw'),('B2','MLP_LN','LN MLP / raw'),('B2_BUDGET','FIXED_BUDGET','Fixed / equal R budget'),
             ('B2_BUDGET','MLP_LN_BUDGET','LN MLP / equal R budget'),('B2','B2_VALIDATION_ONLY','Global off / raw'),('B2','BINARY','Binary / raw')]
y=np.arange(len(comparisons))
for offset,metric,color in zip([-.5,.5],['HOTA','AssA'],colors):
 values=[p[a][metric]-p[b][metric]for a,b,label in comparisons]
 axes[1].barh(y+offset*width,values,width,color=color,label=metric)
 for row,value in zip(y+offset*width,values):axes[1].text(value+(0.003 if value>=0 else -.003),row,f'{value:+.4f}',va='center',ha='left'if value>=0 else 'right',fontsize=8)
axes[1].set_yticks(y,[v[2]for v in comparisons]);axes[1].invert_yaxis();axes[1].axvline(0,color='.3',lw=.8)
axes[1].axvline(.1,color=colors[0],ls=':',lw=1);axes[1].axvline(.2,color=colors[1],ls=':',lw=1)
axes[1].set_xlim(-.155,.23);axes[1].set_xlabel('B2 minus comparator (points)');axes[1].set_title('(b) Same solver; dotted lines = preregistered margins')
for ax in axes:
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y'if ax==axes[0]else'x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Phase VII reassociation attribution — TRAIN controller-heldouts 09 / 10 / 11',fontsize=12)
fig.text(.02,.01,'One pooled TrackEval over six camera sequences. Three videos, one controller seed; no statistical-significance claim. Complete 23-condition results include failed dynamic/raw MLP controls.',fontsize=8)
fig.tight_layout(rect=[0,.055,1,.94]);dest=REPORTS/'figures';dest.mkdir(exist_ok=True)
for extension in ('png','pdf'):fig.savefig(dest/f'PHASE7_REASSOCIATION_ATTRIBUTION.{extension}',dpi=220,bbox_inches='tight')
save(dest/'FIGURE_MANIFEST.json',{'source_metrics_sha256':sha(REPORTS/'PHASE7_REASSOCIATION_ABLATION.json'),
    'files':{p.name:sha(p)for p in dest.glob('PHASE7_REASSOCIATION_ATTRIBUTION.*')},'no_invented_confidence_intervals':True,'binding':binding()})
print(json.dumps({'status':'EXPORTED','directory':str(dest)}))
