"""Standalone scientific figure: bounded offline oracle support, not models."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from jev_phase8_common import *

def main():
    x=json.loads((REPORTS/'NATIVE_CORRECTIVE_ORACLE_AUDIT.json').read_text());out=REPORTS/'figures';out.mkdir(exist_ok=True)
    videos=[12,13,14,16,17,18,19];events=x['events']
    counts=[sum(e['key'][0]==v for e in events)for v in videos]
    direct=[sum(e['key'][0]==v and any(t.startswith('CANDIDATE')for t in e['verified_branches'])for e in events)for v in videos]
    re=[sum(e['key'][0]==v and 'REASSOCIATE'in e['verified_branches']for e in events)for v in videos]
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),gridspec_kw={'width_ratios':[2.3,1]});pos=np.arange(len(videos));w=.25
    axes[0].bar(pos-w,counts,w,color='#b9c0c9',label='Audited representatives')
    axes[0].bar(pos,direct,w,color='#356db5',label='GT-chosen candidate intervention')
    axes[0].bar(pos+w,re,w,color='#e69a3a',label='Native reject + re-solve')
    axes[0].set_xticks(pos,[f'{v:02d}'for v in videos]);axes[0].axvline(3.5,color='#666666',ls=':',lw=1)
    axes[0].set_xlabel('TRAIN12–16  |  Validation17–19');axes[0].set_ylabel('Events: immediate correction + positive H32, birth weight zero')
    axes[0].set_title('Observed successes in fixed representative states');axes[0].legend(frameon=False,fontsize=8)
    verified=[x['train']['verified_events'],x['validation']['verified_events']];gate=[50,20]
    axes[1].bar([0,1],verified,color=['#356db5','#356db5'],width=.5)
    for i,(n,g)in enumerate(zip(verified,gate)):
        axes[1].plot([i-.35,i+.35],[g,g],color='#b93d44',lw=2);axes[1].text(i,n+.7,str(n),ha='center');axes[1].text(i+.35,g,f' gate{g}',fontsize=8,va='center')
    axes[1].set_xticks([0,1],['TRAIN','Validation']);axes[1].set_ylabel('Verified events');axes[1].set_ylim(0,max(verified+gate)+10);axes[1].set_title('Frozen formal data gate')
    for a in axes:a.spines[['right','top']].set_visible(False);a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
    fig.text(.02,.02,'Offline GT chooses interventions; no learned candidate model. Correlated events remain grouped; no heldout or full-video metric claim.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in('png','pdf'):fig.savefig(out/f'PHASE8_BOUNDED_NATIVE_SUPPORT.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    save(out/'FIGURE_MANIFEST.json',{'status':'COMPLETE','source_report_sha256':sha(REPORTS/'NATIVE_CORRECTIVE_ORACLE_AUDIT.json'),
        'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(out.glob('PHASE8_BOUNDED_NATIVE_SUPPORT.*'))],
        'scope':'bounded real-state offline interventions; overlapping windows, not trained-model or full-video outcomes'})
if __name__=='__main__':main()
