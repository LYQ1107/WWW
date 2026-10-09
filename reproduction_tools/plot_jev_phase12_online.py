"""Static scientific plots of every frozen seed, without favorable sequence selection."""
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jev_phase12_common import *

def main():
    protect()
    data=json.loads((REPORTS/'MATCH_VALIDATION_RESULTS.json').read_text());assert data['status']=='COMPLETE'
    rows={r['case']['name']:r for r in data['cases']};seeds=[20261008,20261009,20261010]
    methods=['GMT_OFF','Fixed','CandidateMLP','CandidateDeepSets','CandidateJEV','set_transformer','visual_deepsets','question_plain','full','numerical_only','full_no_risk']
    ablations=['full','no_question_reader','fixed_question','no_option_reader','no_gating','no_shared_state','no_history','no_H32','no_consequence','similarity','full_no_risk']
    # The report/protocol is authoritative for exact model names.
    assert all(f'{n}_s{seeds[0]}' in rows for n in ablations)
    def values(name,metric):
        names=[name] if name in ['GMT_OFF','Fixed'] else [f'{name}_s{s}' for s in seeds]
        return [rows[n]['pooled_metrics']['strict_online'][metric] for n in names]
    out=ROOT/'docs/figures/JEV_PHASE12';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    for filename,names in [('MATCH_ONLINE_COMPARISON',methods),('MATCH_ONLINE_ABLATIONS',ablations)]:
        fig,axes=plt.subplots(1,2,figsize=(12,6.5),sharey=True)
        for ax,metric in zip(axes,['HOTA','AssA']):
            for i,name in enumerate(names):
                v=values(name,metric);mean=statistics.mean(v)
                ax.plot([min(v),max(v)],[i,i],color='#2563eb',lw=2)
                ax.scatter(v,[i]*len(v),s=21,facecolors='white',edgecolors='#2563eb',zorder=3)
                ax.scatter([mean],[i],marker='D',s=28,color='#dc2626',zorder=4)
            ax.axvline(values('GMT_OFF',metric)[0],ls='--',color='#555',label='GMT OFF')
            ax.set_yticks(range(len(names)));ax.set_yticklabels(names);ax.invert_yaxis()
            ax.set_xlabel(f'{metric} (%)');ax.grid(axis='x',alpha=.25);ax.set_title(metric)
        fig.suptitle('Strict online native validation: videos17/18/19\nPooled six camera sequences; circles = all 3 fitted seeds; diamond = mean',fontsize=11)
        fig.tight_layout(rect=[0,0,1,.94])
        for ext in ['png','pdf']:fig.savefig(out/f'{filename}.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    save(REPORTS/'ONLINE_FIGURES_MANIFEST.json',{'status':'COMPLETE','input_SHA256':sha(REPORTS/'MATCH_VALIDATION_RESULTS.json'),'scope':'strict online primary; all3 frozen seeds, range not confidence interval; pooled6 camera sequences, no sequence/seed selection','files_SHA256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(out.glob('MATCH_ONLINE*'))},'models':methods,'ablations':ablations})
if __name__=='__main__':main()
