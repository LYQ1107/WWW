"""Export scientific learning curves from every frozen fit."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jev_phase12_common import REPORTS,ROOT

def main():
    data=json.loads((REPORTS/'MATCH_TRAINING_CURVES.json').read_text())['curves'];out=ROOT/'docs/figures/JEV_PHASE12';out.mkdir(parents=True,exist_ok=True)
    groups=[['full','set_transformer','visual_deepsets','question_plain'],['full','CandidateMLP','CandidateDeepSets','CandidateJEV','numerical_only'],['full','no_question_reader','fixed_question','no_option_reader','no_gating'],['full','no_shared_state','no_history','no_H32','no_consequence','similarity']]
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True,sharey=True)
    for ax,group in zip(axes.flat,groups):
        for name in group:
            curves=[r for r in data if r['model']==name];ys=[[i['validation_correct']*100 for i in c['curve']] for c in curves]
            import numpy as np
            y=np.array(ys);x=[i['update'] for i in curves[0]['curve']];ax.plot(x,y.mean(0),label=name);ax.fill_between(x,y.min(0),y.max(0),alpha=.10)
        ax.set_xlabel('Optimizer updates');ax.set_ylabel('Certified Correct lower bound (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Phase XII frozen native v3 development validation: mean and range of 3 seeds\nPartial labels; UNKNOWN is not a negative. Best checkpoint chosen by known NLL + executed ranking loss.')
    fig.tight_layout();fig.savefig(out/'MATCH_LEARNING_CURVES.png',dpi=150);fig.savefig(out/'MATCH_LEARNING_CURVES.pdf');plt.close(fig)
if __name__=='__main__':main()
