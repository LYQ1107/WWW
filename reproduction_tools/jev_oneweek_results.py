"""Aggregate genuine pooled all-camera metrics; never average per-video HOTA."""
import numpy as np
from jev_oneweek_common import *


def main(quiet=False):
    protect()
    from run_early_pilot_tracking import extract_metrics
    arms=['parent','cosine','fixed_threshold','fixed_fusion','extra_hungarian',
          'linear','state_threshold','threshold','binary','mlp','deepsets','attention',
          'fixed_question','jev','cross_only','stability_only','additive','no_question_reader',
          'XIII_fixed_reference','XV_V3_reference','no_cross_evidence','no_stability_evidence',
          'base_single','base_repeat','base_rule','edge_single','edge_repeat','edge_rule']
    values={};table=[]
    for arm in arms:
        cases=[]
        for seed in SEEDS:
            folder=OUT/'matlab_v1/formal'/arm/f'seed{seed}'/'17_18_19'
            if not (folder/'RESULT.json').exists():continue
            native=read(folder/'RESULT.json')
            evaluation=folder/'pooled_TrackEval/tracking_eval_runtime_state/native/evaluation'
            metrics=extract_metrics(evaluation)
            metrics.update(CVIDF1=native['CVIDF1'],CVMA=native['CVMA'])
            cases.append(dict(seed=seed,metrics=metrics,official=ref(folder/'RESULT.json'),
                              pooled_TrackEval=ref(evaluation/'metrics.json')))
        summary={k:dict(mean=float(np.mean([v['metrics'][k] for v in cases])),
                       std_seed=float(np.std([v['metrics'][k] for v in cases],ddof=1)) if len(cases)>1 else None)
                 for k in ['HOTA','DetA','AssA','IDF1','MOTA','IDSW','Frag','CVIDF1','CVMA']} if cases else None
        values[arm]=dict(status='COMPLETE_THREE_SEEDS' if len(cases)==3 else 'PARTIAL' if cases else 'NOT_RUN',
                         cases=cases,summary=summary,missing_seeds=[s for s in SEEDS if s not in [c['seed'] for c in cases]])
        if summary:table.append('| %s | %d/3 | %.3f | %.3f | %.3f | %.3f | %.1f |'%(
             arm,len(cases),summary['HOTA']['mean'],summary['AssA']['mean'],summary['CVIDF1']['mean'],summary['CVMA']['mean'],summary['IDSW']['mean']))
    save(REPORTS/'FORMAL_CURRENT_RESULTS.json',dict(status='PARTIAL_UNTIL_ALL_FROZEN_JOBS_DONE',arms=values,
         old_different_phase_results_not_spliced=True,three_seeds_not_statistical_significance=True,
         input_frontend='Common frozen Stage1 VFCE1024; inherited same XIV common initializer for incremental architectures',
         original_GMT_separate_reference=read(REPORTS/'MAIN_RESULTS.json')['rows']['M0']))
    doc='# 当前真实正式结果（未完成项保留 NOT_RUN）\n\n每个种子使用完整DEV17/18/19，先真实合并六个相机，再取种子均值。部分种子的均值不是最终主表，也不能因优劣选种子。\n\n| 方法 | 完成种子 | HOTA | AssA | CVIDF1 | CVMA | IDSW |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'+'\n'.join(table)
    doc+='\n\n原始完整GMT HOTA78.110为不同完整前端的冻结参考，已逐字段复现；不能混入此表作单一关联模块因果差值。当前独立JEV贡献 NOT_ESTABLISHED。\n'
    (DOCS/'CURRENT_FORMAL_RESULTS.md').write_text(doc)
    if not quiet:print(doc,flush=True)


if __name__=='__main__':main()
