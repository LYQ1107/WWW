"""Reconcile completed additive diagnostics only; finite research watcher."""
import argparse
from jev_recovery_common import *


def inputs():
    names=['QUESTION_TO_ASSOCIATION_DIAGNOSIS.json','DIAGNOSTIC_FUSION_RESULT_INDEX.json',
           'R2_XIV_MECHANISM_GATE.json','R3_XIII_MECHANISM_GATE.json']
    files=[REPORTS/n for n in names if (REPORTS/n).exists()]
    for folder in ['diagnostic_fusion','diagnostic_fusion_native']:
        files+=list((REPORTS/folder).glob('*.json'))
    files+=list(REPORTS.glob('TRAIN_R2_XIV_*_pilot_SEED20261009.json'))
    files+=list((REPORTS/'R2_XIV').glob('ONLINE_*VIDEO16.json'))
    return sorted(files)


def render():
    text=['# Question 到实际身份关联：补充诊断',
          '', '科学状态：`DEVELOPMENT_NOT_ESTABLISHED`。辅助分类提高不计作主目标成功。原 Gate、原训练标签与权重保留。',
          '', '## 当前标签与科学口径', '',
          '实际 load_records 先读取旧 R2 Cross/历史可靠性标签，再用 first-past-certified known_options/positive 替换 Stability。当前 Persistent 是与第一次过去认证的 ID 归属是否一致。两版分歧统计不能直接比较。混合历史可导致旧可靠性=0；当前已经绑定的过去证书保留。未绑定 ID、未知当前人保持 UNKNOWN。证书不代表真实出生 owner 或未来安全。',
          '', '训练的 3601 个局部冲突组按共享人员／ID 历史连接后只有 3 个保守依赖分量。不能把相邻帧和候选边当作独立纠错事件。标签源 SHA 与实际 loader 见 QUESTION_LABEL_SEMANTICS_AUDIT.json。',
          '', '## 复核用户给出的 Pilot 现象', '',
          '| 模型 | Cross 正例正确/总数 | Persistent 正例正确/总数 | 当前排名准确率 |',
          '|---|---:|---:|---:|']
    for arm in ['jev','threshold','mlp','attention']:
        path=REPORTS/f'TRAIN_R2_XIV_{arm}_pilot_SEED20261009.json'
        if not path.exists():continue
        d=read(path)['final'];q=d['questions']
        text.append(f"| {arm} | {q['cross'].get('1_correct',0)}/{q['cross'].get('1_support',0)} | {q['stability'].get('1_correct',0)}/{q['stability'].get('1_support',0)} | {d['rank_pair_accuracy']:.6f} |")
    text+=['','这里的当前排名 targets 包含 0.25 Persistent 偏好；新诊断另报纯 WHO 排名。这些是固定 128 个复用 TRAIN16 状态的相关样本，不是完整 MOT 或独立验证。',
           '', '## A/B/C 的冻结权重诊断', '',
           '同一个 JEV Pilot checkpoint，全部 2067 个真实 TRAIN16 状态与相同合法方案；不写原生状态。屏蔽的是最后融合的隐藏表示，包括 private DEFER 表示，类型向量和权重保留。简单平均使用原训练评分头，属于分布改变的冻结干预，不代表重新训练的平均架构。']
    path=REPORTS/'QUESTION_TO_ASSOCIATION_DIAGNOSIS.json'
    if path.exists():
        d=read(path)
        text+=['','| 干预 | N01 | N10 | 净纠错 | 对完整模型改配行数 | WHO-only 排名 |',
               '|---|---:|---:|---:|---:|---:|']
        for mode,c in d['interventions'].items():
            rank=c.get('WHO_only_rank_correct',0)/max(1,c.get('WHO_only_rank_pairs',0))
            text.append(f"| {mode} | {c.get('N01',0)} | {c.get('N10',0)} | {c['net_certified_corrections']} | {c.get('rows_changed_vs_full',0)} | {rank:.6f} |")
        c=d['interventions']['full']
        text+=['',f"完整模型中两辅助问题均正确、最终身份错误且存在合法正确替代的行数：{c.get('both_correct_final_wrong_with_legal_correct_alternative',0)}。没有合法正确替代的对应行数：{c.get('both_correct_final_wrong_without_legal_correct_alternative',0)}；后者不定义为排名失败。",
               '',f"已认证正确合法身份未进入最终方案的行数：{c.get('legal_certified_correct_absent_from_plans',0)}。UNKNOWN 比较：{c.get('UNKNOWN_comparison',0)}。完整模型正负辅助交叉表与最多 40 个事件的 ID、历史可用性、早期外观余弦、原始/修正分数保存在诊断 JSON。",
               '', '把两个辅助输出置零后最终分数逐位相等：最终评分没有直接读取辅助 Head 输出。这个结果证明直接输出路径缺失，不能证明辅助训练没有通过隐藏表示起作用。屏蔽／平均的差异检验 B；两问题正确但合法纠错仍失败检验 A；覆盖、UNKNOWN、认证与依赖审计检验 C。三种原因可能共同存在，不能仅由一个比例排他归因。']
    else:text+=['','冻结干预正在执行；尚未生成完成结果。']
    text+=['','## 第5/6项：重新训练的最小融合对照', '',
           '同一冻结 R2 JEV 前端和辅助 Head，四个最终融合器：显式辅助概率门控、两个独立最终 Reader、普通 MLP、普通 Attention。新增参数81786/82274/82476/81826，差距小于1%。同样两路表示、两预测概率、原余量和因果早期外观，同批次与损失；Tiny128 后 fresh Pilot1000，再做全部 TRAIN16 原生在线和身份提交审计。',
           '', '这只对比相同已训练 JEV 前端上的最终融合，不代表从头训练的普通控制器架构。原纯 MLP/Attention 主对照继续独立报告。新增融合梯度只有1000步，属于已允许 R2 范围；既有正式4000和原晋级条件均不变，没有第四个大型版本，也没有 DEV 调参。',
           '', '| 融合器 | Tiny | Pilot | 同状态净纠错 | 完整在线 |',
           '|---|---|---|---:|---|']
    for kind in ['explicit_scores','independent_readers','mlp','attention']:
        states=[]
        for phase in ['tiny','pilot']:
            p=REPORTS/'diagnostic_fusion'/f'{kind}_{phase}.json'
            states.append(read(p)['status'] if p.exists() else 'PENDING')
        p=REPORTS/'diagnostic_fusion'/f'{kind}_CERTIFIED.json'
        net=read(p)['net'] if p.exists() else 'PENDING'
        p=REPORTS/'diagnostic_fusion_native'/f'ONLINE_{kind}_pilot_SEED20261009_VIDEO16.json'
        online=read(p)['status'] if p.exists() else 'PENDING'
        text.append(f'| {kind} | {states[0]} | {states[1]} | {net} | {online} |')
    text+=['','## 原 R2/R3 晋级与强基座', '',
           '继续全部预注册 R2 消融、完整 TRAIN16 和既有机制 Gate，再执行强 XIII Fixed 原始及阈值／MLP／Attention／JEV 增量。同状态 N01/N10 与完整 mutated-state 指标分开报告。两 Gate 失败也不反复调 DEV17/18/19，不修改阈值求成功。强基座历史参考不是本次已完成同基座结果。',
           '', 'Stage2、真实 FPS、参数与三种子正式指标以原 finisher 完成文件为准。此报告不会把 pending 值填成0，也不会宣称独立验证或研究突破。']
    for name in ['R2_XIV_MECHANISM_GATE.json','R3_XIII_MECHANISM_GATE.json']:
        if (REPORTS/name).exists():text.append(f"\n{name}：formal_eligible={read(REPORTS/name)['formal_eligible']}。")
    (DOCS/'QUESTION_TO_ASSOCIATION_REPORT.md').write_text('\n'.join(text)+'\n')
    save(REPORTS/'GOAL_ADDENDUM_STATUS.json',dict(execution='ACTIVE',scientific_goal='DEVELOPMENT_NOT_ESTABLISHED',
        auxiliary_accuracy_is_not_success=True,original_gates_changed=False,
        completed_evidence=[ref(p) for p in inputs()],original_queues_preserved=True))


def main(watch):
    prior=None
    while True:
        current=[(str(p),sha(p)) for p in inputs()]
        if current!=prior:
            render()
            from jev_recovery_publish import publish
            publish('Reconcile actual completed additive question-to-association evidence without altering gates')
            prior=current
        if not watch:break
        if (REPORTS/'DIAGNOSTIC_FUSION_RESULT_INDEX.json').exists() and (OUT/'finish/READY.json').exists():break
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');main(p.parse_args().watch)
