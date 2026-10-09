"""Readable report and static scientific figures from finalized measured evidence."""
import collections
import numpy as np
from jev_phase14_common import *

def total(rows):
    c=collections.Counter()
    for r in rows:c.update(r)
    return dict(c)
def main():
    protect();final=read(REPORTS/'FINAL_GO_NO_GO.json');assert final['execution_deliveries']=='COMPLETE'
    online=read(REPORTS/'ONLINE_VALIDATION.json');official=read(REPORTS/'OFFICIAL_MATLAB_RESULTS.json');fair=read(REPORTS/'FAIR_BASELINE_RESULTS.json');stability=read(REPORTS/'ON_POLICY_STABILITY.json')
    external=read(REPORTS/'EXTERNAL_GENERALIZATION_RESULTS.json');lat=read(REPORTS/'LATENCY_ATTRIBUTION.json');paired=read(REPORTS/'PAIRED_NATIVE_FUTURE_RESULTS.json');formal=online['seed_summary']['formal']
    variants=['fixed_question','multi_question','set_transformer'];labels=['Fixed Question','Multi-question JEV','Set Transformer']
    def mean(phase,v,key):return online['seed_summary'][phase][v][key]['mean']
    def devcounts(phase,v):return total([r['taxonomy_counts'] for r in online['cases'] if r['phase']==phase and r['variant']==v])
    old=read(REPORTS/'ERROR_ATTRIBUTION.json')['aggregate_counts']
    lines=['# Phase XIV 完整科研报告','',
        '本轮执行完成，科学结论为 **NO_GO**。真实 WHO / 身份可用性 / 观测历史可信度能够获得独立监督并被学会，但没有证明多问题 JEV 在身份稳定性上优于同证据普通模型。新的20k方案没有改善冻结 Phase XIII Fixed Question；额外4k也不能自动归因于架构。GTA-free 原生状态与完整视频缓存等价性通过；实时门槛失败，绝对无预训练暴露的全系统泛化条件仍未满足。','',
        '冻结起点 `2881fbdbca6501591470f9814d491a7aa8c1fb17`；实验分支 `jev/www-jev-phase14-reliable-identity-decisions-20261010`。原 Phase XIII报告和323份历史 checkpoint 审计保留。训练只使用12/13/14/16，开发评价17/18/19，20/21/22保持封存。未启动Full24或官方TEST。','',
        '完成矩阵：24个等证据1000update Pilot（另外24个修正前Pilot完整保留）、9个新20k模型、27个同起点追加4k模型、108次开发集完整原生视频、9次完整视频无缓存对照、36个六相机池化官方MATLAB评价、144次TRAIN前256帧真实自身状态审计、468个同前缀16帧未来分支，以及19个冻结外部控制器。所有正式模型使用LAST，三个种子20261008/9/10；没有根据开发结果改门槛或调参。','',
        '以下开发指标每个种子先真实池化全部六个相机，再对三个种子取均值；不是视频指标平均。HOTA/AssA/IDF1/MOTA为百分数。严格原始在线预测是主结果，未来短轨过滤后的canonical结果仅在JSON中单列。','',
        '| 20k 方法 | HOTA | AssA | IDF1 | IDSW | MOTA | 官方 CVIDF1 | 官方 CVMA |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for v,l in zip(variants,labels):
        m=formal[v];o=official['seed_summary']['formal'][v];lines.append('|'+l+'|'+ '|'.join(f'{m[k]["mean"]:.3f}' for k in ['HOTA','AssA','IDF1','IDSW','MOTA'])+'|'+f'{o["CVIDF1"]["mean"]:.3f}|{o["CVMA"]["mean"]:.3f}|')
    lines+=['','Phase XIII固定事实：Original GMT独立完整系统HOTA78.110；同感知Cosine69.085/IDSW82，Full20k75.056/118.667，Fixed20k75.637/105，Set20k74.836，Full24k75.635/143.667。Original GMT使用独立原Stage2前端流程，不能算本轮同输入控制。旧20k与旧24k预算区别保留。','',
        '同监督比较中，Multi-question相对Fixed的HOTA差值仅 '+f'{formal["multi_question"]["HOTA"]["mean"]-formal["fixed_question"]["HOTA"]["mean"]:+.3f}'+ '，且有种子退化；IDSW明显更差。20k三个种子的Multi-question IDSW分别为 '+', '.join(str(int(x)) for x in formal['multi_question']['IDSW']['seeds'])+'。保留全部种子，未去除不利种子。三种子只能支持探索性冻结效应门槛检验，不能声称统计显著性。','',
        '| 24k 对照 | 方法 | HOTA | IDSW | 新跨GT混合 | 可认证split出生 | false birth |','|---|---|---:|---:|---:|---:|---:|']
    for phase,name in [('onpolicy','新目标＋自身状态'),('oldloss_onpolicy','旧目标＋同自身状态'),('offpolicy','新目标＋原数据')]:
        for v,l in zip(variants,labels):
            c=devcounts(phase,v);lines.append(f'|{name}|{l}|{mean(phase,v,"HOTA"):.3f}|{mean(phase,v,"IDSW"):.3f}|{c.get("FALSE_MERGE",0)/3:.3f}|{c.get("FALSE_SPLIT",0)/3:.3f}|{c.get("FALSE_BIRTH",0)/3:.3f}|')
    lines+=['','每个24k分支从同一个20k checkpoint出发，均为4000次额外更新、同batch预算、新AdamW；三个分支部署时使用相同availability映射。旧目标保留旧CE/结构/Brier，其他分支使用新的typed损失。自身状态按身份与64帧块平衡，缺失类别如实记零。TRAIN native风险与reserved输入探针分别记录在[ON_POLICY_STABILITY](../reports/JEV_PHASE14/ON_POLICY_STABILITY.json)，不能把探针正确率当作完整视频稳定性。','',
        '| 外部固定 C1/C2 前320样本 | HOTA | AssA | IDF1 | IDSW |','|---|---:|---:|---:|---:|']
    for name,m in external['seed_summary'].items():lines.append('|'+name+'|'+'|'.join(f'{m[k]["mean"]:.3f}' for k in ['HOTA','AssA','IDF1','IDSW'])+'|')
    lines+=['','外部数据是[EPFL WILDTRACK](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)真实图像与持久personID。只下载冻结C1/C2和对应320个JSON，真实采样率2FPS、160秒；沿用VisionTrack关联超参数，不重采样、不结果后调整。按[官方可见性代码](https://github.com/Chavdarova/WILDTRACK-toolkit/blob/master/annotations_viewer.py)排除-1哨兵框，原始投影框为主、裁剪框敏感性另报。该自定义双相机前缀不是官方七相机benchmark。共享JSON含其他五视角投影，资格审计读过该schema；其他相机图像与后续80个JSON未读，其他视角投影未进入训练/选择。原始过宽“不读取所有其他视角”的标记已由EXTERNAL_DATA_SCOPE_AUDIT明确纠正。','',
        '已知VisionTrack训练清单不含WILDTRACK，但现有通用初始化确切训练集来源未证实。`CH_FPN_1x`文件名不足以证明CrowdHuman/COCO来源；原文件与key-adapted文件模型张量multiset相同。仅能报告已知训练清单下外部场景迁移。Clean Stage1没有合格初始化来源、独立配置与预算，按条件Gate不启动。原前端已暴露全部24个TRAIN；本轮开发数据只对关联控制器未见，不能冒充全系统独立验证。','',
        '| 方法 | 真实场景FPS（3次） | policy-only p95/ms | 历史构建p95/ms（3次） | 总Stage2 p95/ms（3次） |','|---|---|---:|---|---|']
    for x in lat['cases']:
        trials=x['real_image_trials'];lines.append('|'+x['variant']+'|'+', '.join(f'{t["full_sceneFPS"]:.3f}' for t in trials)+f'|{x["policy_only_ms"]["p95"]:.3f}|'+', '.join(f'{t["history_build_ms"]["p95"]:.3f}' for t in trials)+'|'+', '.join(f'{t["total_Stage2_ms"]["p95"]:.3f}' for t in trials)+'|')
    lines+=['','GPU9 V100同图像、同前端、同线程，8帧warmup、三次完整image-to-tracks重复；camera-payloadFPS为sceneFPS的两倍。剖析/GT/日志不进入主FPS计时。独立分项测量中，单相机图像读取与缩放中位数约56ms，backbone约27ms，detector约12.5ms，VFCE pooling/projection合计约0.7ms；两个相机仅这些前端部分就约196ms/场景。该估算来自分项中位数，不能替代实测总耗时。动态Question与OptionReader、历史构建仍有额外开销，嵌套计时不累加。剖析前缀没有REACT/bank调用，相关分项是未观测而非零延迟。','',
        '确实卸载21,750,016个dormant GTA参数，实际释放约85.3MiB显存，ID与每个Instances字段SHA不变；单次未卸载参照不能证明FPS收益。缓存保留原浮点mean顺序，并校验tensor替换、in-place更新、跨相机metadata、恢复和ID复用；106帧真实原生覆盖证明之外，九个完整视频也逐项比较原始/canonical预测、解压后的每条logit与commit日志和最终完整原生状态。严格等价通过，数值差0。25场景FPS与Stage2 p95≤10ms仍为FAIL。','',
        '最终15个研究问题的回答：','',
        '1. **Phase XIII额外IDSW来源**：Full相对Fixed三种子合计多41次，其中纯净旧碎片切换/恢复多38次；错误已有身份合并多4、split出生多2，其他变化抵消3次。主要是已有碎片切换，不是所有IDSW都由新生造成。','',
        '2. **UNKNOWN影响**：旧Full有3063次、Fixed2294次UNKNOWN抢占已认证正确候选（三种子总计）；不少UNKNOWN是已经污染的历史。训练CE/旧结构目标只规范known，但部署完整合法集包含UNKNOWN。新joint目标允许合法UNKNOWN零代价，保留UNKNOWN语义，仍不能把它变成可靠动作证书。','',
        '3. **旧动态Question不如Fixed原因**：旧正式监督主要为MATCH，三个task embeddings不代表三任务学习。错误增加主要是片段间不稳定转换；受控输入删除揭示依赖，不能单凭扰动确定架构因果。','',
        '4. **真正多问题的独立价值**：WHO/Q2/Q3支持已合格，学习能力通过；普通Fixed/Set同样可以学会。正式同证据结果没有通过预注册独立收益门槛，完整稳定性也失败。','',
        '5. **不把UNKNOWN作负例的availability**：可以。90526个自然positive、55个严格negative、6483个可靠候选移除事件支持TRAIN监督；剩余模糊候选不认证为缺席。困难、count-matched presence与成对absence分列，reserved块无梯度。高Q2正确率不等于在线稳定性。','',
        '6. **同时减少split且不增加merge**：'+f'在本轮同监督20k比较，Multi三种子split出生{devcounts("formal","multi_question").get("FALSE_SPLIT",0)}、新混合{devcounts("formal","multi_question").get("FALSE_MERGE",0)}，Fixed为{devcounts("formal","fixed_question").get("FALSE_SPLIT",0)}、{devcounts("formal","fixed_question").get("FALSE_MERGE",0)}；相对旧冻结Fixed，新混合{old["formal/fixed_question"].get("FALSE_MERGE",0)}反而增加，IDSW也失败。不能宣布可靠身份控制成功。','',
        '7. **旧on-policy为何Wrong-anchor减少却出生增加**：旧Full三种子TRAIN first256的birth-anchor错误观测5304→2389，额外出生碎片107→195。新ID重设首GT锚点可以降低anchor错误计数，故它不是无条件纠错。旧开发HOTA仅75.056→75.635、IDSW118.667→143.667。本轮三种24k控制与144次真实TRAIN轨迹检查该权衡，没有以更低Wrong-anchor单指标宣布成功。','',
        '8. **跨相机视觉与全局统计贡献**：已完成9个单因素及3个组合、36个prefix×seed实例/条件、每实例16帧真实mutated未来。其他相机prototype和全局历史/计数删除的计数差见下表；合法候选数量仍通过输入shape可见，own/other角色仍含隐式相机信息，不能声称完全移除所有跨相机信息。','',
        '9. **长期视觉具体依赖**：历史Full固定输入中，seed20261010删除global_mean+counts时已认证correct294→101，193退化；计数单删仅1次退化。新真实分支验证其后续状态依赖，但所有特征删除是训练外OOD，不能据此证明重训结构不可替代。','',
        '10. **JEV是否优于普通Set**：Multi20k均值HOTA略高于Set，但其IDSW更差；同时相对Fixed增益不足且部分seed退化。未证明稳定独立架构优势。','',
        '11. **在线原生契约**：GTA/RPCE throw、真实Gallery/hits/bank/RNG、同相机ID容量1、序列化resume、ID复用以及9个完整缓存开关对照通过。性能运行没有强制GT身份、假候选或teacher-forced历史。工程覆盖分支的合法强制动作仅用于覆盖测试，独立于学习与性能结果。','',
        '12. **耗时瓶颈**：Stage2的历史构建和Question/Option处理仍需优化；端到端首先受图像读取缩放、backbone及detector限制。删除dormant参数释放显存，尚未使任何方法达到25FPS。','',
        '13. **独立泛化条件**：已完成固定外部场景迁移，但初始化全训练清单仍未知，严格未见全系统结论NO_GO。开发/封存Stage1暴露问题未被重命名解决。','',
        '14. **是否继续完整MATCH–REACT–MEMORY**：当前NO_GO。WHO/availability/trust是真实独立监督问题，但不是已学会的REACT/MEMORY动作；自然stale正例仅9，可信污染不是WRITE奖励。补足真实生命周期结果与动作监督前不继续扩展。','',
        '15. **最可信WWW贡献**：可验证原生因果状态管理、UNKNOWN-safe身份缺席/可信度监督、同证据同预算负结果及身份碎片诊断。当前数据不支持“多问题JEV优于普通Attention”的论文主张；下一阶段应优先解决同GT多个纯碎片的时序承诺与UNKNOWN受限证据认证，先用普通控制器公平对照，不继续堆层或仅增加更新。','',
        '| 真实未来删除条件 | 改变的提交行数 | 新跨GT混合差值 | false birth差值 | 近期跨相机不一致差值 |','|---|---:|---:|---:|---:|']
    for x in paired['summary']:
        if x['condition']=='unchanged':continue
        c=x['delta_counts'];lines.append(f'|{x["condition"]}|{x["changed_committed_rows"]}|{c.get("new_cross_GT_gallery_mix",0):+d}|{c.get("false_birth",0):+d}|{c.get("recent_cross_camera_ID_mismatch",0):+d}|')
    lines+=['','以上差值是12个实际B1前缀×3种子累计，分支互相重叠，不能当独立整视频效应；近期跨相机不一致不是官方CVIDF1，碎片hop也不是CLEAR IDSW。完整来源、每次干预的起始状态SHA与日志均在[PAIRED_NATIVE_FUTURE_RESULTS](../reports/JEV_PHASE14/PAIRED_NATIVE_FUTURE_RESULTS.json)。','',
        '| Gate | 最终状态 |','|---|---|']
    for k,v in final['gates'].items():lines.append(f'|{k}|{v["status"]}|')
    lines+=['','成本语义限制必须保留：错选已认证纯历史可构成wrong existing/merge；MATCH terminal的代价是false DEFER/split风险代理，后续可能由bank恢复，不是直接NEW代价。相同GT的多个纯碎片被WHO多positive接受，目前联合损失没有额外惩罚纯碎片hop。生命周期代价先验已记录，但没有虚构独立false-birth/WRONG-REACT训练标签。这解释了为什么正确WHO/availability并不保证低IDSW，具体因果贡献仍需新的合法实验。','',
        '工程修复：修正最初query-only availability对Fixed的不公平输入；所有head接收相同Option证据后重做24个Pilot，旧结果永久保留。补零检测guard（参数及非空运算保持一致）；缓存加入tensor身份与版本以防指针复用；前缀在边界立即序列化，修复测试harness共享引用；处理ZIP32bit偏移并核验成员名/CRC，直接按HTTP Range读取必要外部成员；MATLAB和空数组适配只处理输入格式，不修改官方指标函数。所有失败/修正证据见ENGINEERING_REPAIRS与服务器队列日志。','',
        '科学大文件留在服务器：模型、原生日志、前缀、外部图像与自身状态数据。GitHub只包含代码、协议、紧凑结果与图表。完成后按提交HEAD复核，[SOURCE_INTERFACE_MANIFEST](../reports/JEV_PHASE14/SOURCE_INTERFACE_MANIFEST.json)记录实际运行源码/config/数据/checkpoint SHA，多个不可变运行checkout不会被最终汇总HEAD冒充。','',
        '主要交付：[ONLINE_VALIDATION](../reports/JEV_PHASE14/ONLINE_VALIDATION.json)、[OFFICIAL_MATLAB_RESULTS](../reports/JEV_PHASE14/OFFICIAL_MATLAB_RESULTS.json)、[FAIR_BASELINE_RESULTS](../reports/JEV_PHASE14/FAIR_BASELINE_RESULTS.json)、[ON_POLICY_STABILITY](../reports/JEV_PHASE14/ON_POLICY_STABILITY.json)、[LATENCY_ATTRIBUTION](../reports/JEV_PHASE14/LATENCY_ATTRIBUTION.json)、[EXTERNAL_GENERALIZATION_RESULTS](../reports/JEV_PHASE14/EXTERNAL_GENERALIZATION_RESULTS.json)、[FULL_VIDEO_CACHE_PARITY](../reports/JEV_PHASE14/FULL_VIDEO_CACHE_PARITY.json)、[FINAL_GO_NO_GO](../reports/JEV_PHASE14/FINAL_GO_NO_GO.json)。','']
    text='\n'.join(lines)
    text=text.replace('同监督比较中，Multi-question相对Fixed的HOTA差值仅 ',
        'Multi-question的官方CVIDF1三个种子均高于Fixed，平均88.546对86.405，原始IDF1和AssA也提高；这些正向结果完整保留。但同监督比较中，Multi-question相对Fixed的HOTA差值仅 ')
    text=text.replace('GPU9 V100同图像、同前端、同线程，',
        'GPU9测试期间无其他GPU进程竞争，但同主机其他GPU仍运行实验，CPU/I/O耗时包含该共享主机条件；不是整机独占测速。GPU9 V100同图像、同前端、同线程，')
    text=text.replace('Clean Stage1没有合格初始化来源、独立配置与预算，按条件Gate不启动。',
        '已审计的完整CH前端初始化文件未证实通用训练来源，clean Stage1/GMT及共同控制器的完整配置与独立预算门槛未满足，按条件Gate不启动；这不意味着不存在其他可进一步核验的通用backbone初始化。')
    runtime=read(REPORTS/'EXTERNAL_RUNTIME_DIAGNOSTICS.json');slow=next(x for x in runtime['cases'] if x['variant']=='set_transformer' and x['seed']==20261009 and not x['historical'])
    text=text.replace('最终15个研究问题的回答：',
        f'外部Set Transformer seed20261009实际产生{slow["actions"]["START_NEW"]}次START_NEW，累计原生ID计数{slow["maximum_native_id_count"]}，最近40场景帧提交ID并集峰值{slow["peak_IDs_committed_over_recent40scene_frames"]}。其缓存推理、日志、TrackEval和官方MATLAB合计{slow["cached_run_seconds_including_GT_TrackEval_MATLAB"]:.1f}秒，不能当完整图像FPS。官方身份矩阵为13665×13665，大量碎片也明显增加评价计算。该失败种子完整保留，无临时重置或候选裁剪；历史格式转换器写入的seqinfo.frameRate=30不用于benchmark=VisionTrack的本轮指标函数，实际外部采样率仍为2FPS。详细原生行动与规模见[EXTERNAL_RUNTIME_DIAGNOSTICS](../reports/JEV_PHASE14/EXTERNAL_RUNTIME_DIAGNOSTICS.json)。\n\n最终15个研究问题的回答：')
    (ROOT/'docs/JEV_PHASE14_FINAL_RESEARCH_REPORT.md').write_text(text)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=REPORTS/'figures';figdir.mkdir(exist_ok=True)
    figure,axes=plt.subplots(1,2,figsize=(10,4));colors=['#546e7a','#00796b','#ef6c00'];xx=np.arange(3)
    for ax,key in zip(axes,['HOTA','IDSW']):
        ax.bar(xx,[formal[v][key]['mean'] for v in variants],color=colors,alpha=.8)
        for i,v in enumerate(variants):ax.scatter(i+np.linspace(-.12,.12,3),formal[v][key]['seeds'],color='black',s=18)
        ax.set_xticks(xx,labels,rotation=15,ha='right');ax.set_ylabel(key);ax.grid(axis='y',alpha=.2)
        if key=='IDSW':ax.axhline(102.5,color='red',ls='--',label='Frozen gate102.5');ax.legend()
    figure.suptitle('Same evidence / supervision /20k budget — full native development');figure.tight_layout()
    for ext in ['png','pdf']:figure.savefig(figdir/('FORMAL_STABILITY.'+ext),dpi=180)
    plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(10,4))
    for i,v in enumerate(variants):
        axes[0].plot(np.arange(4),[mean(p,v,'HOTA') for p in ['formal','onpolicy','oldloss_onpolicy','offpolicy']],marker='o',label=labels[i],color=colors[i])
        axes[1].plot(np.arange(4),[mean(p,v,'IDSW') for p in ['formal','onpolicy','oldloss_onpolicy','offpolicy']],marker='o',color=colors[i])
    for ax,key in zip(axes,['HOTA','IDSW']):ax.set_xticks(range(4),['20k','Own +new4k','Own +old4k','Original +new4k'],rotation=15,ha='right');ax.set_ylabel(key);ax.grid(alpha=.2)
    axes[0].legend();figure.suptitle('20k baseline and24k equal-extra-budget controls');figure.tight_layout()
    for ext in ['png','pdf']:figure.savefig(figdir/('EXTRA4K_CONTROLS.'+ext),dpi=180)
    plt.close(figure)
    print('PHASE14_RESEARCH_REPORT_AND_FIGURES_COMPLETE',flush=True)

if __name__=='__main__':main()
