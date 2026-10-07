"""Freeze the final diagnostic decision and write its reviewer-facing report."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'reproduction_tools'))
from audit_jev_phase5 import OUT,save


def read(p):return json.loads((OUT/p).read_text())

def table_row(name,m):return f"| {name} | {m['HOTA']:.4f} | {m['AssA']:.4f} | {m['IDF1']:.4f} | {m['IDSW']:.0f} | {m['MOTA']:.4f} |"


def main():
    variants={v:read(f'ablations/{v}/result.json') for v in ['A0','A1','A2','A3','A4']};models={}
    for condition in ['C1','C2','B2']:
        tracking=read(f'minimal_tracking/{condition}/result.json');binding=read(f'minimal_training/{condition}/binding.json');payload=torch.load(OUT/f'minimal_training/{condition}/model.pth',map_location='cpu');params=sum(v.numel() for v in payload['model'].values())
        assert binding['compact_manifest_sha256']==read('MATCH_TRAINING_DATASET.json')['compact_manifest_sha256']
        models[condition]={'model':payload['model_name'],'params':params,'binding':binding,'tracking':tracking,'calibration':read(f'minimal_training/{condition}/calibration/calibration_val_only.json')}
    assert all(abs(x['params']-models['B2']['params'])/models['B2']['params']<.003 for x in models.values())
    off=variants['A0']['metrics'];primary=models['B2']['tracking'];gate=all(primary['metrics'][k]>m[k] for m in [off,models['C1']['tracking']['metrics'],models['C2']['tracking']['metrics']] for k in ['AssA','HOTA'])
    save('CORRECTED_THREE_WAY_COMPARISON.json',{'status':'COMPLETE','models':models,'same_complete_video_split':'train07/val06; diagnostic01','same_features_labels_weights_and_masks':True,'epochs':20,'batch_size':128,'lr':.001,'seed':20261003,'parameter_difference_limit':.003,'JEV_above_OFF_and_both_corrected_baselines_on_AssA_HOTA':gate,'MLP_collapse_independently_checked':read('MLP_POLICY_SANITY.json'),'scope':'single-seed TRAIN diagnostic; no universal architecture claim'})
    save('FROZEN_MATCH_CONTROLLER.json',{'status':'FROZEN_RESEARCH_CANDIDATE','original_frozen_A1_reference_checkpoint_sha256':variants['A1']['checkpoint_sha256'],'primary_corrected_B2_checkpoint_sha256':primary['checkpoint_sha256'],'primary':'B2 was the predeclared corrected-label condition; retained as a candidate, no heldout hyperparameter selection','online_questions':{'MATCH':'B2 learned JEV','MEMORY':'GMT','REACTIVATION':'GMT'},'stability':read('B2_STABILITY.json'),'native_production_deployment_validated':False,'Full24_GO':False,'next_gate':'small native MATCH-only integration and additional complete TRAIN holdouts; validate feature contract and branch supervision before learning Reactivation/Memory'})
    ablation='\n'.join(table_row(v+' '+label,variants[v]['metrics']) for v,label in [('A0','GMT'),('A1','MATCH'),('A2','MATCH+REACT'),('A3','MATCH+MEMORY'),('A4','Full JEV')])
    comparison='\n'.join([table_row('GMT OFF',off),table_row('C1 Threshold MATCH-only',models['C1']['tracking']['metrics']),table_row('C2 MLP MATCH-only',models['C2']['tracking']['metrics']),table_row('B2 JEV MATCH-only / aligned GT',primary['metrics'])])
    b1=read('minimal_tracking/B1/result.json')['metrics'];d=read('MINIMAL_RETRAINING.json')['conditions']['B2']['delta_vs_OFF'];coord=read('MATCH_TRAINING_DATASET.json');sanity=read('MLP_POLICY_SANITY.json')
    text=f'''# WWW / JEV Phase V 实验报告 — 2026-10-07

本批实验完成了冻结 checkpoint 组件消融、逐决策与时间窗诊断、全量 MEMORY 标签审计、TRAIN 重关联覆盖审计、native 特征契约对照、GT 帧坐标修正、两组最小 MATCH-only 重训，以及修正标签上的三组公平对照。代码和证据保存在独立分支 `jev/www-jev-phase5-20261007`，原 review 分支和原实验工作树保留。

当前候选是 **修正 GT 坐标后的 JEV MATCH-only（B2）**；MEMORY 和 REACTIVATION 仍由 GMT 决策。B2 相对同一个 GMT OFF：HOTA +{d['HOTA']:.4f}、AssA +{d['AssA']:.4f}、IDF1 +{d['IDF1']:.4f}，IDSW 282→87。B2 的两次跨 GPU 重放，预测、动作、在线特征/context 日志 SHA 均一致，全部指标差为 0。

这是 **单 seed、TRAIN 内 video01 诊断结果**。video01 已多次用于分析，不是独立最终测试。没有运行官方 TEST、Full24、seed sweep，也没有据此宣布 WWW 最终结论或原生部署完成。

## 1. 输入与冻结基线

- review 基点：`add60a61e06321448e83808f8253a7f255d4ede5`；原实验基点：`1d2711e80ac5fa00806fd9eed30e90cd51df6b30`。
- GMT `model_20000.pth` SHA：`cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`。
- 原校准 JEV SHA：`2509541a6669d8fc31013111b7bb9a38c4551be79ff5a466b259184029dd871b`。
- 完整 train video07 / val video06；诊断 video01。全部数据来源为 TRAIN。
- A0 / A4 的预测 SHA 与原 OFF / Full JEV 逐字节一致，七项指标误差均为 0。因此原闭环指标有效；后面发现的错误发生在监督标签的 GT 查找。

完整输入路径、annotation/cache/config/checkpoint SHA 见证据包 `PHASE5_BASELINE_LOCK.json`。

## 2. 同一冻结 JEV 的组件消融

| 配置 | HOTA | AssA | IDF1 | IDSW | MOTA |
| --- | ---: | ---: | ---: | ---: | ---: |
{ablation}

每个配置只改变问题由谁控制，后续 proposal、tracker state 和决策全部在线重算。没有把旧 OFF 轨迹中的动作直接当成 learned 轨迹的反事实真值。

MATCH-only 在预先指定的 HOTA、AssA 两项上超过 GMT。加入 REACTIVATION 后变差；MEMORY 单独加入影响很小，但在 learned REACTIVATION 下出现更明显的交互损失。正式 Frag 均为 6。原 A1 checkpoint 继续保留为固定参照，重训没有覆盖它。

## 3. 为什么 IDSW 大降，AssA 反而下降

全局去重的预测 ID 数：GMT 230，冻结 MATCH-only 9，Full JEV 4；它们与 TrackEval 按相机统计的 IDs 数不同。Full JEV 的混合身份轨迹包含 268 个少数身份观测，GMT 为 0。由此可见，减少短 ID 能改善连续性，但把不同真实身份长期放入同一个 ID 会损害关联质量。

数据不支持“Full JEV 过度使用 START_NEW”：MATCH 的 START_NEW 仅 6 次，GMT 为 227 次。MATCH 与 REACTIVATION 对同一检测可以都记录 START_NEW，动作计数不能直接相加当成出生 ID 数。正式 Frag 没有增加，也不应把辅助的 identity-segment proxy 当成正式 Frag。

N01/N10 使用每图 IoU≥0.5 的一对一 GT 匹配，再按相机建立全轨迹的一对一 predicted-ID / GT-ID 映射。GT 只用于离线诊断。此统计是**配对完整轨迹的观察结果**，不能据此把每个 MEMORY 动作单独认定为因果损害。动态出现/消失的问题事件单独报告；重叠未来窗口不当成独立帧累加。

+1/2/4/8/16/32 帧的正确/错误身份时长、切换、身份段、新 ID、恢复与写入冲突见 `PAIRED_DECISION_AUDIT.json`、`TEMPORAL_DAMAGE.json` 和压缩的逐决策原始文件。B2 的附加诊断见同名 `B2_` 文件。

B2 的全局预测 ID 为 108，少数身份观测仅 2；GT 映射后的错误观测从 GMT 的 151 降至 47。该辅助诊断支持 B2 在保留正确身份与创建必要新 ID 之间取得了更好的平衡；最终质量仍以 TrackEval 指标为准。

## 4. MEMORY 与 REACTIVATION 审计

全部 8,490 个 MEMORY 记录均检查了 WRITE/SKIP 分支：video01 与 video06 的 utility 和全部 outcome 100% 相同；video07 的 96.08% 为平局，另外 64 个仅出现窗口内 contamination 计数差异，身份时长、切换和碎片指标没有区分。

预先固定的 video06 64 个完整 future32 事件，在 H8/H16/H32 下依然 100% 平局，修正 GT 坐标后也是如此；H8 控制结果与冻结标签精确一致。该序列没有 Reactivation，普通关联 transformer 读取 association history，不直接读取 stale memory bank。因此 **这一组的 M2 结果不表示 MEMORY 在所有序列中都无效**，但不足以支持只靠延长 horizon 训练 MEMORY。

Full JEV 的 7 次实际 REACTIVATE_OLD：2 次当前 GT 与候选 ID 的历史 GT 不同，5 次当前检测无有效 GT 匹配，均不能计为成功恢复；分数全部低于 GMT bank threshold 0.4。原 train07/val06 的 REACTIVATION 样本数均为 0。

按照全部实际写入历史重建 native bank 的 GT 样本标签，memory length 与每个在线状态记录一致。Full JEV 的 7 次 bank promotion 共 70 个样本全部没有有效 GT 匹配；不能因为“没有混合已匹配的 GT 身份”就称 bank 干净。此项与 MEMORY/REACTIVATION 交互退化一致，但没有完成逐动作的局部因果证明。

24 个 TRAIN trace 的覆盖审计按预先声明的事件数排序选出 train video24（1,556 个旧 trace 重关联事件）、val video23（1,310 个）。这些旧 trace **仅用于覆盖与选序列，不能直接作为新的 canonical 标签**。

## 5. 发现并修正 GT 帧坐标错误

缓存 frame0 对应 metadata.dataset_frame1 / annotation.frame_id1；view0 对应 view_id1。旧 labeler 使用 `(video, view+1, frame)`，并回退到另一种相机索引。三个视频的全部 3,952 个缓存 frame/view 键都查到了错误图像位置，90 个检测的实际 GT target 改变。

修正契约是 `(video, view+1, frame+1)`，不回退到另一帧或另一相机。新 builder CLI 默认 `--gt-coordinate-contract cache0_annotation1`，manifest 和新记录显式标记该契约。函数层仍保留显式 legacy 兼容入口，以复现冻结结果；新的程序调用必须传入正确契约。该新查找与 Phase V 的归一化 GT 重标方法，在三个视频的全部检测上逐项一致。

完整 4,303 个 MATCH 记录都形成修正数据集。按几何/GT 坐标差异穷尽选出受影响窗口，并预先加入 8 个未受影响控制窗口，共重算 766 个事件；保留全 trace/future maps 和冻结 OFF 起始 snapshot。766 个 H8 控制 outcome 全部精确复现，8 个未受影响窗口逐项一致，其余窗口因没有任何检测/GT 匹配差异而可精确复用。

713 个分支 outcome 集合改变，117 个 best-action set 改变。两组数据的 features、questions、legal masks、sample weights、序列/帧/相机/event-order 数组全部逐字节一致；只有 134 行 target probabilities 和 117 行 best masks 改变。GT 没有进入在线 controller 特征。

## 6. 最小重训及公平三组对照

固定 seed20261003、20 epochs、AdamW lr0.001、batch128、最后第20 epoch checkpoint；校准仅使用完整 val video06。train1703 / val2600，全部为 MATCH。没有用 video01 做 checkpoint/超参数选择。

B1 使用旧坐标 MATCH 标签：HOTA {b1['HOTA']:.4f}、AssA {b1['AssA']:.4f}，明显退化。B2 是预先声明的修正坐标主实验，结果如下。删除 MEMORY 后的训练批次数也改变，所以 B1 相对原多问题模型的差异不能只归因于 MEMORY loss。

监督发生变化后，补做相同修正数据、相同训练设置与相同 MATCH-only 在线控制范围的 Threshold / MLP。参数分别为 33,987 / 34,163 / 34,080，相对 JEV 偏差 <0.3%。

| 方法 | HOTA | AssA | IDF1 | IDSW | MOTA |
| --- | ---: | ---: | ---: | ---: | ---: |
{comparison}

MLP 极端失败已单独核对：训练/校准权重相同；CPU 重新计算全部 4,524 个在线输入，动作零不一致；训练 Tensor API 与 runtime list/string API 的首个动作一致，首个状态与 OFF 逐字节一致。首步 START_NEW 概率约 {sanity['first_probabilities'][0][2]:.6f}、ACCEPT 约 {sanity['first_probabilities'][0][0]:.6f}，很小的初始偏好随后进入持续新建 ID 的状态。该 checkpoint 在固定 OFF 状态上的动作则为 ACCEPT4066 / REASSOCIATE296 / START_NEW162。这支持实际在线状态演化导致崩溃的解释，而非加载或动作编号错误；不构成“MLP 普遍无效”的结论。

**本批修正标签的单 seed 诊断中，JEV 的 HOTA 和 AssA 高于 GMT 与两组同数据基线。** 高动作验证准确率仍不能替代闭环跟踪指标；大规模、独立序列和多 seed 结论尚未建立。

B2 校准 checkpoint SHA：`{primary['checkpoint_sha256']}`。模型、校准报告、原始预测/决策、完整 TrackEval 结果均在证据包中。

## 7. native 特征契约与下一步

全部 native/MATCH 4,524 和 MEMORY 4,297 个向量在冻结 float32 envelope 内一致；REACTIVATION 174 个向量均有真实语义差异：native bank 使用局部 query frame、view0、局部 T；研究 controller 使用真实 frame/view/关联窗口。大数值 score/variance 的微小浮点差异已与这三项语义差异分开统计。

WWW 研究候选采用显式 canonical online research 契约。原生集成须单独实现/验证，不能把现有 native bank 特征称为等价，也不能直接把三个问题都交给只训练 MATCH 的 B2。

本批冻结 B2 为后续研究候选，同时保留原 A1。下一项小实验应做 native MATCH-only 集成与额外完整 TRAIN 留出序列验证。若继续学习 REACTIVATION/MEMORY，需要先验证前缀身份/memory anchors 与真实分支 continuation 的监督协议，再收集 canonical trace。当前 `score_rollout` 的窗口 anchors 会重置，未来 action category 仍来自固定 OFF maps；窗口 AssA proxy 不是正式 AssA。这些限制没有被坐标修正自动解决。

`NEXT_SMALL_DATASET_V2_DESIGN.json` 给出四个完整 TRAIN 来源视频的设计（train07/24、val23、诊断01），未开始该构建。MEMORY 的 informative supervision 尚未建立，不人为指定 WRITE/SKIP 标签。**本批结果不等于 Full24 GO。**

## 8. 代码、验证与证据

主要代码：

- `run_jev_phase5_ablation.py`：按问题路由冻结/新 controller；使用即时 GMT fallback，输出完整因果在线 context、进度和失败记录。
- `build_jev_counterfactual_v2.py`：仅过滤输出事件的可选 hook、全 future rollout observer，以及显式 GT 坐标契约。未来 maps 不裁短，冻结 legacy 路径保留。
- `run_jev_phase5_horizons.py`：复用 OFF snapshots 的分段 H8/H16/H32 与受影响 MATCH 窗口重算。
- `prepare_jev_phase5_match_training.py`、`run_jev_phase5_minimal_training.py`：完整序列重标、固定重训和 val-only 校准。
- `audit_jev_phase5*.py`：全量标签、特征、覆盖、N01/N10、时间窗、前缀/bank、MLP 加载/动作诊断。
- `archive_jev_phase5_report.py`：压缩并保存原始证据，记录 source SHA、archive SHA、输入 binding 和排除的大缓存。

验证：新 GT 坐标/错误相机回退与全 future 输出过滤回归、chunk/snapshot 测试共4项通过；branch-local RNG 隔离通过；A0/A4 冻结基线复现通过；64+766 个 H8 控制 outcome 精确复现；8个复用控制通过；B2 两次完整重放精确一致；MLP 4,524 个在线动作独立重算一致。

初次 ablation 初始化失败、初次选择器漏掉首帧 seed 顺序、GPU 被新任务占用后的本轮进程迁移均保留了日志/元数据；没有终止其他任务。GPU 调度按用户新规则优先无进程显卡，显存足够时可叠加。

证据目录：[`reports/WWW_JEV_PHASE5_20261007`](../reports/WWW_JEV_PHASE5_20261007)。包含三视频全部冻结记录、全部 MEMORY 统计、修正/旧 MATCH 数据、native64特征审计视图、十次跟踪结果、checkpoint、训练/校准与运行日志。`ARCHIVE_MANIFEST.json` 和 `SHA256SUMS` 可供逐项核对。大 GMT checkpoint、感知缓存和数据集图像不上传，固定输入 SHA 保留。

运行目录：`/home/liuyeqiang/WWW_jev_phase5_runtime/20261007`。代码在独立工作树 `/data1/liuyeqiang/WWW_jev_phase5`。所有实验选择、重算窗口选择与公平基线补充条件均保存为事前 plan 文件。
'''
    (ROOT/'docs/WWW_JEV_PHASE5_DIAGNOSTICS_20261007.md').write_text(text)
    save('FINAL_PHASE5_DECISION.json',{'status':'COMPLETED_DIAGNOSTIC_BATCH','primary':'B2 corrected-label JEV MATCH-only','candidate_positive_gate':gate,'Full24_GO':False,'native_deployment_validated':False,'original_A1_retained':True,'primary_checkpoint_sha256':primary['checkpoint_sha256'],'tracking_runs_completed':10,'new_single_seed_training_runs':4,'report':'docs/WWW_JEV_PHASE5_DIAGNOSTICS_20261007.md','next':'native MATCH-only integration and independent complete TRAIN holdouts; supervision protocol redesign before Reactivation/Memory training'})
    print(json.dumps({'status':'COMPLETE','candidate_positive_gate':gate,'params':{k:v['params'] for k,v in models.items()}}))
if __name__=='__main__':main()
