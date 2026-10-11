# Phase XVII：无新增决策头的视觉 teacher

本轮使用冻结的12个独立TRAIN困难组、261个全部合法MATCH候选，分别提供真实当前/过去图像。没有Top16截断、未来帧、GT答案标记、跨问题答案或在线teacher干预。每个候选独立问WHO、历史一致性、可辨识性、跨摄像头兼容性与已知假前提；独立softmax不能拼成未经校准的全局身份排序。历史一致性不是GlobalID所有权。

模型为Qwen3-VL-4B-Instruct预训练原LM head和Visual-Jev公开通用QA LoRA，均with_heads=False，未实例化新linear head、未做身份任务训练。通用LoRA已有训练，不能把“无新增head”说成“不需要训练”。独立环境torch2.5.1+cu118/transformers5.17.0、V100 FP16；作者RTX5090 BF16性能不能移用。

242个已知WHO对：预训练准确率49.17%、NLL1.949、ECE0.306；通用QA LoRA准确率28.93%、NLL1.875、ECE0.329。冻结余弦0.75探针准确率95.45%；相对其N01/N10分别为4/116和2/163。这是聚类在12组内的候选对诊断，不是独立样本显著性或公平架构归因。19个混合/未知对单独报告，不自动算错。假前提正确拒答分别97.70%和2.68%，QA适配不能保证迁移后的拒答。

唯一具备合法正确owner的原生错误事件video16/frame21/camera0/row0：正确ID2与错误所选ID6，预训练给same概率0.99986/0.99970，QA LoRA给0.94128/0.92243，均选择same，无法可靠区分。这不证明所有VLM都无用，但本轮没有“数值证据不能区分、视觉teacher能够修复”的合格owner增益。D0原GMT GTA分数和D4同图像MLP/DeepSets/Attention没有在同前端完成，明确NOT_RUN，不能把图像或预训练带来的作用归因于JEV结构。身份任务LoRA及新增head训练也NOT_RUN。

发现并保留上游分词缺陷：prepare_group分别分词prefix/suffix，未调用源码split_ids检查，真实完整分词与拼接不一致。原输出仍在teacher_inference。局部TokenVerifiedVDM改用真实完整tokenization的公共前缀，全部新候选验证拼接完全相同；teacher科学输出来自teacher_inference_v2的独立serial full forward。原失败与新输出均有SHA，不改外部源码、不放宽误差门槛。

V100 FP16缓存执行仍未全部通过0.05有效logit差、0.02概率差且决定一致门槛；选项逆序后只有18/24组语义决定一致。因此缓存/动态选项资格NO_GO。KV fork未共享可变底层，M-RoPE前缀逐问题验证；每个候选/模型重新prepare和prefill，没有跨frame/camera/history的持久缓存。内容键变化测试只是未来缓存准入诊断，不等同部署缓存验证。

N=1/4/8/16/32各1次warmup、3次CUDA同步测量，p95取3次最大值，样本很小。N32共享执行对serial中位数有6.41x/6.73x的诊断加速，每问14.12/20.10ms；N1反而变慢。N>5重复部分问题，K2/3/4/8/16采用重复不确定选项压力测试，不是多ID排序稳定性证明。这些数值不能作为合格在线MOT吞吐或25FPS证据；不同候选图像没有共享同一视觉prefix。

结论：不扩大VLM，不接入在线逐帧控制器，不生成已验证纠错标签。优先审计原生候选/owner证据与数值控制器错误，资格满足后再比较同证据的轻量模型。
