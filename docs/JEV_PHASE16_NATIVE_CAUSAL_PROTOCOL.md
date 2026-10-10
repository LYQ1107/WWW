# Phase XVI 原生因果协议

以 Phase XV TRAIN12/13/14/16 的全部 24 个已冻结 pre-association 前缀为起点，每个前缀执行 A/B/C/D 共四条真实原生未来，共 96 条分支。保留 instances、原始 Gallery、hits、Bank、possible IDs、ID counter、CommitmentState 以及 Python/NumPy/Torch/CUDA/trajectory RNG。GT 只存在于决策后离线 observer，不进入模型、记忆规则、前缀重建或原生求解器。

A 是冻结 XV V3 和无条件写入；B 只改变未来记忆接纳；C 只在目标行第一次决策使用 P1 失败的 E 读取方案；D 同时执行 C 和 B。C/D 是预先固定的负对照，不是新合格候选；其他行仍用原始分数，native Hungarian 可以因容量竞争重新分配其他行，报告其实际变化。首轮 DEFER 仍进入真实合法 stale Bank 恢复，不能等同于安全拒绝或人为强制 START_NEW。此后所有组用同一个冻结四槽解码器，各自继续变异状态，绝不同步未来 Gallery。

第一步原始输入、原始分数、派生前缀和 RNG 指纹在四组必须一致。A 的 H32 全部提交 ID 必须与保留的 XV V3 分支逐条相等。额外 always_write 接口合同在最早有害前缀验证原始状态和独立状态一致、还原无别名以及前缀不被修改。

所有分支报告 H8/H16/H32。H64 限于冻结严格 TRAIN 前缀按 video/key/row 排序的前八个，不根据结果择优。TRAIN14 的重复 GT 被离线风险 observer 标记为 UNKNOWN；原始 GT 和 official metrics 不改写，TRAIN14 整组仅为 permissive CLEAR 诊断，不能参加严格安全门槛。

CLEAR IDSW 使用原始 prepared GT/检测框/分数，只替换真实分支 ID；保留从 frame0 到干预前的 CLEAR matching history。认证 owner 来自过去首次单 GT、至少三条且已知覆盖至少 80% 的观测，一旦建立即固定，后续混合不能抹去 owner 错误。无法认证为 UNKNOWN，不能强制标错或算正确。传播时间仅统计实际检测到的错误帧；观测断点分段并删失，horizon/video end 删失，不能推算不可见时长。

除总错误观测和最长错误 episode，还比较每个 GT/camera 的错误 episode；任何严格前缀、任意预定 H 的混合、错误持续时间或 false_split_birth 恶化否决。至少八个独立 identity/camera/64-frame cluster 的安全收益才可能合格。共享或相邻前缀不能当独立样本。窗口求和只是描述，不是独立全视频指标。配对报告额外公开正确↔错误↔UNKNOWN 的认证覆盖转移，避免把失去认证误当纠错。

工程失败和 source actor 均保留。闭合的真实提交 trace 支持 CPU 重算 observer 的 gap censor flags；禁止因此改动动作、ID、GT、原始失败或原生未来结果。
