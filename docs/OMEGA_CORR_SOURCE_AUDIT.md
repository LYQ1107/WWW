# Ω-Corr 源码与资产审计

新分支从远端594d9d32建立，旧报告与负结果原样保留。已读要求的七份报告及官方aggregator/model/camera/dense代码与现有adapter/geometry。

官方源码48b23c8；512权重SHA重新核验一致。旧96组共96份npz只包含背景坐标，无中间patch tokens，必须重新做一次受控token提取；不重跑F/Pose扫描。

缓存层为零基4/11/17/23；2048拼接当前frame1024与inter-frame1024。frame半部此前已跨视角交互，不是纯单视图。前17个特殊token必须剔除。单视图只使用first-frame特殊嵌入，配对/单图比较有slot差异；人工同图错误partner控制保留原slot并单独报告。

原生训练审计与DEV12封存资产已定位，禁止重跑完整12场景审计。新的场景角色及160组候选在token分数前固定；旧F通过不是本研究前提。

独立SciPy依赖覆盖层位于新runtime/deps，不修改旧环境。完整源文件SHA、旧缓存键、权重来源与后续精度/接口证据见机器可读source_audit和token_preflight。
