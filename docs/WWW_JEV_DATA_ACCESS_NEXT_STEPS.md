# 数据访问下一步 — 2026-10-12

用户已明确回复：目前没有合法的 MMPTRACK 或 SCOUT 副本。此前询问现有副本路径的问题已经得到答案，不再把未提供路径描述为未回复。

优先申请 MMPTRACK。官方页面列出 TRAIN、VAL、TEST 的参与者划分，较符合当前独立验证设计的需要；这只是来源选择依据，尚不能证明新监督满足原生安全门槛。研究者需从官网下载使用协议，填写、签署后按官方要求发给数据提供方。跨环境身份对应的 MMP-Retrieval 标注需要在申请中另行提出，限定 TRAIN/VAL。

- [MMPTRACK 官方说明](https://iccv2021-mmp.github.io/subpage/dataset.html)
- [官方使用协议 Word 文件](https://iccv2021-mmp.github.io/attachments/MMPTracking%20User%20Agreement.docx)
- 官方申请邮箱：iccv2021mmp@outlook.com
- 本仓库提供未发送的申请草稿：`docs/WWW_JEV_MMPTRACK_ACCESS_REQUEST_DRAFT.txt`。姓名、机构等使用占位符，不替研究者声明或签署。

可向提供方询问是否能先取得小量 TRAIN/VAL 人工标注、划分清单和必要格式说明，避免为初次资格审计下载完整图像库。申请可提出本地≤64MiB标注审计需求，但不能假定提供方已支持这种交付方式。若只能提供大归档，先评估正常授权下的取数方式与资源，再处理；不自动突破既有预算。

SCOUT 是备选。官方正常下载流程要求登记研究者信息、验证邮箱和下载授权；标注包页面标称约840MB。实际访问与人工标注角色仍需核实，不能把其自动伪标签当作严格监督。

本轮没有发送邮件、提交注册、代填个人信息、签署协议或下载数据。用户确认没有副本后，执行状态继续为BLOCKED_EVIDENCE；科学状态仍是PIVOT_REQUIRED。获得正常授权与可访问来源后，先核查TRAIN/VAL角色、GT、时间戳、人物/采集依赖，冻结文件和有界实验协议；数据获准不自动开启训练、官方TEST、Full24或VisionTrack20/21/22。
