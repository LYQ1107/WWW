# WWW / JEV 公共发布资料核对

这轮完成公开代码、发布页面和聚合元数据审计。没有读取新数据集的标注内容、图像、压缩包正文或官方 TEST；没有训练、原生干预或 GPU 实验。源码事实在发布审计协议前已经部分发现，因此不把本轮称为性能预注册实验。

| 来源 | 已核实的发布事实 | 对下一步的影响 |
|---|---|---|
| MMPTRACK | 官方给出 TRAIN/VAL/TEST 参与者数量及研究用途的签署访问流程 | 若有合法副本，优先核查 TRAIN/VAL；还需检查环境内 ID 和跨环境 retrieval ID 的范围 |
| WILDTRACK | GT 为 400 帧、2 Hz；释放图像为 10 Hz；原始视频为 60 Hz | 不插值增加严格监督，不把七相机视为七个独立录像；官方完整包 HEAD 大小 6,807,496,358 字节，本轮未下载正文 |
| SCOUT | 发布公告仅对 sequence_1 前八相机提供完整标注，其余是自动伪标签；网页转换版本元数据为 10 Hz | 完整标注范围不能自动认证为独立监督；转换版本频率不等于原始归档时间戳 |

UMPN 的 SCOUT 示例实际在名为 `scoutval` 的 sequence_1 相机 0–7 上训练，再做 90/10 时间切分；WILDTRACK 示例也做时间切分。仓库命名不能证明参与者或录像独立。这里没有读取真实身份标签，不能据此宣称已经发现人员重叠。窗口边界是否交叉也仍需实际接口审计。

另一个不能照搬的流程是缺失归一化统计时的全场景扫描。新适配器必须仅在冻结 TRAIN 拟合统计量，并把标注世界坐标、GT 补检测等信息留在离线监督侧。SCOUT 官方 COCO 读取器将 category_id 当作 track_id；必须核验真实身份字段、逐相机时间戳和缺失标注，不能采用通用单类别 COCO 假设。

下一范围需要合法标签来源和明确的独立 cohort。MMPTRACK 优先，但本地合法副本尚未确定。WILDTRACK 和 SCOUT 的公开页面不能直接保证需要的 8 TRAIN / 3 RESERVED 独立安全组。若后续考虑跨数据集 TRAIN/RESERVED，必须先确认官方角色、授权、标注质量及身份依赖，完整冻结后才能读取和采样；这只是设计候选，不是合格结论。

现有安全监督仍为 3 TRAIN / 1 RESERVED，学习资格不变。公开资料审计完成不证明 JEV 提升，也不完成 MASTER GOAL。

详细来源、固定提交、SHA256、逐项断言位置和未执行项见 `reports/JEV_MASTER/PUBLIC_RELEASE_METADATA_AUDIT.json`。第三方源码和网页快照仅留本地，总计 1138570 字节；GitHub 只提交审计脚本、协议、摘要和结果。

官方来源：[MMPTRACK](https://iccv2021-mmp.github.io/subpage/dataset.html)、[WILDTRACK](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)、[SCOUT](https://scout.epfl.ch/)、[SCOUT 配置](https://github.com/cvlab-epfl/UMPN/blob/0f86307f49314a342dcaa7179647f7f126362158/configs/files/config_scout.yaml)、[分割函数](https://github.com/cvlab-epfl/UMPN/blob/0f86307f49314a342dcaa7179647f7f126362158/datasets/utils.py)、[场景加载器](https://github.com/cvlab-epfl/UMPN/blob/0f86307f49314a342dcaa7179647f7f126362158/datasets/sceneset.py)。
