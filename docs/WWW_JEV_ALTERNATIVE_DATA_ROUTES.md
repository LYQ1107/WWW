# 作者公开的数据入口 — 2026-10-12

找到三类优先候选：CvMHATB、MVCrowdTrack、CityTrack项目资源。入口来自作者自己的仓库，不是来源不明的转存。只核对了公开资料和分享页面，没有下载数据集、GT、图像、模型或官方TEST。

| 候选 | 作者给出的入口 | 提取码 | 已知用途 |
|---|---|---|---|
| CvMHATB | https://pan.baidu.com/s/1tw8voHsDp-Fgv0mWim7kgA | MHAT | 无人机＋穿戴式移动摄像头，跨帧、跨视角人物关联标注；这条是数据集主体，另一条2024 metadata分享是方法生成的信息 |
| MVCrowdTrack | https://pan.baidu.com/s/1X168WJ0QqGkFkLiAHQtRmg | 9xbs | CVPR2026多视角行人跟踪，论文说明有人工逐视角框和一致身份 |
| MVTrackTrans首个分享／CityTrack相关资源 | https://pan.baidu.com/s/1pIMExmYV-ttQd8x2Ky3bRA | 1390 | 早期作者README描述CityTrack入口；最新README只称首个分享，目录内容尚未验证 |

MVCrowdTrack／CityTrack另有作者Dropbox资源包，密码iMUSELab：[分享](https://www.dropbox.com/scl/fo/3jjeq5jkw4gmg488tj6pv/APCR-ifra2RxjOuOWdlq1NA?rlkey=uqt65idvb7jzczuqpxyc3tm6u&st=teiy3zq6&dl=0)。它包含论文、代码及数据的说明来自README；目前只观察到密码页面，没有枚举或下载归档。

三个百度入口HTTP返回200并进入提取码页面；没有显示失效提示。但未提交提取码、登录、读取文件列表或验证实际下载，所以不能声称“文件已成功拿到”。未用公开分享链接代替实际数据使用条款核查。

DIVOTrack目前的官方入口为[Hugging Face](https://huggingface.co/datasets/syhao777/DIVOTrack)。最新README明确ZIP加密、需要提交许可证获得解压密码；因此它仍需要正常申请，不列为免申请的优先路线。MMPTRACK和SCOUT在本次有界搜索中没有发现可信的作者认可免申请镜像，不能据此宣称网络上绝对不存在镜像。

其他可直接访问的参考来源是[EPFL多摄像头行人视频](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/)。官网允许科研使用，但部分GT是稀疏网格位置，人物和场景规模较小；它与当前像素框、长期身份动作评价接口不同，不能仅因可下载就认定满足独立安全监督。

## 后续来源核查范围

研究必要性：当前MMPTRACK／SCOUT副本不存在，而既有WILDTRACK不能提供新RESERVED，需要替代人工身份监督人口。按CvMHATB、MVCrowdTrack、CityTrack顺序，先只查作者条款、目录、分割、标注单独访问和大小；公开资料新增留存≤4MiB、0GPU。任何后续GT审计维持≤64MiB，不先下载视频或整个图像库。

划分必须在标签内容、追踪结果出现前冻结。只读作者TRAIN或合法VAL；官方TEST继续封存。CVPR2026补充材料把MVCrowdTrack的3297帧用于TRAIN、825用于TEST，CityTrack前1948帧TRAIN、余640帧TEST。这些时间分割不证明人员或采集独立。CvMHATB划分尚未核实，不能任选序列；还需检查它与已有VisionTrack的来源、人员及图像暴露关系。匹配GT不能替代detector输出，标注世界位置留在离线评价侧。

本次只有来源发现，未解除3TRAIN/1RESERVED、8＋3的资格限制，不开启任何新训练或原生视频实验。作者共享入口的可用性与科学资格是两个不同结论。

来源：[CvMHATB作者仓库](https://github.com/RuizeHan/CvMHATB)、[MVTrackTrans作者仓库](https://github.com/zqyq/MVTrackTrans)、[CVPR2026补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Zhang_Multi-view_Crowd_Tracking_CVPR_2026_supplemental.pdf)、[DIVOTrack作者仓库](https://github.com/shengyuhao/DIVOTrack)。固定源码提交、SHA及页面观察见ALTERNATIVE_OFFICIAL_DATA_ROUTES.json。

## 下载数量约束

本轮新数据集下载数为0，仅查询说明与分享入口。多个名称用于比较来源，并非批量下载计划。先核查CvMHATB条款、目录、TRAIN/VAL/TEST和小量标注访问；一次只推进一个候选，只有不适合时才考虑备选。新GT审计累计≤64MiB，不对每个候选重新分配64MiB，不先取视频或全部图像。不承诺单个尚未核查的数据集能满足8＋3资格。
