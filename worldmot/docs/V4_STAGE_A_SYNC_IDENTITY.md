# 阶段 A：公平同步身份核验

实际状态：`COMPLETE`。四套既有完整 TEST 轨迹逐文件核验：每套22场、29,019同步时刻。原始预测和GT均未修改。本轮没有运行 V4 在线 TEST，也没有利用 TEST 选择参数。

GT按每摄像头IoU≥0.5执行“先最大匹配数，再最大总IoU”的一对一配对；重复、非法、忽略和人群GT显式屏蔽。Protocol A用各方法自身可评估事件，Protocol B用与GMT共同的GT事件，并额外检查四方法共同交集。此次四套检测匹配事件完全相同，三种协议数值一致。

原始GT合法14场：1/2/3/4/6/7/8/9/10/11/15/16/18/19；共有54,463个同人配对、287,110个异人配对。GT阻塞8场另存干净子集诊断，不能混入正式合法范围。

| 方法 | TP同人 | FN拆分 | FP误并 | Precision % | Recall % | F1 % | False Merge Rate % |
|---|---:|---:|---:|---:|---:|---:|---:|
| GMT Native | 47133 | 7330 | 2744 | 94.4985 | 86.5413 | 90.3450 | 0.9557 |
| R2 Full + Ω | 45004 | 9459 | 893 | 98.0543 | 82.6322 | 89.6851 | 0.3110 |
| R2 Full No-Geometry | 42795 | 11668 | 1512 | 96.5874 | 78.5763 | 86.6559 | 0.5266 |
| R2 Head-Only | 41953 | 12510 | 401 | 99.0532 | 77.0303 | 86.6645 | 0.1397 |

Full对GMT：同人纠正2,626、回归4,755，净−2,129；异人纠正2,528、回归677，净+1,851。Full更少误并，但拆分更多，F1也低于GMT，因此不能宣称整体超过GMT。Full相对单独训练No-Geometry多正确合并2,209对同人并少619次异人误并；这是已训练系统差异，几何因果仍须阶段B冻结控制。

历史清空诊断：Full当前同步决策TP45,223/FP1,444；在线Full为TP45,004/FP893。当前帧收益与历史保守性存在取舍。固定Full头、去几何输入时TP37,131/FP1,834；单独训练No-Geometry当前决策TP43,429/FP2,626。前者只说明固定头的输入敏感性，不能当作无几何模型训练比较，也不是在线MC-MOT。

| Full相对GMT的固定分层 | 配对数 | 纠正 | 回归 | 净纠正 |
|---|---:|---:|---:|---:|
| same_person/both_geometry_eligible | 49830 | 2415 | 3277 | -862 |
| same_person/N_lt20 | 48485 | 2225 | 3850 | -1625 |
| same_person/estimated_Omega_camera_axis_angle_ge60deg | 25252 | 1504 | 2335 | -831 |
| different_person/both_geometry_eligible | 179932 | 1700 | 534 | +1166 |
| different_person/N_lt20 | 202575 | 1702 | 543 | +1159 |
| different_person/estimated_Omega_camera_axis_angle_ge60deg | 157754 | 1885 | 523 | +1362 |
| different_person/one_or_both_geometry_ineligible | 107178 | 828 | 143 | +685 |
| same_person/same_person_low_ROI_cosine | 7342 | 369 | 2300 | -1931 |
| same_person/dense_N_ge20 | 5978 | 401 | 905 | -504 |
| different_person/dense_N_ge20 | 84535 | 826 | 134 | +692 |
| same_person/one_or_both_geometry_ineligible | 4633 | 211 | 1478 | -1267 |
| same_person/both_views_N_ge10 | 2680 | 41 | 512 | -471 |
| different_person/both_views_N_ge10 | 26502 | 194 | 44 | +150 |

分层可重叠，不能相加。低ROI相似度并非大视角；相机夹角来自Ω自身姿态估计，并非标定真值。几何可用层仍受系统其它差异影响，不能直接解释为Ω因果收益。低外观相似度同人层净−1,931，拥挤N≥20同人层净−504，显示拆分问题仍明显。

完整逐场指标、双协议及共同交集、歧义掩码和SHA见 `results/v4_gmt_synchronous_identity_comparison.json`；288条有限纠正/失败样例见 `results/v4_crossview_hard_cases.json`。这些是同步配对诊断，不能称为HOTA、IDF1或CVIDF1。
