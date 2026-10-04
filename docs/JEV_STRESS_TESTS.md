# JEV 压力测试执行入口

结构性压力测试：

```bash
PYTHONPATH=/data1/liuyeqiang/WWW:/data1/liuyeqiang/WWW/third_party/CenterNet2:/data1/liuyeqiang/WWW/reproduction_tools \
  /home/liuyeqiang/anaconda3/envs/GMT/bin/python reproduction_tools/jev_stress_tests.py \
  --output outputs/jev_stress_contract.json
```

它覆盖 same-score/different-state、legal mask、action permutation、固定阈值边界、runtime 无 future-GT import 和一个仅用于结构检查的 ECE 计算。真实 action accuracy、HOTA/IDF1、contamination、recovery 和 risk-coverage 必须在 Stage2 冻结后由真实 counterfactual/online 记录填写；此脚本的 PASS 不等于 JEV tracking 结果 PASS。

