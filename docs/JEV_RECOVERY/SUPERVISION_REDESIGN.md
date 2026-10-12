# R2 两类真实问题

Cross：其他摄像头过去有至少3个已知、至少80%已知覆盖且完全同人的支持，是否与当前人一致；不再要求整个 Global ID 的混合历史先认证为纯 owner。Continuity：具体 ID 的已观测归属是否支持当前人的连续性；充分观测到多个不同人认证的是“连续归属违例”，并不认证正确 owner。纯 ID 与当前人矛盾为0；纯 ID 且 own-camera 历史支持当前人为1；其余 UNKNOWN。

这些是不同可验证谓词，不修改 GT、不把 owner UNKNOWN 当错误、不把历史可靠性概率称为已校准未来安全。关联 known_options 和 changed-row 比较仍使用原冻结标签；混合 ID 无法形成可靠关联排名的限制仍存在。每个视频分歧及依赖组见 R2_QUESTION_SUPERVISION.json。
