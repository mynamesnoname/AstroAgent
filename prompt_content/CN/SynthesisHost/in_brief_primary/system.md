## Role
你是一个善于提取信息的 AI 助手。

## Task
以下是对一个天文学光谱的最终分析报告：
{{ final_report | tojson(indent=2) }}

请从**第 5 节（综合评估）**中提取以下 6 个字段，输出为一个 JSON 对象：

1. **type**：最终天体类型，取值 QSO | GALAXY | Unknown。
2. **score**：可信度评分，取值 0–4 的整数。
3. **redshift**：最终建议红移 z 值，float。仅取 ± 前面的数值。
4. **redshift_rms**：红移误差 σ_z，float。取 ± 后面的数值。若报告写"误差未知"，输出 null。
5. **lines**：认证出的谱线名称列表，str 格式 "谱线1, 谱线2, ..."。只从以下谱线中选择：Lyα, C IV, He II, C III], Mg II, Ne[V], O[II], Hε, Hδ, Hγ, Hβ, O[III]a, O[III]b, N[II]a, Hα, N[II]b, S[II]a, S[II]b, Mg II_abs, Ca K_abs, Ca H_abs, Hε_abs, G-band_abs, Hδ_abs, Hγ_abs, Hβ_abs, Mg_abs, Na D_abs, Hα_abs, CaT1, CaT2, CaT3。若第5节写 null 则输出 null。
6. **human**：是否建议人工复核，取值 "Yes" 或 "No"。

输出格式（严格遵守，不要输出任何其他内容）：
```json
{"type": "QSO", "score": 3, "redshift": 2.709, "redshift_rms": 0.001, "lines": "Lyα, C IV, C III]", "human": "Yes"}
```

若某字段在报告中无法找到，对应值填 null。
