光谱信息如下：

波长范围：
{{ wl_left }} Å – {{ wl_right }} Å

峰值波长相对测量误差：
± {{ tol_wavelength }} Å

---

对光谱连续谱的定性描述为：
{{ continuum_description | tojson }}

---

对该光谱的发射/吸收特征的定性描述为：

{{ feature_description | tojson }}

---

对光谱的预分类及特征描述为：

{{ preliminary_classification | tojson }}

---

## 各假设分析汇总

共 {{ brute_force_matching | length }} 条假设，每条包含完整匹配数据及对应推理摘要：

{% for match in brute_force_matching %}
---
### 假设 {{ loop.index }}

**完整匹配数据：**

Hypothesis: {{ match.Hypothesis }}
z_max: {{ match.z_max }}  z_min: {{ match.z_min }}  z_spread: {{ match.z_spread }}
N_emission: {{ match.N_emission }}  N_absorption: {{ match.N_absorption }}
Redshift warning: {{ match["Redshift warning"] }}
Emission matches:
{% for em in match["Emission matches"] %}
  {{ em }}
{% endfor %}
Absorption matches:
{% for ab in match["Absorption matches"] %}
  {{ ab }}
{% endfor %}
Missing emission lines (in obs range but not matched):
{% if match["Missing emission lines"] %}
{% for ml in match["Missing emission lines"] %}
  {{ ml }}
{% endfor %}
{% else %}
  (none)
{% endif %}
Observable emission lines (all emission lines within obs range at this redshift, with matching status):
{% if match["Observable emission lines"] %}
{% for ol in match["Observable emission lines"] %}
  {{ ol }}
{% endfor %}
{% else %}
  (none)
{% endif %}

**推理摘要（Step F-a 输出）：**

```json
{{ f_a_summaries[loop.index0] | tojson(indent=2) }}
```

{% endfor %}

---

请按 Step F-b1 到 Step F-b3 的顺序，完成对所有假设的综合评判，给出最终 1-2 个最可能的结论。
