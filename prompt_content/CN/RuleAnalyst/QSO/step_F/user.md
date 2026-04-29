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

该光谱的发射/吸收特征的具体信息：

- 峰
{% if peaks %}
{% for p in peaks %}

Wavelength: {{ p.wavelength }}
Amplitude: {{ p.amplitude }}
Amplitude rank: {{ p.amplitude_rank }}
Width in Å: {{ p.FWHM_A }}
Width in km/s: {{ p.FWHM_km_s }}
Width class: {{ p.width_class }}
Covered troughs (Is there any trough covered by the peak?): {{ p.covered_troughs }}
Covered trough centers: {{ p.trough_centers }}
Neighbors:
{{ p.left_neighbor }}
{{ p.right_neighbor }}
Does it touch the edge: {{ p.quality_boundary_touch }}
------------------------------------------------------
{% endfor %}
{% else %}
无显著峰特征
{% endif %}
------------------------------------------------------

------------------------------------------------------
- 谷
{% if troughs %}
{% for t in troughs %}

Wavelength: {{ t.wavelength }}
Amplitude: {{ t.amplitude }}
Amplitude rank: {{ t.amplitude_rank }}
Width in Å: {{ t.FWHM_A }}
Width in km/s: {{ t.FWHM_km_s }}
Neighbors:
{{ t.left_neighbor }}
{{ t.right_neighbor }}
-------------------------------------------------------
{% endfor %}
{% else %}
无显著谷特征
{% endif %}

---

对光谱的预分类及特征描述为：

{{ preliminary_classification | tojson }}

---

当前待分析的暴力谱线匹配假设（第 {{ hypothesis_index }} 条，共 {{ hypothesis_total }} 条）：

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

---

请按 Step F-1 到 Step F-3 的顺序，依次完成对当前假设的分析。
