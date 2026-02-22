对该光谱连续谱的定性描述如下：

{{ continuum_description | tojson }}

--------------------------------------------------
连续谱分类规则：

{% if dataset == "CSST" %}
- 蓝端高、红端低（下降趋势）→ QSO
- 蓝端低、中段高、红端下降（上升→下降）→ QSO
- 蓝端低、红端高（上升趋势）→ Galaxy
{% else %}
- 蓝端高、红端低 → QSO
- 蓝端低、红端高 → Galaxy
{% endif %}

--------------------------------------------------
信噪比规则：

该光谱的最大信噪比为 {{ snr_max }}。

{% if snr_threshold_upper %}
- 当最大信噪比 > {{ snr_threshold_upper }} 时，必须输出 “QSO” 或 “Galaxy”。
- 当 {{ snr_threshold_lower }} < 最大信噪比 ≤ {{ snr_threshold_upper }} 时，
  可以从 “QSO”、“Galaxy”、“Unknown” 中选择。
  - 信噪比越接近 {{ snr_threshold_upper }}，越倾向于明确分类；
  - 信噪比越低，越倾向于选择 “Unknown”。
- 当最大信噪比 ≤ {{ snr_threshold_lower }} 时，必须输出 “Unknown”。
{% else %}
（未提供信噪比阈值，仅依据连续谱判断）
{% endif %}

--------------------------------------------------

请输出你的判断过程和光谱类别
