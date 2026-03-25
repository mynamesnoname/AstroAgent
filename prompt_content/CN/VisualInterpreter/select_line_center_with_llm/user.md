程序在同一波长附近检测到多个可能属于同一谱线的候选结果。

{% if approved_peaks %}
候选发射线数据如下：
{% endif %}

{% if approved_troughs %}
候选吸收线数据如下：
{% endif %}

{{ group_data | tojson }}

请判断最可能的真实谱线波长。

要求：
1. 分析不同 candidate 的证据
2. 简要说明判断理由
3. 选择唯一一个最合理的 wavelength

请仅返回一个 JSON 对象，不要包含任何额外文本。

输出格式：

{
  "selected_wavelength": float,
  "selected_index": int,
  "reason": "简要说明为什么选择这个候选结果"
}
