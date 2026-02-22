这是一张从天文学光谱中提取出的连续谱图像。

## 发射/吸收特征
以下是一些显著的峰或谷：
- 峰
{% for p in peaks %}
波长：{{ p.wavelength }}
Flux:  {{ p.mean_flux }}
Prominence: {{ p.max_prominence }}
Width: {{ p.describe }}

{% endfor %}

- 谷
{% for t in troughs %}
波长：{{ t.wavelength }}
Flux:  {{ t.mean_flux }}
Depth: {{ t.max_depth }}
Width: {{ t.describe }}

{% endfor %}

请根据数据和图片总结：
- 宽线（>2000 km/s）/窄线（<1000 km/s）二者哪个占主导？
- 是否存在哪些非对称的峰？

请严格按以下 JSON 格式输出结果：

{
    "dominant_line_type": "宽线/窄线",
    "asymmetric_peaks_wavelength": List[float] 或 null
}

禁止输出任何其他内容。
