本光谱的 camera/filters 名为：
{{ arm_name | tojson }}

下面是光谱在 arms 交界区域的样本数据：

{% for region in overlap_regions %}
交界区域 {{ region.region }}:
波长：{{ region.wavelength | tojson }}
Flux：{{ region.flux | tojson }}
Flux 误差：{{ region.delta_flux | tojson }}

{% endfor %}

请判断：
是否在交界区域出现了明显的、非物理性的噪声增强？

