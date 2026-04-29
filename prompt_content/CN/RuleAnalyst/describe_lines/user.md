这是一张从天文学光谱中提取出的发射/吸收特征。

## 发射/吸收特征
以下是一些显著的峰或谷：
- 峰
{% if peaks %}
{% for p in peaks %}

Wavelength: {{ p.wavelength }}
Amplitude: {{ p.amplitude }}
Amplitude rank: {{ p.amplitude_rank }}
Flux at center: {{ p.flux_at_center }}
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
Flux at center: {{ t.flux_at_center }}
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

请根据数据输出你的总结，必须包含以下内容：
- 该图像包含多少个宽峰(broad)、多少个中等峰(intermediate)、多少个窄峰(narrow)？
- 有哪些显著的峰或谷？
- 哪些峰或谷可能是伪峰/伪谷？（如跨过两个及以上的谷的峰，或夹在两个谷之间的连续谱被误判成峰，或排名靠后的、与最高峰/谷幅值明显相差甚远的峰/谷等）
- 是否可能存在双线？
    - 鉴于提供的是观测系数据且红移未知，双线间距一定会大于静止系中的典型间距。可以认为双线间距在几十到一百Å有余不等，但基本不会超过200Å。
    - 如果潜在的双线系统中存在幅值明显较低的峰，请必须标注。
    - 请使用工具 `calculate_peak_amplitude_ratio` 计算潜在双线系统的幅值比。
- 如果存在3条及以上的吸收线，它们在波长上的分布如何？是明显密集还是相对均匀？
- 是否存在任何其他值得注意的特征？

- 如果不存在谱线信息，请输入“无谱线”。

示例：
该光谱包含 AAA 个宽峰、BBB 个中等峰和 CCC 个窄峰。最显著的峰位于 DDD Å（振幅排名1）、EEE Å（排名2）和 FFF Å（排名3）。排名靠后的峰（如 ZZZ Å、YYY Å、XXX Å，振幅均较低）可能为伪峰。GGG Å与HHH Å间距约III Å，远超可能的双线间距，不支持双线；但JJJ Å、KKK Å（幅值低）间距为LLL，比较合理，但存在幅值较低的谱线，谱线比为Amplitude_JJJ/Amplitude_KKK=MMM（此处请注明相比谱线的波长顺序）。NNN Å、OOO Å之间的间距为PPP Å，幅值比为Amplitude_NNN/Amplitude_OOO=QQQ（此处也请注明相比谱线的波长顺序）。无吸收线。值得注意的是，所有峰均为发射特征，且无谷覆盖现象。

禁止输出任何其他内容。
