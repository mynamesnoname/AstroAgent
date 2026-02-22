Step 1 任务: Lyα 发射线检测与初始红移估计

--------------------------------------

光谱整体描述：
{{ qualitative_analysis | tojson }}

波长范围：
{{ wl_left }} Å – {{ wl_right }} Å

高斯平滑尺度：
{{ sigma_list }}

代表性发射峰：
{{ peak_list | tojson }}

吸收谷：
{{ trough_list | tojson }}

峰值波长相对测量误差：
± {{ tol_wavelength }} Å

--------------------------------------

假设光谱中存在 Lyα 发射线（λ_rest = 1216 Å）。

{% if lyalpha_candidates %}
候选 Lyα 线：
{{ lyalpha_candidates | tojson }}
{% endif %}

1. 在光谱流量较大，大 smoothing 尺度可见且有一定宽度的峰中，推测哪条最可能为 Lyα 线。
    - 从提供的峰列表中选择
    - 候选谱线宽度相近（20 Å 以内）时，优先考虑流量更高的峰。
2. 输出：
- 观测波长 λ_obs
- 流量 Flux
- 谱线宽度
3. 使用工具 calculate_redshift 计算该峰为 Lyα 发射线时的红移 z。
4. 检查蓝端（短波长方向）是否存在 Lyα forest 特征：吸收线相对更密集、较窄且分布在 Lyα 蓝端附近。请指出并进行简短说明。