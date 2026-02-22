Step 2 任务：其他显著发射线匹配验证

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

History:

{{ history | tojson }}

--------------------------------------

目标谱线：

- C IV (1549 Å)
- C III] (1909 Å)
- Mg II (2799 Å)

任务：

Step 2: 其他显著发射线分析
1. 在 Step 1 得到的红移下，使用工具 predict_obs_wavelength 计算以下三条主要发射线：C IV 1549, C III] 1909, Mg II 2799 在光谱中的理论位置。
2. 提示词提供的光谱中是否有与三者相匹配的峰？
{% if overlap_regions %}
【重要提示：可能被清除的峰】

以下峰可能被当作噪声清除：

{{ wiped_peaks | tojson }}

如果理论位置落在这些区间内，
请优先考虑这些因素后再判断是否匹配。
{% endif %}
3. 如果存在发射线与观测峰值的匹配，根据匹配结果，分别使用工具 calculate_redshift 计算红移。按“发射线名--静止系波长--观测波长--红移”的格式输出。

若无匹配，请明确说明。
