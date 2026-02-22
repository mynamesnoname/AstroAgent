Step 3 任务：替代谱线假设分析

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

Step 4: 补充步骤（假设 Step 1 所选择的谱线并非 Lyα）
- 请抛开前述步骤的分析内容。考虑 Step 1 所选择的谱线实际上是除 Lyα 外的其他主要发射线。
    - 假设该峰值可能对应的谱线为 C IV：
        - 输出该峰对应谱线的信息：
            - 观测波长 λ_obs
            - 流量 Flux
            - 谱线宽度
            - 根据 λ_rest，使用工具 calculate_redshift 初步计算红移 z
        - 使用工具 predict_obs_wavelength 计算在此红移下的其他主要发射线（如 Lyα C III] 和 Mg II）的理论位置。光谱中是否有与它们匹配的发射线？
        - 如果 Lyα 谱线在光谱范围内，检查其是否存在？
        - 如果存在可能的发射线-观测波长匹配结果，使用工具 calculate_redshift 计算它们的红移。按照“发射线名--静止系波长--观测波长--红移”的格式进行输出
    
    - 若以上假设不合理，则假设该峰值可能对应 C III] 等其他主要谱线，重复推断。如果其他谱线（如 Lyα C III] 和 Mg II）在光谱范围内，检查其是否存在？

