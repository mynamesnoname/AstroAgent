"""
astro_agents.py
天文光谱分析多代理系统定义
"""

import json
import matplotlib.pyplot as plt

from .context_manager import SpectroContext
from datetime import datetime

current_datetime = datetime.now()

from .utils import user_query, _detect_axis_ticks, _detect_chart_border, _crop_img, _remap_to_cropped_canvas
from .utils import _pixel_tickvalue_fitting, _process_and_extract_curve_points, _convert_to_spectrum, _find_features_multiscale
from .utils import _plot_spectrum, _plot_features


# ---------------------------------------------------------
# 1. Visual Assistant — 负责图像理解与坐标阅读
# ---------------------------------------------------------

class SpectralVisualInterpreter:
    
    """
    SpectralVisualInterpreter
    用于从科学光谱图中自动提取坐标轴刻度信息、边框、像素映射等。
    可用于 LangChain 或 MCP Agent 环境。
    """


    def __init__(self, vis_llm, main_llm):
        """
        vis_llm: 视觉 LLM (如 GPT-4V)
        main_llm: 主 LLM，用于逻辑推理
        ctx: SpectroContext 实例，存储当前任务上下文
        """
        self.vis_llm = vis_llm
        self.main_llm = main_llm

    async def detect_axis_ticks(self, ctx: SpectroContext):
        image_path = ctx.image_path
        """调用视觉 LLM 检测坐标轴刻度"""
        prompt = """
你是一个专业的视觉分析模型，擅长从科学图表中提取坐标轴刻度信息。
如果输入中不包含图像，请输出 “未输入图像”
如果输入中包含图像，请分析这张图的 x 轴和 y 轴。

严格按照以下 JSON Schema 输出：
{
"x_axis": {
"label and Unit": "str",
"tick_range": {"min": float, "max": float},
"ticks": [float, float, ...]
},
"y_axis": {
"label and Unit": "str",
"tick_range": {"min": float, "max": float},
"ticks": [float, float, ...]
}
}
"""
        message = user_query(prompt, image_path)
        response = await self.vis_llm.ainvoke([message])
        ctx.set("axis_info", response.content)
        return response.content


    async def combine_axis_mapping(self, ctx: SpectroContext):
        """结合视觉结果与 OCR 结果生成像素-数值映射"""
        axis_info_json = json.dumps(ctx.axis_info, ensure_ascii=False)
        ocr_detected_ticks_json = json.dumps(ctx.OCR_detected_ticks, ensure_ascii=False)
        prompt = f"""
我们正在阅读一个科学图表的数轴刻度。
视觉模型给出的刻度结果是：
{axis_info_json}

OCR/Opencv 工具给出的刻度结果是：
{ocr_detected_ticks_json}

请你综合比对两组结果，生成最终的“刻度值-像素位置”映射表。请严格遵循以下规则：

1. 刻度值必须与视觉模型给出的 ticks 匹配，不要遗漏。  
2. 像素位置必须满足单调性：  
- 对于 y 轴（axis=0），数值从小到大时，像素的 position_y 必须严格从大到小递减；  
- 对于 x 轴（axis=1），数值从小到大时，像素的 position_x 必须严格递增。  
3. 如果 OCR 给出的像素位置与单调性冲突，请修正为合理的插值结果。  
4. 去掉重复和明显错误的刻度值。
5. 对缺失的刻度，用 null 补齐 position_x/position_y；对应的 bounding-box-scale_x / bounding-box-scale_y 用 null 填充。  
6. 计算 sigma_pixel：  
- 对 x 轴：sigma_pixel = bounding-box-scale_x / 2  
- 对 y 轴：sigma_pixel = bounding-box-scale_y / 2  
- 如果 bounding-box-scale 缺失，则 sigma_pixel = null  
7. 给出 conf_llm：  
- 原始 OCR 结果可信度高时 conf_llm = 0.9  
- 插值或修正结果 conf_llm = 0.7  
- 缺失但视觉模型预测到的刻度 conf_llm = 0.5  

请严格输出 JSON 数组（array）：
- 数组中的每个元素必须是一个字典。
- 每个对象必须包含以下字段：
"axis"（对于 y 轴：axis='y'，对于 x 轴：axis='x'）, "value", "position_x", "position_y", 
"bounding-box-scale_x", "bounding-box-scale_y", 
"sigma_pixel", "conf_llm"。

不要输出任何解释或额外文字。
"""

        message = user_query(prompt)
        response = await self.main_llm.ainvoke([message])
        ctx.set("tick_pixel_raw", response.content)
        return response.content
    
    async def revise_axis_mapping(self, ctx: SpectroContext):
        axis_mapping_json = json.dumps(ctx.tick_pixel_raw, ensure_ascii=False)
        prompt = f"""
你是一个科学图表阅读助手。
综合视觉大模型和OCR识别结果，检测得到的科学图表的两轴的信息为

{axis_mapping_json}

请检查x轴和y轴的刻度值和像素位置的匹配关系。
具体来说：
   - 对于 y 轴，数值从小到大时，像素的 position_y 必须严格从大到小递减；  
   - 对于 x 轴，数值从小到大时，像素的 position_x 必须严格递增。  
允许存在为 null 的数值。

如果没有问题，请直接输出原来输入的信息。
如果有问题，请输出你修订好的数据。按原格式输出。
不要输出任何解释或额外文字。
"""

        message = user_query(prompt)
        response = self.main_llm.invoke([message])
        ctx.set('tick_pixel_raw', response.content)
        return response.content

    async def run(self, ctx: SpectroContext, plot: bool=True):
        """
        执行完整视觉分析流程
        """
        # 1. 调用视觉模型提取坐标信息
        await self.detect_axis_ticks(ctx)

        # 2 使用OCR/Opencv识别坐标轴刻度和像素位置
        OCR_detected_ticks = _detect_axis_ticks(ctx.image_path)
        ctx.set('OCR_detected_ticks', OCR_detected_ticks)

        # 3. 综合 1 和 2 的结果，给出一个刻度值-像素位置的匹配表
        await self.combine_axis_mapping(ctx)

        # 4. 检查并修正 3 的错误
        await self.revise_axis_mapping(ctx)

        # 5. 检测边框
        chart_border = _detect_chart_border(image_path=ctx.image_path)
        ctx.set('chart_border', chart_border)

        # 6. 裁剪图像
        _crop_img(ctx.image_path, ctx.chart_border, ctx.crop_path)

        # 7. 重新定标
        remapping = _remap_to_cropped_canvas(ctx.tick_pixel_raw, ctx.chart_border)
        ctx.set('tick_pixel_remap', remapping)

        # 8. 拟合数值-像素关系
        pixel_to_value = _pixel_tickvalue_fitting(ctx.tick_pixel_remap)
        ctx.set('pixel_to_value', pixel_to_value)

        # 9. 灰度化/二值化
        curve_points, curve_gray_values = _process_and_extract_curve_points(ctx.crop_path)
        ctx.set('curve_points', curve_points)
        ctx.set('curve_gray_values', curve_gray_values)

        # 10. 将像素还原成光谱
        spectrum = _convert_to_spectrum(ctx.curve_points, ctx.curve_gray_values, ctx.pixel_to_value)
        ctx.set('spectrum', spectrum)

        # 11 检测峰值

        peaks = _find_features_multiscale(ctx, feature="peak", sigma_list=[2,4,16], prom=0.01, tol_pixels=3)
        troughs = _find_features_multiscale(ctx, feature="trough", sigma_list=[2,4,16], prom=0.01, tol_pixels=3)
        ctx.set('peaks', peaks)
        ctx.set('troughs', troughs)

        if plot:
            a = _plot_spectrum(ctx)
            b = _plot_features(ctx)
            ctx.set('spectrum_fig', a)
            ctx.set('features_fig', b)
        return 0
    


# ---------------------------------------------------------
# 2. Rule-based Analyst — 负责基于规则的物理分析
# ---------------------------------------------------------
class SpectralRuleAnalyst:
    """规则驱动型分析师：基于给定的物理与谱线知识进行定性分析"""

    def __init__(self, agents):
        self.main_agent = agents['main']
        self.vis_llm = agents['vis']

    
    async def describe_spectrum_picture(self, ctx: SpectroContext):
        prompt = f"""
你是一位经验丰富的天文学光谱分析助手。

你将看到一条天文光谱曲线（来自未知红移的天体）。

请结合图像，**定性地描述光谱的整体形态**，包括但不限于以下几个方面：

---

### Step 1: 连续谱形态
- 整体的通量分布趋势（例如蓝端增强 / 红端增强 / 大致平坦 / 呈拱形等）。
- 是否可以看出幂律型连续谱、黑体型谱或平坦谱的特征。
- 连续谱中是否存在明显的断裂或折点（例如巴尔末断裂、Lyα forest 区域等）。

### Step 2: 主要发射与吸收特征
- 是否存在突出的发射峰或吸收谷。
- 发射线（或吸收线）的大致数量与相对强弱。
- 这些线是宽的、窄的、对称的还是不规则的。
- 请避免给出具体数值（如精确波长或通量），只需说明它们相对的位置与特征。

### Step 3: 整体结构与噪声特征
- 光谱信噪比的总体印象（高 / 中 / 低）。
- 是否存在噪声波动、异常尖峰或数据缺口。
- 光谱在长波端或短波端的质量变化情况。

---

⚠️ **注意：**
- 请不要输出任何精确数值或表格；
- 不要尝试计算红移；
- 不要调用工具；
- 重点在“视觉与形态描述”，像一个人类天文学家初看光谱时的定性印象。

最后，请以结构化的方式输出你的观察结果，例如使用分节标题：
-（连续谱）
-（发射与吸收）
-（噪声与数据质量）
"""
        
        message = user_query(prompt, ctx.image_path)
        response = await self.vis_llm.ainvoke([message])
        ctx.set("visual_interpretation", response.content)
        return response.content
    
    async def preliminary_classification(self, ctx: SpectroContext) -> str:
        """初步分类：根据光谱形态初步判断天体类型"""

        visual_interpretation_json = json.dumps(ctx.visual_interpretation, ensure_ascii=False)
        prompt = f"""
你是一位经验丰富的天文学光谱分析助手。

你将看到一条天文光谱曲线（来自未知红移的天体），它可能属于以下三类之一：
- **Star（恒星）**：连续谱较强，谱线通常是吸收线（如 Balmer 系列、金属线等），几乎没有明显红移。
- **Galaxy（星系）**：有一定红移，常见发射线或吸收线（如 [O II], Hβ, [O III], Hα），谱线较窄，连续谱相对较弱。
- **QSO（类星体/类星体候选）**：强烈的宽发射线覆盖可见/紫外波段，谱线宽度显著大于普通星系，通常有明显红移。

前一位天文学助手已经定性地描述了光谱的整体形态：

{visual_interpretation_json}

请根据他的描述进行判断，猜测该光谱可能属于哪一类或几类，给出置信度。

只输出中等置信度以上的回答。

你的回答格式请严格遵循：

猜测 1：
- **类别**: Star / Galaxy / QSO （三选一）
- **理由**: 用简洁的语言解释分类原因（如谱线宽度、红移特征、连续谱形态）
- **置信度**: 高 / 中 / 低
猜测 2：
- **类别**: Star / Galaxy / QSO （三选一）
- **理由**: 用简洁的语言解释分类原因（如谱线宽度、红移特征、连续谱形态）
- **置信度**: 高 / 中 / 低
等等。

⚠️ **注意：**
- 请不要输出任何精确数值或表格；
- 不要尝试计算红移；
- 不要调用工具；
- 重点在“视觉与形态描述”，像一个人类天文学家初看光谱时的定性印象。
"""

        message = user_query(prompt, ctx.image_path)
        response = self.vis_llm.invoke([message])
        # print(response.content)
        ctx.set('preliminary_classification', response.content)
        return 0

    def _common_prompt_header(self, ctx, include_rule_analysis=True):
        """构造每个 step 公共的 prompt 前段"""
        visual_json = json.dumps(ctx.visual_interpretation, ensure_ascii=False)
        peak_json = json.dumps(ctx.peaks[:10], ensure_ascii=False)
        trough_json = json.dumps(ctx.troughs, ensure_ascii=False)

        header = f"""
你是一位天文学光谱分析助手。

以下信息可能来自于一个未知红移的 QSO 光谱。

之前的助手已经对这个光谱进行了初步描述：
{visual_json}
"""

        if include_rule_analysis and ctx.rule_analysis:
            rule_json = json.dumps("\n".join(str(item) for item in ctx.rule_analysis), ensure_ascii=False)
            header += f"\n之前的助手已经在假设光谱中存在 lyα 谱线的情况下进行了初步分析:\n{rule_json}\n"

        header += f"""
综合原曲线和 sigma=2、sigma=4、sigma=16 三条高斯平滑曲线，使用 scipy 函数进行了峰/谷识别。
关于峰/谷的讨论以以下数据为准：
- 代表性的前 10 条发射线：
{peak_json}
- 可能的吸收线：
{trough_json}
"""
        return header

    def _common_prompt_tail(self, step_title, extra_notes=""):
        """构造每个 step 公共尾部，保留 step 特有输出/分析指示"""
        tail = f"""
---

输出格式为：
{step_title}
...

---

🧭 注意：
- 计算得来的非原始数据，最终保留3位小数。
- 不需要进行重复总结。
- 不需要逐行地重复输入数据；
- 重点在物理推理与合理解释；
- 请保证最终输出完整，不要中途截断。
"""
        if extra_notes:
            tail = extra_notes + "\n" + tail
        return tail
    
    async def step_1(self, ctx):
        header = self._common_prompt_header(ctx, include_rule_analysis=False)
        tail = self._common_prompt_tail("Step 1: Lyα 分析")

        prompt = header + """
请按以下步骤分析:

Step 1: Lyα 谱线检测
假设该光谱中存在 Lyα 发射线（λ_rest = 1216 Å）：
1. 找出最可能对应 Lyα 的观测发射线（从提供的峰列表中选择）。
2. 输出：
   - λ_obs (观测波长)
   - 光强（可取相对强度或定性描述）
   - 线宽（FWHM 或像素宽度近似）
3. 使用工具 calculate_redshift 计算基于该发射线的红移 z。
4. 检查蓝端（短波长方向）是否存在 Lyα forest 特征：  
   若吸收线相对更密集、较窄且分布在 Lyα 蓝端附近，请指出并给出简短说明。
""" + tail
        
        # messages = user_query(prompt, ctx.image_path)
        messages = user_query(prompt)
        response = await self.main_agent.ainvoke({"messages": messages}, config={"recursion_limit": 75})
        ctx.append('rule_analysis', response['messages'][-1].content)

########################################

#     async def step_1_5(self, ctx):
#         header = self._common_prompt_header(ctx, include_rule_analysis=False)
#         tail = self._common_prompt_tail("Step 1.5: Lyα forest 检测")

#         prompt = header + """
# 请按以下步骤分析:

# Step 1.5: Lyα forest 检测
# 1. 检查蓝端（短波长方向）是否存在 Lyα forest 特征：  
#    若吸收线相对更密集、较窄且分布在 Lyα 蓝端附近，请指出并给出简短说明。
# """ + tail
        
#         message = user_query(prompt, ctx.image_path)
#         response = self.vis_agent.invoke([message])
#         ctx.append('rule_analysis', response.content)

#########################################

    async def step_2(self, ctx):
        header = self._common_prompt_header(ctx)
        tail = self._common_prompt_tail("Step 2: 其他显著发射线分析")

        prompt = header + """
请继续分析:

Step 2: 其他显著发射线分析
1. 以 Step 1 得到的红移为标准，使用工具 predict_obs_wavelength 检查光谱中是否可能存在其他显著发射线（如 C IV 1549, C III] 1909, Mg II 2799, Hβ, Hα 等）。不要自行计算。
2. 还有什么需要注意的发射线？
""" + tail

        response = self.main_agent.invoke({"messages": prompt}, config={"recursion_limit": 75})
        ctx.append('rule_analysis', response['messages'][-1].content)

    async def step_3(self, ctx):
        header = self._common_prompt_header(ctx)
        tail = self._common_prompt_tail("Step 3: 综合判断")

        prompt = header + """
请继续分析:

Step 3: 综合判断
- 在 Step 1 到 Step 2 中，如果 Lyα 的存在证据不足（例如对应波长没有明显峰值或红移与其他谱线不一致），请**优先假设 Lyα 不存在**，并结束分析。  
- 仅在 Lyα 的存在有充分证据（显著峰值 + 红移与其他谱线一致）时，才将 Lyα 纳入综合红移计算。
- 如果 Step 1 和 Step 2 的红移计算结果一致，请综合 Step 1 到 Step 2 的分析，使用 Step 1 和 Step 2 得到的谱线匹配，给出：
    - 各个谱线的红移
    - 由各谱线在 sigma=2 平滑下的强度 flux 作为权重，使用工具 weighted_average 进行加权平均，输出得到的加权红移值 z
    - 可能的红移范围 Δz
    - 涉及计算红移的流程必须使用工具 calculate_redshift，不允许自行计算。
- 给出该红移下，你能确定的各个发射线的波长和发射线名。
""" + tail

        response = self.main_agent.invoke({"messages": prompt}, config={"recursion_limit": 100})
        ctx.append('rule_analysis', response['messages'][-1].content)

    async def step_4(self, ctx):
        header = self._common_prompt_header(ctx)
        tail = self._common_prompt_tail("Step 4: 补充步骤（假设 lyα 不存在时的主要谱线推测）")

        prompt = header + """
请继续分析:

Step 4: 补充步骤（假设最高发射线不是 lyα 时的主要谱线推测）
- 根据 QSO 的典型谱线特征，找出光谱中**强度最高的峰值**。
- 猜测该峰值可能对应的谱线（例如 C IV, C III], Mg II, Hβ, Hα 等）。
- 仿照 Step1-3 的逻辑进行判断。涉及红移计算的请使用工具 calculate_redshift；涉及观测线波长计算的请使用工具 predict_obs_wavelength。不允许自行计算。
    - 输出该峰对应谱线的信息：
        - 谱线名
        - λ_obs
        - 光强
        - 谱线宽度
        - 根据 λ_rest 初步计算红移 z。不允许自行计算。
    - 如果可能，推测其他可见发射线，并计算红移
    - 综合所有谱线，给出最可能的红移和红移范围
- 以上判断是否支持 lyα 不存在的假设？
""" + tail

        response = self.main_agent.invoke({"messages": prompt}, config={"recursion_limit": 75})
        ctx.append('rule_analysis', response['messages'][-1].content)

    async def run(self, ctx):
        await self.step_1(ctx)
        # await self.step_1_5(ctx)
        await self.step_2(ctx)
        await self.step_3(ctx)
        await self.step_4(ctx)



# ---------------------------------------------------------
# 3. Revision Supervisor — 负责交叉审核与评估
# ---------------------------------------------------------
class SpectralAnalysisAuditor:
    """结果监督者：审查并校正其他分析 agent 的输出"""

    def get_system_prompt(self) -> str:
        return f"""
你是一位严谨的【天文学光谱报告审查官】。

任务目标：
- 审核其他 agent 的光谱分析报告
- 识别其中的逻辑漏洞、计算漏洞、不一致或错误推断
- 提出修正意见或补充分析方向

工作原则：
- 保持客观与批判性思维
- 不重复原分析，只指出问题与改进建议
- 若原报告合理，应明确确认其有效性

输出要求：
- 简明列出审查意见（例如：“结论偏早”，“谱线解释正确”）
- 对每个发现附上改进建议
- 最后给出整体评价（可靠/部分可信/不可信）
"""


# ---------------------------------------------------------
# 4. Reflective Analyst — 自由回应审查并改进
# ---------------------------------------------------------
class SpectralRefinementAssistant:
    """改进者：回应审查并改进分析"""

    def get_system_prompt(self) -> str:
        return f"""
你是一位具备反思能力的【天文学光谱再分析师】。

任务目标：
- 阅读并理解他人的光谱分析报告
- 阅读并理解审查官提出的反馈
- 对自身或他人先前的分析进行改进
- 提出新的解释或修正结论

工作原则：
- 认真回应每条反馈，逐一说明改进之处
- 如果认为原结论正确，需给出充分理由
- 最终输出一个更严谨、完善的分析版本

输出要求：
- 列出收到的反馈及对应回应
- 提供改进后的光谱分析总结
- 说明修改内容及其科学合理性
"""


# ---------------------------------------------------------
# 🧩 5. Host Integrator — 汇总与总结多方观点
# ---------------------------------------------------------
class SpectralSynthesisHost:
    """汇总主持人：整合多Agent的分析与结论"""

    def get_system_prompt(self) -> str:
        return f"""
你是一位负责统筹的【天文学光谱分析主持人】。

任务目标：
- 汇总视觉分析师、规则分析师、审查官和再分析师的所有输出
- 综合不同角度的结论，形成最终的光谱解释
- 清楚指出各方意见的差异与一致点

工作原则：
- 不盲从任何单一分析
- 保持整体科学性与逻辑一致性
- 最终输出必须具备可追溯性（说明来自哪些agent的依据）

输出要求：
- 简明概述每个分析角色的主要观点
- 指出关键的共识与分歧
- 给出最终综合结论及可信度评级（高/中/低）
- 如果仍存在不确定性，请明确指出
"""
