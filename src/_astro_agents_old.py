import json
import os

from .context_manager import SpectroContext
from .base_agent import BaseAgent
from .mcp_manager import MCPManager

from .context_manager import SpectroContext
from .base_agent import BaseAgent
from .mcp_manager import MCPManager

from .utils import (
    _detect_axis_ticks, _detect_chart_border, _crop_img,
    _remap_to_cropped_canvas, _pixel_tickvalue_fitting,
    _process_and_extract_curve_points, _convert_to_spectrum,
    _find_features_multiscale, _plot_spectrum, _plot_features
)

# ---------------------------------------------------------
# 1. Visual Assistant — 负责图像理解与坐标阅读
# ---------------------------------------------------------

class SpectralVisualInterpreter(BaseAgent):
    """
    SpectralVisualInterpreter
    从科学光谱图中自动提取坐标轴刻度、边框、像素映射等信息
    """

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Visual Interpreter',
            mcp_manager=mcp_manager
        )

    # --------------------------
    # Step 1.1: 检测坐标轴刻度
    # --------------------------
    async def detect_axis_ticks(self, ctx: SpectroContext):
        """调用视觉 LLM 检测坐标轴刻度，如果无图像或非光谱图报错"""
        class NoImageError(Exception): pass
        class NotSpectralImageError(Exception): pass

        if not ctx.image_path or not os.path.exists(ctx.image_path):
            raise NoImageError("❌ 未输入图像或图像路径不存在")

        prompt = """
你是一个专业视觉分析模型，擅长从科学图表提取坐标轴刻度信息。
如果输入中不包含光谱图，请输出 “非光谱图”。
严格按照以下 JSON Schema 输出：
{
  "x_axis": {
    "label_and_Unit": "str",
    "tick_range": {"min": float, "max": float},
    "ticks": [float]
  },
  "y_axis": {
    "label_and_Unit": "str",
    "tick_range": {"min": float, "max": float},
    "ticks": [float]
  }
}
"""

        axis_info = await self.call_llm_with_context(
            prompt,
            image_path=ctx.image_path,
            parse_json=True,
            description="坐标轴信息"
        )

        if axis_info in ["未输入图像", "非光谱图"]:
            raise NotSpectralImageError(f"❌ 图像不是光谱图，LLM 输出: {axis_info}")

        ctx.set("axis_info", axis_info)

    # --------------------------
    # Step 1.2~1.3: 合并视觉+OCR刻度
    # --------------------------
    async def combine_axis_mapping(self, ctx: SpectroContext):
        """结合视觉结果与 OCR 结果生成像素-数值映射"""
        axis_info_json = json.dumps(ctx.axis_info, ensure_ascii=False)
        ocr_json = json.dumps(ctx.OCR_detected_ticks, ensure_ascii=False)

        prompt = f"""
你是科学图表阅读助手。
输入两组刻度信息：
1. 视觉模型：{axis_info_json}
2. OCR/Opencv：{ocr_json}

任务：
- 合并两组结果，生成最终的刻度值-像素映射
- x 轴 pixel 单调递增，y 轴 pixel 单调递减
- 修正 OCR 与单调性冲突的 pixel
- 缺失刻度用 null，bounding-box-scale_x/y 缺失也为 null
- sigma_pixel = bounding-box-scale / 2，缺失为 null
- conf_llm: OCR 高可信度 0.9，插值/修正 0.7，缺失视觉预测 0.5

输出：
- 严格 JSON 数组，每个元素包含：
  "axis" ("x" 或 "y"), "value", "position_x", "position_y",
  "bounding-box-scale_x", "bounding-box-scale_y",
  "sigma_pixel", "conf_llm"
- 不要输出任何解释或文字
"""
        tick_pixel_raw = await self.call_llm_with_context(
            prompt,
            image_path=ctx.image_path,
            parse_json=True,
            description="刻度-像素映射"
        )

        ctx.set("tick_pixel_raw", tick_pixel_raw)

    # --------------------------
    # Step 1.4: 校验与修正
    # --------------------------
    async def revise_axis_mapping(self, ctx: SpectroContext):
        """检查并修正刻度值与像素位置匹配关系"""
        axis_mapping_json = json.dumps(ctx.tick_pixel_raw, ensure_ascii=False)

        prompt = f"""
你是科学图表阅读助手。
检查以下刻度值与像素映射：
{axis_mapping_json}

规则：
- y 轴: 数值从小到大 pixel 应严格递减
- x 轴: 数值从小到大 pixel 应严格递增
允许存在 null
如果有问题，请修订并输出 JSON；否则直接返回原输入
不要输出任何解释或额外文字
"""

        tick_pixel_revised = await self.call_llm_with_context(
            prompt,
            image_path=ctx.image_path,
            parse_json=True,
            description="修正后的刻度映射"
        )

        ctx.set("tick_pixel_raw", tick_pixel_revised)

    # --------------------------
    # 读取环境变量
    # --------------------------
    def _load_feature_params(self):
        """安全读取峰值/谷值检测参数"""
        def parse_list(val, default):
            if not val or not val.strip():
                return default
            try:
                cleaned = val.strip().strip("[]")
                if not cleaned:
                    return default
                return [int(x.strip()) for x in cleaned.split(",")]
            except Exception:
                print(f"⚠️ SIGMA_LIST 格式错误: {val}，使用默认值 {default}")
                return default

        def getenv_int(name, default):
            val = os.getenv(name)
            if val and val.strip():
                try: return int(val.strip())
                except Exception: print(f"⚠️ {name} 格式错误: {val}，使用默认值 {default}")
            return default

        def getenv_float(name, default):
            val = os.getenv(name)
            if val and val.strip():
                try: return float(val.strip())
                except Exception: print(f"⚠️ {name} 格式错误: {val}，使用默认值 {default}")
            return default

        sigma_list = parse_list(os.getenv("SIGMA_LIST"), [2, 4, 16])
        tol_pixels = getenv_int("TOL_PIXELS", 3)
        prom_peaks = getenv_float("PROM_THRESHOLD_PEAKS", 0.01)
        prom_troughs = getenv_float("PROM_THRESHOLD_TROUGHS", 0.01)
        weight_original = getenv_float("WEIGHT_ORIGINAL", 1.0)
        plot_peaks = getenv_int("PLOT_PEAKS_NUMBER", 10)
        plot_troughs = getenv_int("PLOT_TROUGHS_NUMBER", 15)

        return sigma_list, tol_pixels, prom_peaks, prom_troughs, weight_original, plot_peaks, plot_troughs

    # --------------------------
    # Step 1.1~1.11: 主流程
    # --------------------------
    async def run(self, ctx: SpectroContext, plot: bool = True):
        """执行完整视觉分析流程"""
        try:
            # Step 1.1: 视觉 LLM 提取坐标轴
            await self.detect_axis_ticks(ctx)

            # Step 1.2: OCR 提取刻度
            ctx.set("OCR_detected_ticks", _detect_axis_ticks(ctx.image_path))

            # Step 1.3: 合并
            await self.combine_axis_mapping(ctx)

            # Step 1.4: 修正
            await self.revise_axis_mapping(ctx)

            # Step 1.5: 边框检测与裁剪
            chart_border = _detect_chart_border(ctx.image_path)
            ctx.set("chart_border", chart_border)
            _crop_img(ctx.image_path, chart_border, ctx.crop_path)

            # Step 1.6: 重映射像素
            ctx.set("tick_pixel_remap", _remap_to_cropped_canvas(ctx.tick_pixel_raw, chart_border))

            # Step 1.7: 拟合像素-数值
            ctx.set("pixel_to_value", _pixel_tickvalue_fitting(ctx.tick_pixel_remap))

            # Step 1.8: 提取曲线 & 灰度化
            curve_points, curve_gray_values = _process_and_extract_curve_points(ctx.crop_path)
            ctx.set("curve_points", curve_points)
            ctx.set("curve_gray_values", curve_gray_values)

            # Step 1.9: 光谱还原
            ctx.set("spectrum", _convert_to_spectrum(ctx.curve_points, ctx.curve_gray_values, ctx.pixel_to_value))

            # Step 1.10: 检测峰值/谷值
            sigma_list, tol_pixels, prom_peaks, prom_troughs, weight_original, plot_peaks, plot_troughs = self._load_feature_params()
            ctx.set('sigma_list', sigma_list)
            ctx.set("peaks", _find_features_multiscale(ctx, "peak", sigma_list, prom_peaks, tol_pixels, weight_original))
            ctx.set("troughs", _find_features_multiscale(ctx, "trough", sigma_list, prom_troughs, tol_pixels, weight_original))

            # Step 1.11: 可选绘图
            if plot:
                ctx.set("spectrum_fig", _plot_spectrum(ctx))
                ctx.set("features_fig", _plot_features(ctx, sigma_list, [plot_peaks, plot_troughs]))

        except Exception as e:
            print(f"❌ run pipeline terminated with error: {e}")
            raise

# ---------------------------------------------------------
# 2. Rule-based Analyst — 负责基于规则的物理分析
# ---------------------------------------------------------
class SpectralRuleAnalyst(BaseAgent):
    
    """规则驱动型分析师：基于给定的物理与谱线知识进行定性分析"""

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Rule Analyst',
            mcp_manager=mcp_manager
        )
    # def __init__(self, agents):
    #     self.main_agent = agents['main']
    #     self.vis_llm = agents['vis']

    
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
- 这些线是宽的还是窄的、对称的还是不对称的。
- 请避免给出具体数值（如精确波长或通量），只需说明它们相对的位置与特征。

### Step 3: 整体结构与噪声特征
- 光谱信噪比的总体印象（高 / 中 / 低）。
- 是否存在噪声波动、异常尖峰或数据缺口。
- 光谱在长波端或短波端的质量变化情况。

---

⚠️ **注意：**
- 不输出精确数值或表格
- 不尝试计算红移
- 重点在视觉与形态描述，像人类天文学家一样进行定性判断
- 不要调用工具；

最后，请以结构化的方式输出你的观察结果，例如使用分节标题：
-（连续谱）
-（发射与吸收）
-（噪声与数据质量）
"""
        
        response = await self.call_llm_with_context(
            prompt,
            image_path=ctx.image_path,
            parse_json=False,
            description="视觉光谱定性描述"
        )
        ctx.set('visual_interpretation', response)
    
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
- 只输出中等置信度以上的回答
- 不输出精确数值或表格
- 不尝试计算红移
- 重点在视觉与形态描述，像人类天文学家一样进行定性判断
- 不要调用工具；
"""
        response = await self.call_llm_with_context(
            prompt,
            image_path=ctx.image_path,
            parse_json=False,
            description="初步分类"
        )
        ctx.set('preliminary_classification', response)
        
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
综合原曲线和 sigma={ctx.sigma_list} 的高斯平滑曲线，使用 scipy 函数进行了峰/谷识别。
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
        
        response = await self.call_llm_with_context(prompt, parse_json=False, description="Step 1 Lyα分析")
        ctx.append('rule_analysis', response)

    async def step_2(self, ctx):
        header = self._common_prompt_header(ctx)
        tail = self._common_prompt_tail("Step 2: 其他显著发射线分析")

        prompt = header + """
请继续分析:

Step 2: 其他显著发射线分析
1. 以 Step 1 得到的红移为标准，使用工具 predict_obs_wavelength 检查光谱中是否可能存在其他显著发射线（如 C IV 1549, C III] 1909, Mg II 2799, Hβ, Hα 等）。不要自行计算。
2. 还有什么需要注意的发射线？
""" + tail

        response = await self.call_llm_with_context(prompt, parse_json=False, description="Step 2 发射线分析")
        ctx.append('rule_analysis', response)

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
    - 由各谱线在 sigma=2 平滑下的强度 flux 作为权重，使用工具 weighted_average 进行加权平均，输出得到的加权红移值 z ± Δz
    - 涉及计算红移的流程必须使用工具 calculate_redshift，不允许自行计算。
- 给出该红移下，你能确定的各个发射线的波长和发射线名。
""" + tail

        response = await self.call_llm_with_context(prompt, parse_json=False, description="Step 3 综合判断")
        ctx.append('rule_analysis', response)

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

        response = await self.call_llm_with_context(prompt, parse_json=False, description="Step 4 补充分析")
        ctx.append('rule_analysis', response)

    # --------------------------
    # Run 全流程
    # --------------------------
    async def run(self, ctx: SpectroContext):
        """执行规则分析完整流程"""
        await self.describe_spectrum_picture(ctx)
        await self.preliminary_classification(ctx)
        await self.step_1(ctx)
        await self.step_2(ctx)
        await self.step_3(ctx)
        await self.step_4(ctx)



# ---------------------------------------------------------
# 3. Revision Supervisor — 负责交叉审核与评估
# ---------------------------------------------------------
class SpectralAnalysisAuditor(BaseAgent):
    """结果监督者：审查并校正其他分析 agent 的输出"""

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Analysis Auditor',
            mcp_manager=mcp_manager
        )

    def _common_prompt_header(self, ctx) -> str:
        peak_json = json.dumps(ctx.peaks[:10], ensure_ascii=False)
        trough_json = json.dumps(ctx.troughs, ensure_ascii=False)
        rule_analysis = "\n\n".join(str(item) for item in ctx.rule_analysis)
        return f"""
你是一位严谨的【天文学光谱报告审查官】。

任务目标：
- 审核其他 agent 的光谱分析报告或想法
- 识别其中的逻辑漏洞、计算漏洞、不一致或错误推断
- 提出修正意见或补充分析方向

工作原则：
- 保持客观与批判性思维
- 不重复原分析，只指出问题与改进建议
- 若原报告合理，应明确确认其有效性
- 涉及红移和光谱观测波长的计算必须使用工具 calculate_redshift 和  predict_obs_wavelength。不允许自行计算。

输出要求：
- 请输出说明性的语言
- 简明列出审查意见（例如：“结论偏早”，“谱线解释正确”）
- 对每个发现附上改进建议
- 最后给出整体评价（可靠/部分可信/不可信）

已知：综合原曲线和 sigma=2、sigma=4、sigma=16 三条高斯平滑曲线，使用 scipy 函数进行了峰/谷识别。
关于峰/谷的讨论以以下数据为准：
- 代表性的前 10 条发射线：
{peak_json}
- 可能的吸收线：
{trough_json}

其他分析师给出的光谱分析报告为：

{rule_analysis}

该报告在红移计算时保留了 2 位小数。
"""
    
    async def auditing(self, ctx: str) -> str:
        header = self._common_prompt_header(ctx)

        body = f"""
请对这份分析报告进行检查。
"""
        prompt = header + body
        response = await self.call_llm_with_context(prompt, parse_json=False, description="报告审查")
        ctx.append('auditing_history', response)

    async def further_auditing(self, ctx: str) -> str:
        header = self._common_prompt_header(ctx)
        auditing_history_json = ctx.auditing_history[-1]
        response_history_json = ctx.refine_history[-1]

        body = f"""
你对这份分析报告的最新质疑为
{auditing_history_json}

其他分析师的回答为
{response_history_json}

请回应其他分析师的回答，并继续进行审查。
"""
        prompt = header + body
        response = await self.call_llm_with_context(prompt, parse_json=False, description="报告审查")
        ctx.append('auditing_history', response)



# ---------------------------------------------------------
# 4. Reflective Analyst — 自由回应审查并改进
# ---------------------------------------------------------
class SpectralRefinementAssistant(BaseAgent):
    """改进者：回应审查并改进分析"""

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Refinement Assistant',
            mcp_manager=mcp_manager
        )

    # def __init__(self, agents):
    #     self.main_agent = agents['main']

    def _common_prompt_header(self, ctx) -> str:
        peak_json = json.dumps(ctx.peaks[:10], ensure_ascii=False)
        trough_json = json.dumps(ctx.troughs, ensure_ascii=False)
        rule_analysis = "\n\n".join(str(item) for item in ctx.rule_analysis)
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
- 涉及红移和光谱观测波长的计算必须使用工具 calculate_redshift 和  predict_obs_wavelength。不允许自行计算。

输出要求：
- 请输出说明性的语言
- 列出收到的反馈及对应回应
- 提供改进后的光谱分析总结
- 说明修改内容及其科学合理性

已知：综合原曲线和 sigma=2、sigma=4、sigma=16 三条高斯平滑曲线，使用 scipy 函数进行了峰/谷识别。
关于峰/谷的讨论以以下数据为准：
- 代表性的前 10 条发射线：
{peak_json}
- 可能的吸收线：
{trough_json}
其他分析师给出的光谱分析报告为：
{rule_analysis}

这份报告在红移计算时保留了 2 位小数。
"""

    async def refine(self, ctx):
        header = self._common_prompt_header(ctx)
        auditing = ctx.auditing_history[-1]
        body = f"""
负责核验报告的分析师给出的最新建议为
{auditing}

请对建议进行回应。
"""
        prompt = header + body
        response = await self.call_llm_with_context(prompt, parse_json=False, description="回应审查")
        ctx.append('refine_history', response)



# ---------------------------------------------------------
# 🧩 5. Host Integrator — 汇总与总结多方观点
# ---------------------------------------------------------
from typing import Union
class SpectralSynthesisHost(BaseAgent):
    """汇总主持人：整合多Agent的分析与结论"""

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Synthesis Host',
            mcp_manager=mcp_manager
        )

    def get_system_prompt(self) -> str:
        return f"""
你是一位负责统筹的【天文学光谱分析主持人】。

任务目标：
- 汇总视觉分析师、规则分析师、审查官和再分析师的所有输出
- 综合不同角度的结论，形成最终的光谱解释
- 清楚指出各方意见的差异与一致点

工作原则：
- 无需调用工具
- 不盲从任何单一分析
- 保持整体科学性与逻辑一致性
- 最终输出必须具备可追溯性（说明来自哪些agent的依据）

输出要求：
- 输出说明性文字
- 输出数据保留2位小数
- 只需输出分析内容，无需声明各段分析文字的来源
- 给出最终综合结论及可信度评级（高/中/低）
- 如果仍存在不确定性，请明确指出
"""


    def summary(self, ctx) -> str:
        visual_interpretation_json = json.dumps(ctx.visual_interpretation, ensure_ascii=False)
        rule_analysis = "\n\n".join(str(item) for item in ctx.rule_analysis)
        rule_analysis_json = json.dumps(rule_analysis, ensure_ascii=False)
        auditing = "\n\n".join(str(item) for item in ctx.auditing_history)
        auditing_json = json.dumps(auditing, ensure_ascii=False)
        refine = "\n\n".join(str(item) for item in ctx.refine_history)
        refine_json = json.dumps(refine, ensure_ascii=False)

        header = self.get_system_prompt()

        prompt = f"""

对光谱的视觉描述
{visual_interpretation_json}

规则分析师的观点：
{rule_analysis_json}

审查官的观点：
{auditing_json}

再分析师的观点：
{refine_json}

输出格式如下：

- 光谱的视觉特点
- 分析报告（综合规则分析师、审查官和再分析师的所有输出，逐个 Step 进行结构化输出）
- 结论
    - 该天体的天体类型和红移 z ± Δz
    - 认证出的谱线（输出 谱线名-λ_rest-λ_obs）
    - 光谱的信噪比如何
    - 分析报告的可信度评分（如果能认证出2条以上的谱线，则可信度为“高”；能认证出1条谱线，可信度为“中”；其他情况为“低”）
    - 是否需要人工介入判断
"""
        return header + prompt

    async def run(self, ctx: SpectroContext) -> str:
        prompt = self.summary(ctx)
        response = await self.call_llm_with_context(prompt, parse_json=False, description="总结")
        ctx.set('summary', response)