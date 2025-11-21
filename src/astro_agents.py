import json
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

from .context_manager import SpectroState
from .base_agent import BaseAgent
from .mcp_manager import MCPManager

from .utils import (
    _detect_axis_ticks, _detect_chart_border, _crop_img,
    _remap_to_cropped_canvas, _pixel_tickvalue_fitting,
    _process_and_extract_curve_points, _convert_to_spectrum,
    _find_features_multiscale, _plot_spectrum, _plot_features,
    parse_list, getenv_float, getenv_int, _load_feature_params, 
    _ROI_features_finding, merge_features, plot_merged_features, safe_to_bool
)

# ---------------------------------------------------------
# 1. Visual Assistant — 负责图像理解与坐标阅读
# ---------------------------------------------------------

class SpectralVisualInterpreter(BaseAgent):
    """
    SpectralVisualInterpreter

    从科学光谱图中自动提取坐标轴刻度、边框、像素映射、峰/谷等信息
    """

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Visual Interpreter',
            mcp_manager=mcp_manager
        )

    # --------------------------
    # Step 1.1: 检测坐标轴刻度
    # --------------------------
    async def detect_axis_ticks(self, state: SpectroState):
        """
        调用视觉 LLM 检测坐标轴刻度，如果无图像或非光谱图报错
        """
        class NoImageError(Exception): pass
        class NotSpectralImageError(Exception): pass

        if not state['image_path'] or not os.path.exists(state['image_path']):
            print(state['image_path'])
            raise NoImageError("❌ 未输入图像或图像路径不存在")

        system_prompt = state['prompt'][f'{self.agent_name}']['detect_axis_ticks']['system_prompt']
        user_prompt = state['prompt'][f'{self.agent_name}']['detect_axis_ticks']['user_prompt']

        axis_info = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=state['image_path'],
            parse_json=True,
            description="坐标轴信息"
        )
        if axis_info == "非光谱图":
            raise NotSpectralImageError(f"❌ 图像不是光谱图，LLM 输出: {axis_info}")
        # print(axis_info)
        state["axis_info"] = axis_info

    # --------------------------
    # Step 1.2~1.3: 合并视觉+OCR刻度
    # --------------------------
    async def combine_axis_mapping(self, state: SpectroState):
        """结合视觉结果与 OCR 结果生成像素-数值映射"""
        axis_info_json = json.dumps(state['axis_info'], ensure_ascii=False)
        ocr_json = json.dumps(state['OCR_detected_ticks'], ensure_ascii=False)

        system_prompt = state['prompt'][f'{self.agent_name}']['combine_axis_mapping']['system_prompt']
        user_prompt = state['prompt'][f'{self.agent_name}']['combine_axis_mapping']['user_prompt'].format(
            axis_info_json=axis_info_json,
            ocr_json=ocr_json
        )
        tick_pixel_raw = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=None,
            parse_json=True,
            description="刻度-像素映射"
        )
        state["tick_pixel_raw"] = tick_pixel_raw

    # --------------------------
    # Step 1.4: 校验与修正
    # --------------------------
    async def revise_axis_mapping(self, state: SpectroState):
        """检查并修正刻度值与像素位置匹配关系"""
        axis_mapping_json = json.dumps(state['tick_pixel_raw'], ensure_ascii=False)

        system_prompt = state['prompt'][f'{self.agent_name}']['revise_axis_mapping']['system_prompt']
        user_prompt = state['prompt'][f'{self.agent_name}']['revise_axis_mapping']['user_prompt'].format(
            axis_mapping_json=axis_mapping_json
        )

        tick_pixel_revised = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=None,
            parse_json=True,
            description="修正后的刻度映射"
        )
        state["tick_pixel_raw"] = tick_pixel_revised
        # print(tick_pixel_revised)

    # --------------------------
    # Step 1.1~1.11: 主流程
    # --------------------------
    async def run(self, state: SpectroState, plot: bool = True):
        """执行完整视觉分析流程"""
        try:
            # Step 1.1: 视觉 LLM 提取坐标轴
            await self.detect_axis_ticks(state)

            # Step 1.2: OCR 提取刻度
            state["OCR_detected_ticks"] = _detect_axis_ticks(state['image_path'])
            # for i in state["OCR_detected_ticks"]:
            #     print(i)

            # Step 1.3: 合并
            await self.combine_axis_mapping(state)
            # for i in state["tick_pixel_raw"]:
            #     print(i)

            # Step 1.4: 修正
            await self.revise_axis_mapping(state)
            # for i in state["tick_pixel_raw"]:
            #     print(i)

            # Step 1.5: 边框检测与裁剪
            state["chart_border"] = _detect_chart_border(state['image_path'])
            _crop_img(state['image_path'], state["chart_border"], state['crop_path'])

            # Step 1.6: 重映射像素
            state["tick_pixel_remap"] = _remap_to_cropped_canvas(state['tick_pixel_raw'], state["chart_border"])

            # Step 1.7: 拟合像素-数值
            state["pixel_to_value"] = _pixel_tickvalue_fitting(state['tick_pixel_remap'])

            # Step 1.8: 提取曲线 & 灰度化
            curve_points, curve_gray_values = _process_and_extract_curve_points(state['crop_path'])
            state["curve_points"] = curve_points
            state["curve_gray_values"] = curve_gray_values

            # Step 1.9: 光谱还原
            state["spectrum"] = _convert_to_spectrum(state['curve_points'], state['curve_gray_values'], state['pixel_to_value'])
            # print(state["spectrum"]['new_wavelength'])
            # print(state["spectrum"]['weighted_flux'])
            # Step 1.10: 检测峰值/谷值
            sigma_list, tol_pixels, prom_peaks, prom_troughs, weight_original, plot_peaks, plot_troughs = _load_feature_params()
            state['sigma_list'] = sigma_list
            try:
                spec = state["spectrum"]
                wavelengths = np.array(spec["new_wavelength"])
                flux = np.array(spec["weighted_flux"])
                state["peaks"] = _find_features_multiscale(
                    wavelengths, flux,
                    state, feature="peak", sigma_list=sigma_list,
                    prom=prom_peaks, tol_pixels=tol_pixels, weight_original=weight_original,
                    use_continuum_for_trough=True
                )
                # print(f"peaks: \n {state['peaks']}")
                # print(state["peaks"])
                state["troughs"] = _find_features_multiscale(
                    wavelengths, flux,
                    state, feature="trough", sigma_list=sigma_list,
                    prom=prom_troughs, tol_pixels=tol_pixels, weight_original=weight_original,
                    use_continuum_for_trough=True,
                    min_depth=0.08
                )
                # print(f"troughs: \n {state['troughs']}")
            except Exception as e:
                print(f"❌ find features multiscale terminated with error: {e}")
                raise

            # Step 1.11: 可选绘图
            if plot:
                try:
                    state["spectrum_fig"] = _plot_spectrum(state)
                    # state["features_fig"] = _plot_features(state, sigma_list, [plot_peaks, plot_troughs])
                except Exception as e:
                    print(f"❌ plot spectrum or features terminated with error: {e}")
                    raise

            return state

        except Exception as e:
            print(f"❌ run pipeline terminated with error: {e}")
            raise


# ---------------------------------------------------------
# 2. Rule-based Analyst — 负责基于规则的物理分析
# ---------------------------------------------------------
class SpectralRuleAnalyst(BaseAgent):
    """
    规则驱动型分析师：基于给定的物理与谱线知识进行定性分析
    """

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Rule Analyst',
            mcp_manager=mcp_manager
        )

    async def describe_spectrum_picture(self, state: SpectroState):
        function_prompt = state['prompt'][f'{self.agent_name}']['describe_spectrum_picture']
        async def _filter_noise(state):
            BR = [5650, 5850]
            RZ = [7500, 7700]
            spec = state['spectrum']
            wv = np.array(spec['new_wavelength'])
            ceiling = np.array(spec['max_unresolved_flux'])
            floor = np.array(spec['min_unresolved_flux'])
            delta = ceiling - floor
            mask_BR = (wv >= BR[0]) & (wv <= BR[1])
            mask_RZ = (wv >= RZ[0]) & (wv <= RZ[1])
            wv_BR, delta_BR = wv[mask_BR], delta[mask_BR]
            wv_RZ, delta_RZ = wv[mask_RZ], delta[mask_RZ]
            def truncate(arr, N=150):
                return arr[:N] if len(arr) > N else arr
            wv_BR_t = truncate(wv_BR)
            wv_BR_t = wv_BR_t.tolist()
            delta_BR_t = truncate(delta_BR)
            delta_BR_t = delta_BR_t.tolist()
            wv_RZ_t = truncate(wv_RZ)
            wv_RZ_t = wv_RZ_t.tolist()
            delta_RZ_t = truncate(delta_RZ)
            delta_RZ_t = delta_RZ_t.tolist()

            system_prompt = function_prompt['_filter_noise']['system_prompt']
            user_prompt = function_prompt['_filter_noise']['user_prompt'].format(
                BR_L=BR[0],
                BR_R=BR[1],
                RZ_L=RZ[0],
                RZ_R=RZ[1],
                wv_BR_t=wv_BR_t,
                delta_BR_t=delta_BR_t,
                wv_RZ_t=wv_RZ_t,
                delta_RZ_t=delta_RZ_t
            )

            response = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=None,
                parse_json=True,
                description="Filter噪声判断"
            )
            return(response)

        async def _visual(state):
            system_prompt = function_prompt['_visual']['system_prompt']
            user_prompt_1 = function_prompt['_visual']['user_prompt_continuum']
            response_1 = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt_1,
                image_path=state['spec_extract_path'],
                parse_json=True,
                description="视觉光谱定性描述"
            )

            user_prompt_2 = function_prompt['_visual']['user_prompt_lines']
            response_2 = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt_2,
                image_path=state['spec_extract_path'],
                parse_json=True,
                description="视觉光谱定性描述"
            )

            user_prompt_3 = function_prompt['_visual']['user_prompt_quality']
            response_3 = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt_3,
                image_path=state['spec_extract_path'],
                parse_json=True,
                description="视觉光谱定性描述"
            )
            return '\n'.join([response_1, response_2, response_3])

        async def _get_ROI(state):
            _visual_json = json.dumps(state['visual_interpretation'][1], ensure_ascii=False)
            system_prompt = function_prompt['_get_ROI']['system_prompt']
            user_prompt = function_prompt['_get_ROI']['user_prompt'].format(_visual_json=_visual_json)

            response_2 = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_json=True,
                description="视觉光谱定性描述"
            )
            return response_2

        async def _integrate(state):
            filter_noise_json = json.dumps(state['visual_interpretation'][0], ensure_ascii=False)
            visual_json       = json.dumps(state['visual_interpretation'][1], ensure_ascii=False)
            roi_json          = json.dumps(state['visual_interpretation'][2], ensure_ascii=False)

            system_prompt = function_prompt['_integrate']['system_prompt']
            user_prompt_integrate = function_prompt['_integrate']['user_prompt'].format(
                filter_noise_json=filter_noise_json,
                visual_json=visual_json,
                roi_json=roi_json
            )
            response = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt_integrate,
                parse_json=True,
                description="视觉光谱定性描述"
            )
            return response

        result_filter_noise = await _filter_noise(state)
        state['visual_interpretation'] = [result_filter_noise]
        result_visual = await _visual(state)
        state['visual_interpretation'].append(result_visual)
        result_ROI = await _get_ROI(state)
        state['visual_interpretation'].append(result_ROI)
        result_integrate = await _integrate(state)
        state['visual_interpretation'] = result_integrate

        visual_interpretation_path = os.path.join(state['output_dir'], f'{state['image_name']}_visual_interpretation.txt')
        with open(visual_interpretation_path, 'w', encoding='utf-8') as f:
            json_str = json.dumps(state['visual_interpretation'], indent=2, ensure_ascii=False)
            f.write(json_str)
    
    async def preliminary_classification(self, state: SpectroState) -> str:
        """初步分类：根据光谱形态初步判断天体类型"""

        visual_interpretation_json = json.dumps(state['visual_interpretation'], ensure_ascii=False)
        sigma_list_json = json.dumps(state['sigma_list'], ensure_ascii=False)
        peaks_info = [
            {
                "wavelength": pe.get('wavelength'),
                "flux": pe.get('mean_flux'),
                "width": pe.get('width_mean'),
                "prominance": pe.get('max_prominence'),
                # "seen_in_scales_of_sigma": pe.get('seen_in_scales_of_sigma'),
            }
            for pe in state.get('merged_peaks', [])[:10]
        ]
        peak_json = json.dumps(peaks_info, ensure_ascii=False)
        trough_info = [
            {
                "wavelength": tr.get('wavelength'),
                "flux": tr.get('mean_flux'),
                "width": tr.get('width_mean'),
                # "seen_in_scales_of_sigma": tr.get('seen_in_scales_of_sigma')
            }
            for tr in state.get('merged_troughs', [])[:15]
        ]
        trough_json = json.dumps(trough_info, ensure_ascii=False)
#         prompt = f"""
# 你是一位经验丰富的天文学光谱分析助手。

# 你将看到一条天文光谱曲线（来自未知红移的天体），它可能属于以下三类之一：
# 1. **Star**：
#     - 连续谱较强，谱线通常是吸收线（如 Balmer 系列、金属线等），几乎没有明显红移。
# 2. **Galaxy**：
#     - 有一定红移，常有发射线或吸收线，谱线较窄。
#     - 连续谱（与发射线及噪声相比）强度较弱。
#     - 部分星系的连续谱呈现蓝端较低而红端显著升高的趋势。
# 3. **QSO**：
#     - 具有**强发射线**。谱线宽度明显。
#     - 连续谱覆盖可见/紫外波段。
#     - 通常有明显红移。

# 前一位天文学助手已经定性地描述了光谱的整体形态：

# {visual_interpretation_json}

# 综合原曲线和 sigma={state['sigma_list']} 的高斯平滑曲线，使用 scipy 函数进行了峰/谷识别。
# 关于峰/谷的讨论以以下数据为准：
# - 代表性的前 10 条发射线：
# {peak_json}
# - 可能的吸收线：
# {trough_json}

# 请根据他的描述进行判断，猜测该光谱可能属于哪一类或几类，给出置信度。

# 你的回答格式请严格遵循：

# 猜测 1：
# - **类别**: Star / Galaxy / QSO （三选一）
# - **理由**: 用简洁的语言解释分类原因（如谱线宽度、红移特征、连续谱形态）
# - **置信度**: 高 / 中 / 低
# 猜测 2：
# - **类别**: Star / Galaxy / QSO （三选一）
# - **理由**: 用简洁的语言解释分类原因（如谱线宽度、红移特征、连续谱形态）
# - **置信度**: 高 / 中 / 低
# 等等。

# ⚠️ **注意**：
# - 只输出**中等置信度**以上的回答
# - 不输出精确数值或表格
# - 不尝试计算红移
# - 重点在视觉与形态描述，像人类天文学家一样进行定性判断
# - 不要调用工具；
# """
        system_prompt = """
你是一位经验丰富的天文学光谱分析助手。

你的任务是根据光谱的定性描述和特征数据，猜测天体可能属于的类别。

可选的类别：
1. **Star**：
    - 连续谱较强，谱线通常是吸收线（如 Balmer 系列、金属线等），几乎没有明显红移。
2. **Galaxy**：
    - 有一定红移，常有发射线或吸收线。
    - 谱线通常较窄。
    - 连续谱较不明显。
    - 部分星系的连续谱呈现蓝端较低而红端显著升高的趋势。
3. **QSO**：
    - 具有**强发射线**。谱线宽度明显。
    - 连续谱覆盖可见/紫外波段。
    - 通常有明显红移。

输出要求：
- 每个猜测包含：类别、理由、置信度
- 不输出精确数值或表格
- 不尝试计算红移
- 重点在视觉与形态描述，像人类天文学家一样进行定性判断
- 不要调用工具

输出格式：
猜测 1：
- **类别**: Star / Galaxy / QSO （三选一）
- **理由**: 用简洁的语言解释分类原因（如谱线宽度、红移特征、连续谱形态）
- **置信度**: 高 / 中 / 低
猜测 2：
...
"""
        user_prompt = f"""
请根据以下光谱数据进行分析：

前一位天文学助手已经定性地描述了光谱的整体形态：
{visual_interpretation_json}
其中 filter noise 是因为在不同 filter（如 B,R,Z）重叠处出现的非物理的噪声。

在全局上，综合原曲线和 sigma={sigma_list_json} 的高斯平滑曲线，使用 scipy 函数进行了峰/谷识别。
在 ROI (region of interest) 上，综合局部的原曲线和 sigma={sigma_list_json} 的高斯平滑曲线，使用 scipy 函数进行了峰/谷识别。

关于峰/谷的讨论以以下数据为准：
- 代表性的前 10 条发射线：
{peak_json}
- 可能的吸收线：
{trough_json}

请根据这些描述和数据，猜测该光谱可能属于哪一类或几类天体。
"""
        response = await self.call_llm_with_context(
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            image_path=state['image_path'],
            parse_json=True,
            description="初步分类"
        )
        state['preliminary_classification'] = response

    async def preliminary_classification_monkey(self, state):
        """ My dear monkey friend and its typewriter """
        preliminary_classification_json = json.dumps(state['preliminary_classification'], ensure_ascii=False)
        prompt = f"""
你是一个天文学光谱分析助手。
你接收到的是其他助手对一张光谱的光源类别的初步猜测：
{preliminary_classification_json}

请输出这份猜测里给出的光源类别。

输出格式为数组 List[str]，数组的元素必须在 "Star", "Galaxy" 和 "QSO" 中选择。

- 注意：即使只有一个满足条件的光源类别，也要以 List[str] 的格式输出。
"""
        response = await self.call_llm_with_context(
            system_prompt = '',
            user_prompt = prompt,
            parse_json=True,
            description="初步分类猴子"
        )
        return response
    ###################################
    # QSO part
    ###################################
    async def _QSO(self, state):
        """QSO"""
        def _common_prompt_header_QSO(state, include_rule_analysis=True, include_step_1_only=False):
            """构造每个 step 公共的 prompt 前段"""
            visual_json = json.dumps(state['visual_interpretation'], ensure_ascii=False)
            # peak_json = json.dumps(state['peaks'][:10], ensure_ascii=False)
            # trough_json = json.dumps(state['troughs'], ensure_ascii=False)
            peaks_info = [
                {
                    "wavelength": pe.get('wavelength'),
                    "flux": pe.get('mean_flux'),
                    "width": pe.get('width_mean'),
                    "prominance": pe.get('max_prominence'),
                    "seen_in_scales_of_sigma": pe.get('seen_in_scales_of_sigma'),
                }
                for pe in state.get('merged_peaks', [])[:10]
            ]
            peak_json = json.dumps(peaks_info, ensure_ascii=False)
            trough_info = [
                {
                    "wavelength": tr.get('wavelength'),
                    "flux": tr.get('mean_flux'),
                    "width": tr.get('width_mean'),
                    "seen_in_scales_of_sigma": tr.get('seen_in_scales_of_sigma')
                }
                for tr in state.get('merged_troughs', [])[:15]
            ]
            trough_json = json.dumps(trough_info, ensure_ascii=False)

            header = f"""
    你是一位天文学光谱分析助手。

    以下信息可能来自于一个未知红移的 QSO 光谱。

    之前的助手已经对这个光谱进行了初步描述：
    {visual_json}

    该光谱的波长范围是{state['spectrum']['new_wavelength'][0]} Å 到 {state['spectrum']['new_wavelength'][-1]} Å。
    """

            if include_rule_analysis and state['rule_analysis_QSO']:
                if include_step_1_only==True:
                    rule_json = json.dumps(state['rule_analysis_QSO'][0], ensure_ascii=False)
                else:
                    rule_json = json.dumps("\n".join(str(item) for item in state['rule_analysis_QSO']), ensure_ascii=False)
                header += f"\n之前的助手已经进行了一些分析:\n{rule_json}\n"

            tol_pixels = getenv_int("TOL_PIXELS", 10)
            a_x = state['pixel_to_value']['x']['a']
            tol_wavelength = a_x * tol_pixels
            header += f"""
    综合原曲线和 sigma={state['sigma_list']} 的高斯平滑曲线，使用 scipy 函数进行了峰/谷识别。
    关于峰/谷的讨论以以下数据为准：
    - 代表性的前 10 条发射线：
    {peak_json}
    - 可能的吸收线：
    {trough_json}
    - 波长误差在 ~ ±{tol_wavelength/2} Å 的量级或更大
    """
            return header

        def _common_prompt_tail(step_title, extra_notes=""):
            """构造每个 step 公共尾部，保留 step 特有输出/分析指示"""
            tail = f"""
    ---

    输出格式为：
    {step_title}
    ...

    ---

    🧭 注意：
    - 计算得来的非原始数据，输出时保留 3 位小数。
    - 不需要进行重复总结。
    - 不需要逐行地重复输入数据；
    - 重点在物理推理与合理解释；
    - 请保证最终输出完整，不要中途截断。
    """
            if extra_notes:
                tail = extra_notes + "\n" + tail
            return tail
    
        async def step_1_QSO(state):
            header = _common_prompt_header_QSO(state, include_rule_analysis=False)
            tail = _common_prompt_tail("Step 1: Lyα 谱线检测")

            prompt = header + """
请按以下步骤分析:

Step 1: Lyα 谱线检测
假设该光谱中存在 Lyα 发射线（λ_rest = 1216 Å）：
1. 在光谱蓝端，流量较大，且有一定宽度的峰中，推测哪条最可能为 Lyα 线（从提供的峰列表中选择）。
2. 输出：
- 观测波长 λ_obs
- 流量 Flux
- 谱线宽度
3. 使用工具 calculate_redshift 计算该峰为 Lyα 发射线时的红移 z。
4. 检查蓝端（短波长方向）是否存在 Lyα forest 特征：吸收线相对更密集、较窄且分布在 Lyα 蓝端附近。请指出并进行简短说明。
""" + tail
            
            response = await self.call_llm_with_context(
                system_prompt='', 
                user_prompt=prompt, 
                parse_json=True, 
                description="Step 1 Lyα 分析"
            )
            state['rule_analysis_QSO'].append(response)

        async def step_2_QSO(state):
            header = _common_prompt_header_QSO(state)
            tail = _common_prompt_tail("Step 2: 其他显著发射线分析")

            prompt = header + """
请继续分析:

Step 2: 其他显著发射线分析
1. 在 Step 1 得到的红移下，使用工具 predict_obs_wavelength 计算以下三条主要发射线：C IV 1549, C III] 1909, Mg II 2799 在光谱中的理论位置。
2. 光谱中是否有与三者相匹配的峰？
3. 如果存在发射线与观测峰值的匹配，根据匹配结果，分别使用工具 calculate_redshift 计算红移。按“发射线名--静止系波长--观测波长--红移”的格式输出。
""" + tail

            response = await self.call_llm_with_context('', prompt, parse_json=False, description="Step 2 发射线分析")
            state['rule_analysis_QSO'].append(response)

        async def step_3_QSO(state):
            header = _common_prompt_header_QSO(state)
            tail = _common_prompt_tail("Step 3: 综合判断")

            prompt = header + """
请继续分析:

Step 3: 综合判断
1. 在 Step 1 到 Step 2 中，如果：
    - C IV 和 C III] 两条主要谱线存在缺失或大幅偏移
    - 使用 lyα 谱线计算的红移与其他谱线的计算结果不一致，
此时请输出“应优先假设 Lyα 谱线未被找峰程序捕获”，并结束 Step 3 的分析。不要输出其他信息。
2.仅在有显著的 Lyα 峰值，且红移计算结果与其他谱线基本一致时，进行以下操作：
    - 因为天文学中存在外流等现象，请将当前所有匹配中**最低电离态谱线的红移**作为光谱的红移。输出红移结果。（因为存在不对称和展宽，Lyα的置信度是较低的）
""" + tail

            response = await self.call_llm_with_context('', prompt, parse_json=False, description="Step 3 综合判断")
            state['rule_analysis_QSO'].append(response)
            
        async def step_4_QSO(state):
            header = _common_prompt_header_QSO(state, include_step_1_only=True)
            tail = _common_prompt_tail("Step 4: 补充步骤（假设 Step 1 所选择的谱线并非 Lyα）")

            prompt = header + """
请继续分析:

Step 4: 补充步骤（假设 Step 1 所选择的谱线并非 Lyα）
- 请抛开前述步骤的分析内容。考虑 Step 1 所选择的谱线实际上是除 Lyα 外的其他主要发射线。
    - 假设该峰值可能对应的谱线为 C IV：
        - 输出该峰对应谱线的信息：
            - 观测波长 λ_obs
            - 流量 Flux
            - 谱线宽度
            - 根据 λ_rest，使用工具 calculate_redshift 初步计算红移 z
        - 使用工具 predict_obs_wavelength 计算在此红移下的其他主要发射线（如 C III] 和 Mg II）的理论位置。光谱中是否有与它们匹配的发射线？
        - 如果存在可能的发射线-观测波长匹配结果，使用工具 calculate_redshift 计算它们的红移。按照“发射线名--静止系波长--观测波长--红移”的格式进行输出
    
    - 若以上假设不合理，则假设该峰值可能对应 C III] 等其他主要谱线，重复推断。

    - 综合 Step 4 的所有分析，给出：
        - **最低电离态谱线的红移** 作为光谱红移
        - 输出 “发射线名--静止系波长--观测波长--红移” 匹配

- 注意：允许在由于光谱边缘的信号残缺或信噪比不佳导致部分发射线不可见。   
- 抛开其他步骤的分析内容，本节的判断是否支持 Lyα 谱线未被找峰程序捕获的假设？
""" + tail

            response = await self.call_llm_with_context('', prompt, parse_json=False, description="Step 4 补充分析")
            state['rule_analysis_QSO'].append(response)
        
        await step_1_QSO(state)
        await step_2_QSO(state)
        await step_3_QSO(state)
        await step_4_QSO(state)

#     ###################################
#     # Galaxy part
#     ###################################
#     # async def further_discription_galaxy(self, state):

#     def _common_prompt_header_galaxy(self, state, include_rule_analysis=True, include_step_1_only=False):
#         """构造每个 step 公共的 prompt 前段"""
#         visual_json = json.dumps(state['visual_interpretation'], ensure_ascii=False)
#         peaks_info = [
#             {
#                 "wavelength": pe.get('wavelength'),
#                 "flux": pe.get('mean_flux'),
#                 "width": pe.get('width_mean'),
#                 "prominance": pe.get('max_prominence'),
#                 "seen_in_scales_of_sigma": pe.get('seen_in_scales_of_sigma'),
#             }
#             for pe in state.get('peaks', [])[:10]
#         ]
#         peak_json = json.dumps(peaks_info, ensure_ascii=False)
#         trough_info = [
#             {
#                 "wavelength": tr.get('wavelength'),
#                 "flux": tr.get('mean_flux'),
#                 "width": tr.get('width_mean'),
#                 "seen_in_scales_of_sigma": tr.get('seen_in_scales_of_sigma'),
#                 "prominance": tr.get('max_prominence'),
#             }
#             for tr in state.get('troughs', [])[:15]
#         ]
#         trough_json = json.dumps(trough_info, ensure_ascii=False)

#         header = f"""
# 你是一位天文学光谱分析助手。

# 以下信息可能来自于一个未知红移的 Galaxy 光谱。

# 之前的助手已经对这个光谱进行了初步描述：
# {visual_json}

# 该光谱的波长范围是{state['spectrum']['new_wavelength'][0]} Å 到 {state['spectrum']['new_wavelength'][-1]} Å。
# """

#         if include_rule_analysis and state['rule_analysis_galaxy']:
#             if include_step_1_only==True:
#                 rule_json = json.dumps(state['rule_analysis_galaxy'][0], ensure_ascii=False)
#             else:
#                 rule_json = json.dumps("\n".join(str(item) for item in state['rule_analysis_galaxy']), ensure_ascii=False)
#             header += f"\n之前的助手已经进行了一些分析:\n{rule_json}\n"

#         tol_pixels = getenv_int("TOL_PIXELS", 10)
#         a_x = state['pixel_to_value']['x']['a']
#         tol_wavelength = a_x * tol_pixels
#         header += f"""
# 综合原曲线和 sigma={state['sigma_list']} 的高斯平滑曲线，使用 scipy 函数进行了峰/谷识别。
# 关于峰/谷的讨论以以下数据为准：
# - 代表性的前 10 条发射线：
# {peak_json}
# - 可能的吸收线：
# {trough_json}
# - 波长误差在 ~ ±{tol_wavelength/2} Å 的量级或更大
# """
#         return header
    
#     async def step_1_galaxy(self, state):
#         try:
#             # 确保 state['rule_analysis_galaxy'] 已初始化为列表
#             if 'rule_analysis_galaxy' not in state:
#                 state['rule_analysis_galaxy'] = []

#             header = self._common_prompt_header_galaxy(state, include_rule_analysis=False)
#             tail = self._common_prompt_tail("Step 1: O [III] 谱线检测")

#             prompt = header + """
# 请按以下步骤分析:

# Step 1: O [III] 谱线检测
# 假设该光谱中存在 O [III] 发射线（因峰值识别的分辨率有限，只考虑双线中最强的 λ_rest = 5008.2 Å 这一条）：
# 1. 在光谱中流量较大的窄峰中，推测哪条最可能为 O [III] 线（从提供的峰列表中选择）。
# 2. 输出：
# - 观测波长 λ_obs
# - 流量 Flux
# - 谱线宽度
# 3. 使用工具 calculate_redshift 计算该峰为 O [III] 发射线时的红移 z。
# """ + tail

#             response = await self.call_llm_with_context(
#                 prompt,
#                 parse_json=False,
#                 description="Step 1 O [III] 谱线检测"
#             )

#             # 添加到 rule_analysis_galaxy
#             state['rule_analysis_galaxy'].append(response)

#         except Exception as e:
#             print("❌ Step 1 Galaxy 分析出错：", e)
#             # 可以选择继续抛出异常或者记录错误
#             raise

#     async def step_2_galaxy(self, state):
#         header = self._common_prompt_header_galaxy(state)
#         tail = self._common_prompt_tail("Step 2: 其他主要发射线分析")

#         prompt = header + """
# 请继续分析:

# Step 2: 其他主要发射线分析
# 1. 在 Step 1 得到的红移下，使用工具 predict_obs_wavelength 计算以下主要谱线在观测光谱上的理论位置：
#     - 发射线
#         - O [II] = 3727.1 Å / 3729.9 Å 双线
#         - N [II] = 6549.8 Å / 6585.3 Å 双线
#         - S [II] = 6718.3 Å / 6732.7 Å 双线
#     - 吸收线
#         - Ca (K) = 3934.8 Å
#         - Ca (H) = 3969.6 Å
#         - G-band = 4305.6 Å
#         - Mg = 5176.7 Å
#         - Na = 5895.6 Å
#         - CaT = 8498, 8542, 8662 Å 三线
#     - 发射线或吸收线：Balmer 线系
#         - Hδ = 4102.9 A
#         - Hγ = 4341.7 A
#         - Hβ = 4862.7 A
#         - Hα = 6564.6 A
# 2. 光谱中是否有与这些谱线相匹配的峰或谷？
# 3. 如果存在发射线与观测峰/谷的匹配，根据匹配结果，分别使用工具 calculate_redshift 计算红移。按“谱线性质（发射线/吸收线）--谱线名--静止系波长--观测波长--红移”的格式输出。

# """ + tail

#         response = await self.call_llm_with_context(prompt, parse_json=False, description="Step 2 其他主要发射线分析")
#         state['rule_analysis_galaxy'].append(response)

#     async def step_3_galaxy(self, state):
#         header = self._common_prompt_header_galaxy(state)
#         tail = self._common_prompt_tail("Step 3: 综合判断")
#         a = state["pixel_to_value"]["x"]["a"]
#         rms = state["pixel_to_value"]["x"]["rms"]
#         tolerence = getenv_int("TOL_PIXELS", 10)

#         prompt = header + f"""
# 请继续分析:

# Step 3: 综合判断
# 1. 在 Step 1 到 Step 2 中，如果：
#     - 缺失 O [II] 的可能匹配
#     - 使用 O [III] 谱线计算的红移与其他谱线的计算结果不一致，
# 此时请输出“应优先假设 O [III] 谱线未被找峰程序捕获”，并结束 Step 3 的分析。不要输出其他信息。
# 2.仅在有显著的 O [III] 峰值，且红移计算结果与其他谱线基本一致时，进行以下操作：
#     - 使用工具 galaxy_weighted_average，以 flux 为权重计算光谱的红移。
#         - 工具输入为
#             wavelength_obs: List[float],
#             wavelength_rest: List[float],
#             flux: List[float],
#             a: float = {a}, 
#             tolerance: int = {tolerence}, 
#             rms_lambda: float = {rms} 
#     输出红移结果和误差 z ± Δz。
# """ + tail

#         response = await self.call_llm_with_context(prompt, parse_json=False, description="Step 3 综合判断")
#         state['rule_analysis_galaxy'].append(response)
        
#     async def step_4_galaxy(self, state):
#         header = self._common_prompt_header_galaxy(state, include_step_1_only=True)
#         tail = self._common_prompt_tail("Step 4: 补充步骤（假设 Lyα 谱线未被找峰过程捕获）")
        
#         prompt = header + f"""
# 请继续分析:

# Step 4: 补充步骤（假设 O [III] 谱线未被找峰程序捕获）
# - 请抛开前述步骤的分析内容。考虑 Step 1 所选择的峰值谱线实际上可能是 Balmer 线系的 Hα 谱线。
#         - 输出该峰对应谱线的信息：
#             - 观测波长 λ_obs
#             - 流量 Flux
#             - 谱线宽度
#             - 根据 λ_rest，使用工具 calculate_redshift 初步计算红移 z
#         - 使用工具 predict_obs_wavelength 计算在此红移下的其他 Balmer 线
#             - Hδ = 4102.9 A
#             - Hγ = 4341.7 A
#             - Hβ = 4862.7 A
#             - Hα = 6564.6 A
#         的理论位置。光谱中是否有与它们匹配的发射线？其他发射线如
#             - 发射线
#                 - O [II] = 3727.1 Å / 3729.9 Å 双线
#                 - N [II] = 6549.8 Å / 6585.3 Å 双线
#                 - S [II] = 6718.3 Å / 6732.7 Å 双线
#         是否存在？
#         - 如果存在可能的发射线-观测波长匹配结果:
#             - 使用工具 calculate_redshift 分别计算它们的红移。按照“发射线名--静止系波长--观测波长--红移”的格式进行输出。
    
#     - 若以上假设不合理，则假设最强的谷值可能对应 Hα 谱线，重复推断。

# - 抛开其他步骤的分析内容，本节的判断是否支持 O [III] 谱线未被找峰程序捕获的假设？
# """ + tail

#         response = await self.call_llm_with_context(prompt, parse_json=False, description="Step 4 补充步骤")
#         state['rule_analysis_galaxy'].append(response)

#     async def step_5_galaxy(self, state):
#         header = self._common_prompt_header_galaxy(state, include_step_1_only=True)
#         tail = self._common_prompt_tail("Step 5: 补充步骤（Ca 的 K&H 吸收线检测）")

#         prompt = header + """
# 请继续分析:

# Step 5: 补充步骤（Ca 的 K&H 吸收线检测）
# - 请抛开前述步骤的分析内容。假设光谱中prominance的谷值为 Ca 的 K 吸收线。
#         - 输出该峰对应谱线的信息：
#             - 观测波长 λ_obs
#             - 流量 Flux
#             - 谱线宽度
#             - 根据 λ_rest，使用工具 calculate_redshift 初步计算红移 z
#         - 使用工具 predict_obs_wavelength 计算在此红移下的其他主要吸收线
#             - Ca (H) = 3969.6 Å
#             - G-band = 4305.6 Å
#             - Mg = 5176.7 Å
#             - Na = 5895.6 Å
#             - CaT = 8498, 8542, 8662 Å 三线
#         的理论位置。光谱中是否有与它们匹配的谷值？特别注意 Ca 的 H 吸收线。如果该线丢失，则节本判断的可信度低。
#         - 如果存在可能的吸收线线-观测波长匹配结果：
#             - 使用工具 calculate_redshift 分别计算它们的红移。按照“吸收线名--静止系波长--观测波长--红移”的格式进行输出。

# - 抛开其他步骤的分析内容，本节的判断是否支持最明显的谷值为 Ca 的 K 吸收线的假设？
# """ + tail

#         response = await self.call_llm_with_context(prompt, parse_json=False, description="Step 5 补充步骤")
#         state['rule_analysis_galaxy'].append(response)
#     # --------------------------
#     # Run 全流程
#     # --------------------------
    async def run(self, state: SpectroState):
        """执行规则分析完整流程"""
        try:
            await self.describe_spectrum_picture(state)
            ROI_peaks, ROI_troughs = _ROI_features_finding(state)
            # print(f"ROI_peaks:\n{ROI_peaks}")
            # print(f"ROI_troughs:\n{ROI_troughs}")
            state['merged_peaks'], state['merged_troughs'] = merge_features(
                global_peaks=state['peaks'],
                global_troughs=state['troughs'],
                ROI_peaks=ROI_peaks,
                ROI_troughs=ROI_troughs, 
                tol_pixels=10,
            )
            
            plot_merged_features(state)
            
            await self.preliminary_classification(state)
            # print(state['preliminary_classification'])

            _shakespear = await self.preliminary_classification_monkey(state)
            state['possible_object'] = _shakespear
            # print(f"Monkeys types: {_shakespear}")

            if "QSO" in _shakespear:
                await self._QSO(state)
                # await self.step_1_QSO(state)
                # await self.step_2_QSO(state)
                # await self.step_3_QSO(state)
                # await self.step_4_QSO(state)
            # if "Galaxy" in _shakespear:
            #     await self.step_1_galaxy(state)
            #     await self.step_2_galaxy(state)
            #     await self.step_3_galaxy(state)
            #     await self.step_4_galaxy(state)
            #     await self.step_5_galaxy(state)
            return state
        except Exception as e:
            import traceback
            print("❌ An error occurred during spectral analysis:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            # 可选：返回当前状态或抛出异常
            raise  # 如果你希望调用者也能捕获该异常
        


# # ---------------------------------------------------------
# # 3. Revision Supervisor — 负责交叉审核与评估
# # ---------------------------------------------------------
class SpectralAnalysisAuditor(BaseAgent):
    """审查分析师：审查并校正其他分析 agent 的输出"""

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Analysis Auditor',
            mcp_manager=mcp_manager
        )

    def _common_prompt_header(self, state: SpectroState, obj) -> str:
        peak_json = json.dumps(state['peaks'][:10], ensure_ascii=False)
        trough_json = json.dumps(state['troughs'], ensure_ascii=False)
        a = state["pixel_to_value"]["x"]["a"]
        rms = state["pixel_to_value"]["x"]["rms"]
        tolerence = getenv_int("TOL_PIXELS", 10)
        rule_analysis = "\n\n".join(str(item) for item in state['rule_analysis_QSO'])
        prompt_1 = f"""
你是一位严谨的【天文学光谱报告审查分析师】。

任务目标：
- 审核其他分析师的光谱分析报告或想法
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

该报告在红移计算时保留了 3 位小数。

该光谱的波长范围是{state['spectrum']['new_wavelength'][0]} Å 到 {state['spectrum']['new_wavelength'][-1]} Å。
"""
        prompt_2 = f"""

我希望光谱分析报告能够尽可能好地匹配 Lyα、C IV、C III]、Mg II 等典型发射线，但也允许在由于光谱边缘的信号残缺或信噪比不佳导致部分发射线不可见。

同时，在信噪比不佳时，寻找谱线的算法也会受到影响，因此也允许线宽与期望存在一定的的差异。

由于天文学上外流效应的影响，应使用最低电离态的发射线的红移作为光谱红移的最佳结果。

使用工具 QSO_rms 计算红移误差 ± Δz
    - 工具的输入为
        wavelength_rest: List[float], #最低电离态的发射线的静止系波长
        a: float = {a},           
        tolerance: int = {tolerence},     
        rms_lambda = {rms}: float    

如果分析中不支持2条及以上主要谱线（指 Lyα, C IV, C III, Mg II）出现的证据，则首先转向考虑是Galaxy的可能性。
对 Galaxy 的认证无需考虑谱线和红移，仅需从形态上进行分析
"""
        return prompt_1 + prompt_2

    async def auditing(self, state: SpectroState, obj):
        header = self._common_prompt_header(state, obj)

        if state['count'] == 0:
            body = f"""
请对这份分析报告进行检查。
"""
        elif state['count']: 
            auditing_history = state['auditing_history_QSO'][-1] if obj == 'QSO' else state['auditing_history_galaxy'][-1] 
            auditing_history_json = json.dumps(auditing_history, ensure_ascii=False)
            response_history = state['refine_history_QSO'][-1] if obj == 'QSO' else state['refine_history_galaxy'][-1] 
            response_history_json = json.dumps(response_history, ensure_ascii=False)

            body = f"""
你对这份分析报告的最新质疑为
{auditing_history_json}

其他分析师的回答为
{response_history_json}

请回应其他分析师的回答，并继续进行审查。
"""
        prompt = header + body
        response = await self.call_llm_with_context('', prompt, parse_json=False, description="报告审查")
        state['auditing_history_QSO'].append(response) if obj == 'QSO' else state['auditing_history_galaxy'].append(response)

    async def run(self, state: SpectroState) -> SpectroState:
        if 'QSO' in state['possible_object']:
            await self.auditing(state, obj='QSO')
        # if 'Galaxy' in state['possible_object']:
        #     await self.auditing(state, obj='Galaxy')
        return state



# # ---------------------------------------------------------
# # 4. Reflective Analyst — 自由回应审查并改进
# # ---------------------------------------------------------
class SpectralRefinementAssistant(BaseAgent):
    """改进者：回应审查并改进分析"""

    def __init__(self, mcp_manager: MCPManager):
        super().__init__(
            agent_name='Spectral Refinement Assistant',
            mcp_manager=mcp_manager
        )

    def _common_prompt_header(self, state, obj) -> str:
        peak_json = json.dumps(state['peaks'][:10], ensure_ascii=False)
        trough_json = json.dumps(state['troughs'], ensure_ascii=False)
        rule_analysis = "\n\n".join(str(item) for item in state['rule_analysis_QSO'])
        a = state["pixel_to_value"]["x"]["a"]
        rms = state["pixel_to_value"]["x"]["rms"]
        tolerence = getenv_int("TOL_PIXELS", 10)
        prompt_1 = f"""
你是一位具备反思能力的【天文学光谱分析师】。

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

该报告在红移计算时保留了 3 位小数。

该光谱的波长范围是{state['spectrum']['new_wavelength'][0]} Å 到 {state['spectrum']['new_wavelength'][-1]} Å。
"""

        prompt_2 = f"""

我希望光谱分析报告能够尽可能好地匹配 Lyα、C IV、C III]、Mg II 等典型发射线，但也允许在由于光谱边缘的信号残缺或信噪比不佳导致部分发射线不可见。

同时，在信噪比不佳时，寻找谱线的算法也会受到影响，因此也允许线宽与期望存在一定的的差异。

由于天文学上外流效应的影响，应使用最低电离态的发射线的红移作为光谱红移的最佳结果。

使用工具 QSO_rms 计算红移误差 ± Δz
    - 工具的输入为
        wavelength_rest: List[float], # 最低电离态的发射线的静止系波长
        a: float = {a},           
        tolerance: int = {tolerence},     
        rms_lambda = {rms}: float 

如果分析中不支持2条及以上主要谱线（指 Lyα, C IV, C III, Mg II）出现的证据，则首先转向考虑是Galaxy的可能性。
对 Galaxy 的认证无需考虑谱线和红移，仅需从形态上进行分析
"""
        return prompt_1 + prompt_2

    async def refine(self, state: SpectroState, obj):
        header = self._common_prompt_header(state, obj)
        auditing_history = state['auditing_history_QSO'][-1] if obj == 'QSO' else state['auditing_history_galaxy'][-1]
        auditing_history_json = json.dumps(auditing_history, ensure_ascii=False)
        body = f"""
负责核验报告的审查分析师给出的最新建议为
{auditing_history_json}

请对建议进行回应。
"""
        prompt = header + body
        response = await self.call_llm_with_context('', prompt, parse_json=False, description="回应审查")
        state['refine_history_QSO'].append(response) if obj == 'QSO' else state['refine_history_galaxy'].append(response)

    async def run(self, state: SpectroState) -> SpectroState:
        try:
            if 'QSO' in state['possible_object']:
                await self.refine(state, obj='QSO')
            # if 'Galaxy' in state['possible_object']:
            #     await self.refine(state, obj='Galaxy')
            return state
        except Exception as e:
            import traceback
            print("❌ An error occurred during spectral analysis:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            # 可选：返回当前状态或抛出异常
            raise  # 如果你希望调用者也能捕获该异常


# ---------------------------------------------------------
# 🧩 5. Host Integrator — 汇总与总结多方观点
# ---------------------------------------------------------
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
- 输出数据保留 3 位小数
- 只需输出分析内容，无需声明各段分析文字的来源
- 给出最终综合结论及可信度评级（高/中/低）
- 如果仍存在不确定性，请明确指出
- 按格式输出。不要输出多余内容
"""


    async def summary(self, state):
        try:
            preliminary_classification_json = json.dumps(state['preliminary_classification'], ensure_ascii=False)
            visual_interpretation_json = json.dumps(state['visual_interpretation'], ensure_ascii=False)
        except Exception as e:
            print("❌ An error occurred during spectral analysis:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise

        header = self.get_system_prompt()

        prompt_1 = f"""

对光谱的视觉描述
{visual_interpretation_json}

光谱的初步分类
{preliminary_classification_json}
"""
        if "QSO" in state['preliminary_classification']:
            rule_analysis_QSO = "\n\n".join(str(item) for item in state['rule_analysis_QSO'])
            rule_analysis_QSO_json = json.dumps(rule_analysis_QSO, ensure_ascii=False)
            auditing_QSO = "\n\n".join(str(item) for item in state['auditing_history_QSO'])
            auditing_QSO_json = json.dumps(auditing_QSO, ensure_ascii=False)
            refine_QSO = "\n\n".join(str(item) for item in state['refine_history_QSO'])
            refine_QSO_json = json.dumps(refine_QSO, ensure_ascii=False)
            prompt_2 = f"""

规则分析师的观点：
{rule_analysis_QSO_json}

审查分析师的观点：
{auditing_QSO_json}

完善分析师的观点：
{refine_QSO_json}
"""
        prompt_3 = f"""

输出格式如下：

- 光谱的视觉特点
- 分析报告（综合规则分析师、审查分析师和完善分析师的所有观点，逐个 Step 进行结构化输出）
    - Step 1
    - Step 2
    - Step 3
    - Step 4
- 结论
    - 该天体最有可能的的天体类型（Star，Galaxy 还是 QSO），如果分析中不支持2条及以上主要谱线（指 Lyα, C IV, C III, Mg II）出现的证据，则转向考虑是Galaxy的可能性
    - 如果天体是QSO，输出红移 z ± Δz
    - 认证出的谱线（输出 谱线名 - λ_rest - λ_obs - 红移）
    - 光谱的信噪比如何
    - 分析报告的可信度评分（0-4）
        - 对于QSO：
            如果能认证出 2 条以上的主要谱线（指 Lyα, C IV, C III, Mg II），则可信度为 3；
            能认证出 1 条主要谱线，且有其他较弱的特征，则可信度为 2；
            能认证出 1 条主要谱线，但没有其他特征辅助判断，则可信度为 1；
            光谱信噪比极低，含义进行推断，则可信度为 0.
        - 对于 Galaxy
            如果基本满足
    - 是否需要人工介入判断（可信度为 0-2 时必须引入人工判断。其余情况自行决策。）
"""
        prompt = header + prompt_1 + prompt_2 + prompt_3
        response = await self.call_llm_with_context('', prompt, parse_json=False, description="总结")
        state['summary'] = response
    async def in_brief(self, state):
        summary_json = json.dumps(state['summary'], ensure_ascii=False)
        prompt_type = f"""
你是一位负责统筹的【天文学光谱分析主持人】

你已经对一张天文学光谱做了总结
{summary_json}

- 请输出 **结论** 部分中的 **天体类型**（从这三个词语中选择：Star, Galaxy, QSO）

- 输出格式为 str
- 不要输出其他信息
"""
        response_type = await self.call_llm_with_context('', prompt_type, parse_json=False, description="总结")
        state['in_brief']['type'] = response_type

        prompt_redshift = f"""
你是一位负责统筹的【天文学光谱分析主持人】

你已经对一张天文学光谱做了总结
{summary_json}

请输出 **结论** 部分中的 **红移 z**（不需要输出 ± Δz）

- 输出格式为 float
- 不要输出其他信息
"""
        response_redshift = await self.call_llm_with_context('', prompt_redshift, parse_json=False, description="总结")
        state['in_brief']['redshift'] = response_redshift

        prompt_rms = f"""
你是一位负责统筹的【天文学光谱分析主持人】

你已经对一张天文学光谱做了总结
{summary_json}

请输出 **结论** 部分中的 **红移误差 Δz**（不需要输出 z）

- 输出格式为 float
- 不要输出其他信息
"""
        response_rms = await self.call_llm_with_context('', prompt_rms, parse_json=False, description="总结")
        state['in_brief']['rms'] = response_rms

        prompt_human = f"""
你是一位负责统筹的【天文学光谱分析主持人】

你已经对一张天文学光谱做了总结
{summary_json}

请输出 **结论** 部分中的 **是否需要人工介入判断**

- 仅输出“是”或“否”
- 输出格式为 str
- 不要输出其他信息
"""
        response_human = await self.call_llm_with_context('', prompt_human, parse_json=False, description="总结")
        state['in_brief']['human'] = response_human
    
    async def run(self, state: SpectroState) -> SpectroState:
        try:
            await self.summary(state)
            await self.in_brief(state)
            return state
        except Exception as e:
            import traceback
            print("❌ An error occurred during spectral analysis:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            # 可选：返回当前状态或抛出异常
