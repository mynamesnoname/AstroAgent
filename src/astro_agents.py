import json
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

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
    _ROI_features_finding, merge_features, plot_cleaned_features, 
    safe_to_bool, find_overlap_regions
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
        print(tick_pixel_raw)

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
    # Step 1.5 图像裁剪
    # --------------------------
    async def check_border(self, state):
        system_prompt = """
你是一个专业的科学图表分析助手，专注于处理天文学领域的 matplotlib 光谱图。你具备识别图像边缘是否残留坐标轴边框或装饰性直线的能力，并能基于视觉内容做出精准判断。
"""
        user_prompt = """
你将接收到两张图像：
- 一张是原始光谱图像，可能带有绘图边框。
- 一张是经过 OCR 与 OpenCV 预处理后的 matplotlib 天文学光谱图。已尝试裁剪掉原始图表的边框及其外部区域。

请判断图像四条边缘（上、右、下、左）是否仍残留有明显的直线型边框痕迹（例如：长而直的黑色或深色线段，通常为坐标轴外框的一部分）。

判断标准：
- 如果某一边缘**完全看不到**此类直线段，则视为“裁剪干净”。
- 如果某一边缘**仍可见**明显的直线段（即使很细），则视为“未裁剪干净”。

请严格按以下 JSON 格式输出结果，仅包含四个键，值必须为字符串 'true'（表示干净）或 'false'（表示不干净）：

{
    "top": "true" or "false",
    "right": "true" or "false",
    "bottom": "true" or "false",
    "left": "true" or "false"
}

不要输出其他内容。
"""
        response = await self.call_llm_with_context(
            system_prompt,
            user_prompt,
            image_path=[state['image_path'], state['crop_path']],
            parse_json=True,
            description='检查裁剪'
        )
        try:
            response['top'] = safe_to_bool(response['top'])
            response['right'] = safe_to_bool(response['right'])
            response['bottom'] = safe_to_bool(response['bottom'])
            response['left'] = safe_to_bool(response['left'])
            return response
        except:
            logging.error(f"LLM 输出格式错误: {response}")

    async def peak_trough_detection(self, state: SpectroState):
        try:
            sigma_list, tol_pixels, prom_peaks, prom_troughs, weight_original, _, _ = _load_feature_params()
            state['sigma_list'] = sigma_list

            spec = state["spectrum"]
            wavelengths = np.array(spec["new_wavelength"])
            flux = np.array(spec["weighted_flux"])

            state["peaks"] = _find_features_multiscale(
                wavelengths, flux,
                state, feature="peak", sigma_list=sigma_list,
                prom=prom_peaks, tol_pixels=tol_pixels, weight_original=weight_original,
                use_continuum_for_trough=True
            )
            state["troughs"] = _find_features_multiscale(
                wavelengths, flux,
                state, feature="trough", sigma_list=sigma_list,
                prom=prom_troughs, tol_pixels=tol_pixels, weight_original=weight_original,
                use_continuum_for_trough=True,
                min_depth=0.08
            )
            # print(len(state["peaks"]), len(state["troughs"]))

            # 把wavelengths按照每500埃为一个ROI进行划分，分别进行峰谷检测
            ROI_peaks = []
            ROI_troughs = []
            roi_size = 500  # 每个ROI的宽度，单位为埃
            roi_edges = np.arange(wavelengths[0], wavelengths[-1], roi_size)
            for i in range(len(roi_edges)-1):
                roi_start = roi_edges[i]
                roi_end = roi_edges[i+1]
                mask = (wavelengths >= roi_start) & (wavelengths < roi_end)
                roi_wavelengths = np.where(mask, wavelengths, 0)
                roi_flux = np.where(mask, flux, 0)
                # roi_wavelengths = wavelengths[mask]
                # roi_flux = flux[mask]
                # 如果roi_wavelengths长度非0
                if len(roi_wavelengths) == 0:
                    continue
                roi_peaks = _find_features_multiscale(
                    roi_wavelengths, roi_flux,
                    state, feature="peak", sigma_list=sigma_list,
                    prom=prom_peaks, tol_pixels=tol_pixels, weight_original=weight_original,
                    use_continuum_for_trough=True
                )
                roi_troughs = _find_features_multiscale(
                    roi_wavelengths, roi_flux,
                    state, feature="trough", sigma_list=sigma_list,
                    prom=prom_troughs, tol_pixels=tol_pixels, weight_original=weight_original,
                    use_continuum_for_trough=True,
                    min_depth=0.08
                )
                ROI_peaks.extend(roi_peaks)
                ROI_troughs.extend(roi_troughs)
            roi_edges_ = roi_edges + 250
            for i in range(len(roi_edges_)-1):
                roi_start = roi_edges_[i]
                roi_end = roi_edges_[i+1]
                mask = (wavelengths >= roi_start) & (wavelengths < roi_end)
                # roi_wavelengths长度与wavelengths相同，mask之外的位置为0，mask内的位置为原始值
                roi_wavelengths = np.where(mask, wavelengths, 0)
                roi_flux = np.where(mask, flux, 0)
                # roi_wavelengths = wavelengths[mask]
                # roi_flux = flux[mask]
                # 如果roi_wavelengths长度非0
                if len(roi_wavelengths) == 0:
                    continue
                roi_peaks = _find_features_multiscale(
                    roi_wavelengths, roi_flux,
                    state, feature="peak", sigma_list=sigma_list,
                    prom=prom_peaks, tol_pixels=tol_pixels, weight_original=weight_original,
                    use_continuum_for_trough=True
                )
                roi_troughs = _find_features_multiscale(
                    roi_wavelengths, roi_flux,
                    state, feature="trough", sigma_list=sigma_list,
                    prom=prom_troughs, tol_pixels=tol_pixels, weight_original=weight_original,
                    use_continuum_for_trough=True,
                    min_depth=0.08
                )
                ROI_peaks.extend(roi_peaks)
                ROI_troughs.extend(roi_troughs)
            state["ROI_peaks"] = ROI_peaks
            state["ROI_troughs"] = ROI_troughs
            state['merged_peaks'], state['merged_troughs'] = merge_features(
                wavelengths, flux,
                global_peaks=state["peaks"],
                global_troughs=state["troughs"],
                ROI_peaks=state["ROI_peaks"],
                ROI_troughs=state["ROI_troughs"],
                tol_pixels=tol_pixels
            )
        except Exception as e:
            print(f"❌ peak_trough_detection: {e}")
        return state

    async def continuum_fitting(self, state: SpectroState):
        """简单的continuum拟合"""
        try:
            spec = state["spectrum"]
            wavelengths = np.array(spec["new_wavelength"])
            flux = np.array(spec["weighted_flux"])

            band_name = state['band_name']
            band_wavelength = state['band_wavelength']
            print('cut continuum')
            if band_name:
                overlap_regions = find_overlap_regions(band_name, band_wavelength)
                # 初始化 mask 为全 False
                mask = np.zeros_like(wavelengths, dtype=bool)
                for key in overlap_regions:
                    low, high = overlap_regions[key]
                    region_mask = (wavelengths >= low) & (wavelengths <= high)
                    mask = mask | region_mask  # 或者用 mask |= region_mask
                wavelengths = wavelengths[~mask]
                flux = flux[~mask]

            sigma_contunuum = getenv_int('CONTINUUM_SMOOTHING_SIGMA', None)
            print(f'CONTINUUM_SMOOTHING_SIGMA: {sigma_contunuum}')
            if sigma_contunuum == None:
                logging.error("CONTINUUM_SMOOTHING_SIGMA 未设置")
                return
            continuum_flux = gaussian_filter1d(flux, sigma=sigma_contunuum)
            state['continuum'] = {
                'wavelength': wavelengths.tolist(),
                'flux': continuum_flux.tolist()
            }
        except Exception as e:
            print(f"❌ continuum_fitting: {e}")
        return state

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
            print(state["OCR_detected_ticks"])
            # Step 1.3: 合并
            await self.combine_axis_mapping(state)
            # Step 1.4: 修正
            await self.revise_axis_mapping(state)
            # Step 1.5: 边框检测与裁剪
            state['margin'] = {
                'top': 20,
                'right': 10,
                'bottom': 15,
                'left': 10,
            }
            stop = False
            while stop is False:
                state["chart_border"] = _detect_chart_border(state['image_path'], state['margin'])
                _crop_img(state['image_path'], state["chart_border"], state['crop_path'])
                box_new = await self.check_border(state)
                values = [box_new['top'], box_new['bottom'], box_new['left'], box_new['right']]
                margin = [state['margin']['top'], state['margin']['right'], state['margin']['bottom'], state['margin']['left']] 
                if all(values):  # 所有都是 True（非零/非False）
                    stop = True
                elif any(m > 30 for m in margin):
                    stop = True
                else:
                    for k, v in box_new.items():
                        if v == True:
                            state['margin'][k] = state['margin'][k]
                        else:
                            state['margin'][k] = state['margin'][k] + 2
                print(f"box_new: {box_new}")
                print(f"margin: {state['margin']}")
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
            # Step 1.10: 检测峰值/谷值
            await self.peak_trough_detection(state)
            print(f"Detected {len(state['merged_peaks'])} peaks and {len(state['merged_troughs'])} troughs.")
            # Step 1.10.5: continuum拟合
            await self.continuum_fitting(state)
            # Step 1.11: 可选绘图
            if plot:
                try:
                    state["spectrum_fig"] = _plot_spectrum(state)
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
            band_name = state['band_name']
            band_wavelength = state['band_wavelength']

            if not band_name or not band_wavelength:
                return {
                    "filter_noise": 'false',
                    "filter_noise_wavelength": None
                }
            else:
                # 找出重叠区域
                overlap_regions = find_overlap_regions(band_name, band_wavelength)
                spec = state['spectrum']
                wl = np.array(spec['new_wavelength'])
                d_f = np.array(spec['delta_flux'])

                system_prompt = function_prompt['_filter_noise']['system_prompt']
                band_name_json = json.dumps(band_name, ensure_ascii=False)
                ham = f"""
本光谱的 camera/filters 名为
{band_name_json}
下面是光谱在 camera/filters 交界区域的样本数据。
"""
                for key in overlap_regions.keys():
                    overlap = overlap_regions[key]
                    scale = overlap[1] - overlap[0]
                    scale = scale * 2
                    center = (overlap[0] + overlap[1]) / 2
                    left = center - scale / 2
                    right = center + scale / 2
                    mask = (wl >= left) & (wl <= right)
                    wl_t = wl[mask]
                    wl_t = wl_t.tolist()
                    wl_t_json = json.dumps(wl_t, ensure_ascii=False)
                    delta_t = d_f[mask]
                    delta_t = delta_t.tolist()
                    delta_t_json = json.dumps(delta_t, ensure_ascii=False)

                    ham += f"""
交界区域 {key}:
波长：{wl_t_json}
Flux 误差：{delta_t_json}
"""
                user_prompt = function_prompt['_filter_noise']['user_prompt']
                user_prompt = ham + user_prompt

                response = await self.call_llm_with_context(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_path=None,
                    parse_json=True,
                    description="Filter噪声判断"
                )
                return(response)
            
        async def _cleaning(state):
            filter_nosie = state['visual_interpretation'][0]
            if not safe_to_bool(filter_nosie.get('filter_noise', False)):
                state['cleaned_peaks'] = state['merged_peaks']
                state['cleaned_troughs'] = state['merged_troughs']
            else:
                filter_noise_wl = filter_nosie.get('filter_noise_wavelength', [])
                filter_noise_wl = np.array(filter_noise_wl)
                wavelength = np.array(state['spectrum']['new_wavelength'])
                peaks = state['merged_peaks']
                cleaned_peaks = []
                wiped_peaks = []
                for p in peaks:
                    wl = p['wavelength']
                    width = p['width_mean']

                    distance = abs(wl - filter_noise_wl)
                    # 如果在distance中至少有一个值小于 width，则认为该峰在噪声区域内
                    if np.any(distance <= width):
                        is_artifact = True
                    else:
                        is_artifact = False
                    if not is_artifact:
                        if p['width_in_km_s'] is not None and p['wavelength'] > wavelength[0]:
                            if p['width_in_km_s'] > 2000:
                                p['describe'] = '宽线'
                            elif p['width_in_km_s'] < 1000:
                                p['describe'] = '窄线'
                            else:
                                p['describe'] = '中等宽度'
                            cleaned_peaks.append(p)
                    else:
                        wiped_peaks.append(p)
                state['cleaned_peaks'] = cleaned_peaks
                state['wiped_peaks'] = wiped_peaks

                cleaned_troughs = []
                for t in state['merged_troughs']:
                    wl = t['wavelength']
                    distance = abs(wl - filter_noise_wl)
                    if np.any(distance <= width):
                        is_artifact = True
                    else:
                        is_artifact = False
                    if not is_artifact:
                        if t['width_in_km_s'] is not None and t['wavelength'] > wavelength[0]:
                            if t['width_in_km_s'] > 2000:
                                t['describe'] = '宽谷'
                            elif t['width_in_km_s'] < 1000:
                                t['describe'] = '窄谷'
                            else:
                                t['describe'] = '中等宽度'
                        else:
                            t['describe'] = '未处理'
                        cleaned_troughs.append(t)
                state['cleaned_troughs'] = cleaned_troughs
            return state

        async def _visual(state):
            system_prompt = function_prompt['_visual']['system_prompt']
            user_prompt_1 = function_prompt['_visual']['user_prompt_continuum']
            response_1 = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt_1,
                image_path=state['continuum_path'],
                parse_json=True,
                description="视觉光谱定性描述——continuum"
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

            response_1_json = json.dumps(response_1, ensure_ascii=False)
            response_2_json = json.dumps(response_2, ensure_ascii=False)
            response_3_json = json.dumps(response_3, ensure_ascii=False)
            return '\n'.join([response_1_json, response_2_json, response_3_json])

        async def _integrate(state):
            visual_json       = json.dumps(state['visual_interpretation'][1], ensure_ascii=False)

            system_prompt = function_prompt['_integrate']['system_prompt']
            ham = f"""
{visual_json}
"""
            user_prompt_integrate = function_prompt['_integrate']['user_prompt'] + ham
            response = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt_integrate,
                parse_json=True,
                description="视觉光谱定性描述"
            )
            return response

        result_filter_noise = await _filter_noise(state)
        state['visual_interpretation'] = [result_filter_noise]
        await _cleaning(state)
        result_visual = await _visual(state)
        state['visual_interpretation'].append(result_visual)
        result_integrate = await _integrate(state)
        state['visual_interpretation'] = result_integrate

        visual_interpretation_path = os.path.join(state['output_dir'], f'{state['image_name']}_visual_interpretation.txt')
        with open(visual_interpretation_path, 'w', encoding='utf-8') as f:
            json_str = json.dumps(state['visual_interpretation'], indent=2, ensure_ascii=False)
            f.write(json_str)
        print('finished describe_spectrum_picture')
    
    async def preliminary_classification(self, state: SpectroState) -> str:
        """初步分类：根据光谱形态初步判断天体类型"""

        continuum_interpretation_json = json.dumps(state['visual_interpretation']['continuum_description'], ensure_ascii=False)
        line_interpretation_json = json.dumps(state['visual_interpretation']['lines_description'], ensure_ascii=False)

        system_prompt = """
你是一位经验丰富的天文学光谱分析助手。

你的任务是根据光谱的定性描述和特征数据，猜测天体可能属于的类别。

如果连续谱呈现蓝端较高，红端较低的趋势（即高→低），则该天体为 QSO；
如果连续谱呈现蓝端较低，中段较高，红端下降的趋势（即低→高→低），则该天体为 QSO ；

如果连续谱呈现蓝端较低，红端较高的趋势（即低→高），则该天体为 Galaxy ；

比较两种光源的概率，给出你的选择。

输出天体类别，格式为 json，格式如下：
{
    'type': str,  # 天体类别，可能的取值为 "Galaxy", "QSO"
}

仅输出唯一选项。不要输出其他信息。
"""
        user_prompt = f"""
请根据以下光谱数据进行分析：

前一位天文学助手已经定性地描述了光谱的整体形态：
{continuum_interpretation_json}

请根据描述和图像，猜测该光谱可能属于哪一类天体。
"""+"""
输出为 json，格式如下：
{
    'type': str,  # 天体类别，可能的取值为 "Galaxy", "QSO"
}
"""
#         user_prompt = f"""
# 请根据图像，猜测该光谱可能属于哪一类天体。
# """
        response = await self.call_llm_with_context(
            system_prompt = system_prompt,
            user_prompt = user_prompt,
            image_path=None,
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
"""+"""
格式为 json，格式如下：
{
    'type': str,  # 天体类别，可能的取值为 "Galaxy", "QSO"
}
不要输出其他内容
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
        try:
            peaks_info = [
                {
                    "wavelength": pe.get('wavelength'),
                    "flux": pe.get('mean_flux'),
                    "width": pe.get('width_mean'),
                    "width_in_km_s": pe.get('width_in_km_s'),
                    "prominance": pe.get('max_prominence'),
                    "seen_in_max_global_smoothing_scale_sigma": pe.get('max_global_sigma_seen', None),
                    "seen_in_max_local_smoothing_scale_sigma": pe.get('max_roi_sigma_seen', None),
                    "describe": pe.get('describe')
                }
                for pe in state.get('cleaned_peaks', [])[:15]
            ]
            peak_json = json.dumps(peaks_info, ensure_ascii=False)

            # 初始化Lyα候选线列表
            Lyalpha_candidate = []

            # 获取光谱波长范围
            wl_left = state['spectrum']['new_wavelength'][0]
            wl_right = state['spectrum']['new_wavelength'][-1]
            mid_wavelength = (wl_left + wl_right) / 2

            # 筛选条件1：优先使用全局平滑尺度的信噪比
            for peak in peaks_info:
                # 检查谱线宽度是否足够（>=2000 km/s）
                if peak['width_in_km_s'] is not None and peak['width_in_km_s'] >= 2000:
                    # 检查全局平滑尺度的信噪比条件
                    if (peak['seen_in_max_global_smoothing_scale_sigma'] is not None and 
                        peak['seen_in_max_global_smoothing_scale_sigma'] > 2):
                        Lyalpha_candidate.append(peak['wavelength'])

            # 筛选条件2：如果条件1没有找到候选，使用局部平滑尺度的信噪比
            if len(Lyalpha_candidate) == 0:
                for peak in peaks_info:
                    if peak['width_in_km_s'] is not None and peak['width_in_km_s'] >= 2000:
                        # 检查局部平滑尺度的信噪比条件
                        if (peak['seen_in_max_local_smoothing_scale_sigma'] is not None and 
                            peak['seen_in_max_local_smoothing_scale_sigma'] > 2):
                            Lyalpha_candidate.append(peak['wavelength'])

            # 将候选线转换为JSON格式并打印
            Lyalpha_candidate_json = json.dumps(Lyalpha_candidate, ensure_ascii=False)
            print(f"Lyalpha_candidate: {Lyalpha_candidate}")

            trough_info = [
                {
                    "wavelength": tr.get('wavelength'),
                    "flux": tr.get('mean_flux'),
                    "width": tr.get('width_mean'),
                    "seen_in_scales_of_sigma": tr.get('seen_in_scales_of_sigma')
                }
                for tr in state.get('cleaned_troughs', [])[:15]
            ]
            trough_json = json.dumps(trough_info, ensure_ascii=False)
            # print(f"trough_info: {trough_info}")
        except Exception as e:
            logging.error(f"Error in _QSO: {e}")
            raise e

        def _common_prompt_header_QSO(state, include_rule_analysis=True, include_step_1_only=False):
            """构造每个 step 公共的 prompt 前段"""
            try:
                visual_json = json.dumps(state['visual_interpretation'], ensure_ascii=False)
                # peak_json = json.dumps(state['peaks'][:10], ensure_ascii=False)
                # trough_json = json.dumps(state['troughs'], ensure_ascii=False)
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
综合原曲线和 smoothing 尺度为 sigma={state['sigma_list']} 的高斯平滑曲线，使用 scipy 函数进行了峰/谷识别。
关于峰/谷的讨论以以下数据为准：
- 代表性的前 10 条发射线：
{peak_json}
- 可能的吸收线：
{trough_json}
- 波长误差在 ~ ±{tol_wavelength/2} Å 的量级或更大
"""
                return header
            except Exception as e:
                logging.error(f"Error in _common_prompt_header_QSO: {e}")
                raise e

        def _common_prompt_tail(step_title, extra_notes=""):
            """构造每个 step 公共尾部，保留 step 特有输出/分析指示"""
            try:
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
            except Exception as e:
                logging.error(f"Error in _common_prompt_tail: {e}")
                raise e
        
        async def step_1_QSO(state):
            try:
                print("Step 1: Lyα 谱线检测")
                header = _common_prompt_header_QSO(state, include_rule_analysis=False)
                tail = _common_prompt_tail("Step 1: Lyα 谱线检测")
                if len(Lyalpha_candidate) > 0:
                    candidate_str = f"\n算法筛选的 Lyα 候选线包括：\n{Lyalpha_candidate_json}\n你也可以自己推测其他选项。\n"
                else:
                    candidate_str = ""

                system_prompt = header + tail
                user_prompt = f"""
请按以下步骤分析:

Step 1: Lyα 谱线检测
假设该光谱中存在 Lyα 发射线（λ_rest = 1216 Å）：
{candidate_str}
1. 在光谱流量较大，大 smoothing 尺度可见且有一定宽度的峰中，推测哪条最可能为 Lyα 线。
    - 从提供的峰列表中选择
    - 候选谱线宽度相近（20 Å 以内）时，优先考虑流量更高的峰。
2. 输出：
- 观测波长 λ_obs
- 流量 Flux
- 谱线宽度
3. 使用工具 calculate_redshift 计算该峰为 Lyα 发射线时的红移 z。
4. 检查蓝端（短波长方向）是否存在 Lyα forest 特征：吸收线相对更密集、较窄且分布在 Lyα 蓝端附近。请指出并进行简短说明。
""" 
                
                response = await self.call_llm_with_context(
                    system_prompt=system_prompt, 
                    user_prompt=user_prompt, 
                    parse_json=True, 
                    description="Step 1 Lyα 分析"
                )
                state['rule_analysis_QSO'].append(response)
            except Exception as e:
                logging.error(f"Error in step_1_QSO: {e}")
                raise e

        async def step_2_QSO(state):
            print("Step 2: 其他显著发射线分析")
            try:
                header = _common_prompt_header_QSO(state)
                tail = _common_prompt_tail("Step 2: 其他显著发射线分析")
                system_prompt = header + tail

                band_name = state['band_name']
                band_wavelength = state['band_wavelength']
                if band_name: 
                    overlap_regions = find_overlap_regions(band_name, band_wavelength)
                    wws = np.max([wp.get('width_mean') for wp in state.get('wiped_peaks', [])[:5]])
                    print(f"wws: {wws}")
                    for key in overlap_regions:
                        range = overlap_regions[key]
                        overlap_regions[key] = [range[0]-wws, range[1]+wws] # Broaden the overlap regions to make sure LLM won't miss them
                    overlap_regions_json = json.dumps(overlap_regions, ensure_ascii=False)
                    wiped = [
                        {
                            "wavelength": wp.get('wavelength'),
                            "flux": wp.get('mean_flux'),
                            "width": wp.get('width_mean'),
                            # "seen_in_scales_of_sigma": wp.get('seen_in_scales_of_sigma')
                        }
                        for wp in state.get('wiped_peaks', [])[:5]
                    ]
                    wiped_json = json.dumps(wiped, ensure_ascii=False)
                    advanced = f"""\n    - 注意：如果某些理论峰值落在以下区间附近：\n        {overlap_regions_json}\n    则峰值可能被当作噪声信号清除。这些峰值是：\n        {wiped_json}\n    请优先考虑这些因素，再次分析"""
                else:
                    advanced = ""

                user_prompt = f"""
请继续分析:

Step 2: 其他显著发射线分析
1. 在 Step 1 得到的红移下，使用工具 predict_obs_wavelength 计算以下三条主要发射线：C IV 1549, C III] 1909, Mg II 2799 在光谱中的理论位置。
2. 提示词提供的光谱中是否有与三者相匹配的峰？{advanced}
3. 如果存在发射线与观测峰值的匹配，根据匹配结果，分别使用工具 calculate_redshift 计算红移。按“发射线名--静止系波长--观测波长--红移”的格式输出。
"""

                response = await self.call_llm_with_context(system_prompt, user_prompt, parse_json=False, description="Step 2 发射线分析")
                state['rule_analysis_QSO'].append(response)
            except Exception as e:
                logging.error(f"Error in step_2_QSO: {e}")
                raise e

        async def step_3_QSO(state):
            try:
                header = _common_prompt_header_QSO(state)
                tail = _common_prompt_tail("Step 3: 综合判断")
                system_prompt = header + tail

                user_prompt = """
请继续分析:

Step 3: 综合判断
1. 在 Step 1 到 Step 2 中，如果：
    - C IV 和 C III] 两条主要谱线存在缺失或大幅偏移
    - 使用 lyα 谱线计算的红移与其他谱线的计算结果不一致，
此时请输出“应优先假设 Lyα 谱线未被找峰程序捕获”，并结束 Step 3 的分析。不要输出其他信息。
2.仅在有显著的 Lyα 峰值，且红移计算结果与其他谱线基本一致时，进行以下操作：
    - 因为天文学中存在外流等现象，请将当前所有匹配中**最低电离态谱线的红移**作为光谱的红移。输出红移结果。（因为存在不对称和展宽，Lyα的置信度是较低的）
"""
                response = await self.call_llm_with_context(system_prompt, user_prompt, parse_json=False, description="Step 3 综合判断")
                state['rule_analysis_QSO'].append(response)
            except Exception as e:
                logging.error(f"Error in step_3_QSO: {e}")
                raise e
            
        async def step_4_QSO(state):
            try: 
                header = _common_prompt_header_QSO(state, include_step_1_only=True)
                tail = _common_prompt_tail("Step 4: 补充步骤（假设 Step 1 所选择的谱线并非 Lyα）")
                system_prompt = header + tail

                user_prompt = """
请继续分析:

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

- 注意：允许在由于光谱边缘的信号残缺或信噪比不佳导致部分发射线不可见。
""" + tail

                response = await self.call_llm_with_context(system_prompt, user_prompt, parse_json=False, description="Step 4 补充分析")
                state['rule_analysis_QSO'].append(response)
            except Exception as e:
                logging.error(f"Error in step_4_QSO: {e}")
                raise e
        
        await step_1_QSO(state)
        await step_2_QSO(state)
        await step_3_QSO(state)
        await step_4_QSO(state)

#     # --------------------------
#     # Run 全流程
#     # --------------------------
    async def run(self, state: SpectroState):
        """执行规则分析完整流程"""
        try:
            await self.describe_spectrum_picture(state)

            plot_cleaned_features(state)
            await self.preliminary_classification(state)
            print(state['preliminary_classification'])

            # _shakespear = await self.preliminary_classification_monkey(state)
            # state['possible_object'] = _shakespear
            # print(f"Monkeys types: {_shakespear}")

            if state['preliminary_classification']['type'] == "QSO":
                await self._QSO(state)
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

    def _common_prompt_header(self, state: SpectroState) -> str:
        try:
            peaks_info = [
                {
                    "wavelength": pe.get('wavelength'),
                    "flux": pe.get('mean_flux'),
                    "width": pe.get('width_mean'),
                    "prominance": pe.get('max_prominence'),
                    "seen_in_scales_of_sigma": pe.get('seen_in_scales_of_sigma'),
                    "describe": pe.get('describe')
                }
                for pe in state.get('cleaned_peaks', [])[:15]
            ]
            peak_json = json.dumps(peaks_info, ensure_ascii=False)
            trough_info = [
                {
                    "wavelength": tr.get('wavelength'),
                    "flux": tr.get('mean_flux'),
                    "width": tr.get('width_mean'),
                    "seen_in_scales_of_sigma": tr.get('seen_in_scales_of_sigma'), 
                }
                for tr in state.get('cleaned_troughs', [])[:15]
            ]
            trough_json = json.dumps(trough_info, ensure_ascii=False)
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
            band_name = state['band_name']
            band_wavelength = state['band_wavelength']
            if band_name: 
                overlap_regions = find_overlap_regions(band_name, band_wavelength)
                wws = np.max([wp.get('width_mean') for wp in state.get('wiped_peaks', [])[:5]])
                for key in overlap_regions:
                    range = overlap_regions[key]
                    overlap_regions[key] = [range[0]-wws, range[1]+wws] # Broaden the overlap regions to make sure LLM won't miss them
                overlap_regions_json = json.dumps(overlap_regions, ensure_ascii=False)
                wiped = [
                    {
                        "wavelength": wp.get('wavelength'),
                        "flux": wp.get('mean_flux'),
                        "width": wp.get('width_mean'),
                        # "seen_in_scales_of_sigma": wp.get('seen_in_scales_of_sigma')
                    }
                    for wp in state.get('wiped_peaks', [])[:5]
                ]
                wiped_json = json.dumps(wiped, ensure_ascii=False)
                advanced = f"""如果报告中的峰值落在以下区间附近\n    {overlap_regions_json}\n则峰值可能被当作噪声信号清除。这些峰值是：\n      {wiped_json}\n请注意考察这些峰值作为 C IV 或 C III] 的可能性"""
            else:
                advanced = ""
            prompt_2 = f"""

我希望光谱分析报告能够尽可能好地匹配 Lyα、C IV、C III]、Mg II 等典型发射线，但也允许在由于光谱边缘的信号残缺或信噪比不佳导致部分发射线不可见。

同时，在信噪比不佳时，寻找谱线的算法也会受到影响，因此也允许线宽与期望存在一定的的差异。

如果 Lyα 谱线应该在光谱范围内，但却未被报告列出，请显著降低该报告的可信度。

如果 Lyα 谱线被报告列出，请检查 Lyα 谱线与其他谱线的流量大小。如果 Lyα 流量显著低于其他谱线（如 C IV、C III]），请指出并降低该报告的可信度。

由于天文学上外流效应的影响，应使用最低电离态的发射线的红移作为光谱红移的最佳结果。

使用工具 QSO_rms 计算红移误差 ± Δz
    - 工具的输入为
        wavelength_rest: List[float], # 最低电离态的发射线的静止系波长（Lyα易受展宽影响，不适用于此处，尽量选择Lyα外的谱线）
        a: float = {a},           
        tolerance: int = {tolerence},     
        rms_lambda = {rms}: float    
"""
            return prompt_1 + advanced + prompt_2
        except Exception as e:
            print(f"Error in _common_prompt_header: {e}")
            return ""

    async def auditing(self, state: SpectroState):
        try:
            system_prompt = self._common_prompt_header(state)

            if state['count'] == 0:
                body = f"""
请对这份分析报告进行检查。
"""
            elif state['count']: 
                auditing_history = state['auditing_history_QSO'][-1] 
                auditing_history_json = json.dumps(auditing_history, ensure_ascii=False)
                response_history = state['refine_history_QSO'][-1]
                response_history_json = json.dumps(response_history, ensure_ascii=False)

                body = f"""
你对这份分析报告的最新质疑为
{auditing_history_json}

其他分析师的回答为
{response_history_json}

请回应其他分析师的回答，并继续进行审查。
"""
            user_prompt = body
            response = await self.call_llm_with_context(system_prompt, user_prompt, parse_json=False, description="报告审查")
            state['auditing_history_QSO'].append(response)
        except Exception as e:
            print(f"Error in auditing: {e}")

    async def run(self, state: SpectroState) -> SpectroState:
        if state['preliminary_classification']['type'] == "QSO":
            await self.auditing(state)
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

    def _common_prompt_header(self, state) -> str:
        try:
            peaks_info = [
                {
                    "wavelength": pe.get('wavelength'),
                    "flux": pe.get('mean_flux'),
                    "width": pe.get('width_mean'),
                    "prominance": pe.get('max_prominence'),
                    "seen_in_global_scales_of_sigma": pe.get('max_global_sigma_seen', None),
                    "describe": pe.get('describe')
                }
                for pe in state.get('cleaned_peaks', [])[:15]
            ]
            peak_json = json.dumps(peaks_info, ensure_ascii=False)

            trough_info = [
                {
                    "wavelength": tr.get('wavelength'),
                    "flux": tr.get('mean_flux'),
                    "width": tr.get('width_mean'),
                    "seen_in_scales_of_sigma": tr.get('seen_in_scales_of_sigma')
                }
                for tr in state.get('cleaned_troughs', [])[:15]
            ]
            trough_json = json.dumps(trough_info, ensure_ascii=False)
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
            band_name = state['band_name']
            band_wavelength = state['band_wavelength']
            if band_name: 
                overlap_regions = find_overlap_regions(band_name, band_wavelength)
                wws = np.max([wp.get('width_mean') for wp in state.get('wiped_peaks', [])[:5]])
                for key in overlap_regions:
                    range = overlap_regions[key]
                    overlap_regions[key] = [range[0]-wws, range[1]+wws] # Broaden the overlap regions to make sure LLM won't miss them
                overlap_regions_json = json.dumps(overlap_regions, ensure_ascii=False)
                wiped = [
                    {
                        "wavelength": wp.get('wavelength'),
                        "flux": wp.get('mean_flux'),
                        "width": wp.get('width_mean'),
                        # "seen_in_scales_of_sigma": wp.get('seen_in_scales_of_sigma')
                    }
                    for wp in state.get('wiped_peaks', [])[:5]
                ]
                wiped_json = json.dumps(wiped, ensure_ascii=False)
                advanced = f"""如果报告中的峰值落在以下区间附近\n    {overlap_regions_json}\n则峰值可能被当作噪声信号清除。这些峰值是：\n      {wiped_json}\n请注意考察这些峰值作为 C IV 或 C III] 的可能性"""
            else:
                advanced = ""

            prompt_2 = f"""

我希望光谱分析报告能够尽可能好地匹配 Lyα、C IV、C III]、Mg II 等典型发射线，但也允许在由于光谱边缘的信号残缺或信噪比不佳导致部分发射线不可见。

同时，在信噪比不佳时，寻找谱线的算法也会受到影响，因此也允许线宽与期望存在一定的的差异。

如果 Lyα 谱线应该在光谱范围内，但却未被报告列出，请显著降低该报告的可信度。

如果 Lyα 谱线被报告列出，请检查 Lyα 谱线与其他谱线的流量大小。如果 Lyα 流量显著低于其他谱线（如 C IV、C III]），请指出并降低该报告的可信度。

由于天文学上外流效应的影响，应使用最低电离态的发射线的红移作为光谱红移的最佳结果（Lyα易受展宽影响，不适用于此处，尽量选择Lyα外的谱线）。

使用工具 QSO_rms 计算红移误差 ± Δz
    - 工具的输入为
        wavelength_rest: List[float], # 最低电离态的发射线的静止系波长
        a: float = {a},           
        tolerance: int = {tolerence},     
        rms_lambda = {rms}: float 
"""
            return prompt_1 + advanced + prompt_2
        except Exception as e:
            logging.error(f"Error in _common_prompt_header: {e}")
            raise e


    async def refine(self, state: SpectroState):
        try:
            system_prompt = self._common_prompt_header(state)
            auditing_history = state['auditing_history_QSO'][-1]
            auditing_history_json = json.dumps(auditing_history, ensure_ascii=False)
            body = f"""
负责核验报告的审查分析师给出的最新建议为
{auditing_history_json}

请对建议进行回应。
"""
            user_prompt = body
            response = await self.call_llm_with_context(system_prompt, user_prompt, parse_json=False, description="回应审查")
            state['refine_history_QSO'].append(response)
        except Exception as e:
            logging.error(f"Error in refine: {e}")
            raise e

    async def run(self, state: SpectroState) -> SpectroState:
        try:
            if state['preliminary_classification']['type'] == "QSO":
                await self.refine(state)
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

        prompt_1 = f"""

对光谱的视觉描述
{visual_interpretation_json}

光谱的初步分类
{preliminary_classification_json}
"""
        system_prompt = self.get_system_prompt() + prompt_1

        if state['preliminary_classification']['type'] == "QSO":
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
    - 该天体的天体类型（Galaxy 还是 QSO）
    - 如果天体是QSO，输出红移 z ± Δz
    - 认证出的谱线（输出 谱线名 - λ_rest - λ_obs - 红移）
    - 光谱的信噪比如何
    - 分析报告的可信度评分（0-4）
        如果能认证出 2 条以上的主要谱线（指 Lyα, C IV, C III, Mg II），则可信度为 3；
        能认证出 1 条主要谱线，且有其他较弱的特征，则可信度为 2；
        能认证出 1 条主要谱线，但没有其他特征辅助判断，则可信度为 1；
        光谱信噪比极低，含义进行推断，则可信度为 0.
    - 是否需要人工介入判断（可信度为 0-2 时必须引入人工判断。信噪比较低且无 Lyα 时必须引入人工判断。其余情况自行决策。）
"""
            user_prompt = prompt_2 + prompt_3
        else:
            user_prompt = f"""
输出格式如下：

- 光谱的视觉特点
- 结论
    - 该天体的天体类型（Galaxy 还是 QSO）
    - 如果天体是QSO，输出红移 z ± Δz
    - 认证出的谱线（输出 谱线名 - λ_rest - λ_obs - 红移）
    - 光谱的信噪比如何
    - 分析报告的可信度评分（0-4）
        如果认证为 Galaxy，则可信度为 2；否则为 0。
    - 是否需要人工介入判断（如果类型为Galaxy，则必须要求人工介入判断）
"""
        response = await self.call_llm_with_context(system_prompt, user_prompt, parse_json=False, description="总结")
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

- 输出格式为 float 或 None
- 不要输出其他信息
"""
        response_redshift = await self.call_llm_with_context('', prompt_redshift, parse_json=False, description="总结")
        state['in_brief']['redshift'] = response_redshift

        prompt_rms = f"""
你是一位负责统筹的【天文学光谱分析主持人】

你已经对一张天文学光谱做了总结
{summary_json}

请输出 **结论** 部分中的 **红移误差 Δz**（不需要输出 z）

- 输出格式为 float 或 None
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
