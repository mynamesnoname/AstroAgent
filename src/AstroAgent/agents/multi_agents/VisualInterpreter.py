import json
import os
import numpy as np
import logging

from AstroAgent.agents.common.state import SpectroState
from AstroAgent.agents.common.base_agent import BaseAgent
from AstroAgent.core.runtime.runtime_container import RuntimeContainer

from AstroAgent.agents.multi_agents.utils.usage import safe_to_bool, find_overlap_regions

# from AstroAgent.agents.multi_agents.utils.RA import get_overlap_window

from AstroAgent.agents.multi_agents.utils.VI import (
    _detect_chart_border, _crop_img,
    _remap_to_cropped_canvas, _pixel_tickvalue_fitting,
    _convert_to_spectrum,
    _detect_axis_ticks_tesseract,
    _detect_axis_ticks_paddle,
    group_features_for_llm,
    run_peak_trough_detection,
    run_mask_peaks_and_troughs,
    run_continuum_fitting,
    save_peak_trough_groups,
    save_resolved_features,
    save_cleaned_features,
)

from astropy.io import fits

from AstroAgent.agents.multi_agents.utils.plot import (
    plot_spec_extract,
    plot_spectrum_snr,
    plot_continuum,
    plot_residual_spectrum,
    plot_masked_spectrum,
    plot_cleaned_features,
)


# ---------------------------------------------------------
# 1. Visual Assistant — 负责图像理解与坐标阅读
# ---------------------------------------------------------
class VisualInterpreter(BaseAgent):
    """
    从科学光谱图中自动提取坐标轴刻度、边框、像素映射、峰/谷等信息
    """

    agent_name = "VisualInterpreter"

    def __init__(self, runtime: RuntimeContainer):
        super().__init__(runtime)

    # ======================================================
    # LLM 交互方法
    # ======================================================

    def _load_spectrum_from_fits(self, state: SpectroState) -> SpectroState:
        """
        从 FITS 文件中读取光谱数据。
        适用于 input_format='fits' 的情况。
        """
        fits_path = state['file_path']
        
        with fits.open(fits_path) as hdul:
            # 查找包含数据的 HDU
            data = None
            for i, hdu in enumerate(hdul):
                if hdu.data is not None:
                    data = hdu.data
                    logging.info(f"Found data in HDU {i}")
                    break
            
            if data is None:
                raise ValueError(f"No data found in FITS file: {fits_path}")
            
            # 检查数据类型，确定如何提取列
            if hasattr(data, 'dtype') and hasattr(data.dtype, 'names') and data.dtype.names is not None:
                # 这是 recarray（二进制表）
                col_names = data.dtype.names
                logging.info(f"FITS columns: {col_names}")
                
                # 查找波长和流量列（不区分大小写）
                wavelength_col = None
                flux_col = None
                snr_col = None
                
                for name in col_names:
                    name_upper = name.upper()
                    if name_upper in ['WAVELENGTH', 'WAVE', 'LAMBDA']:
                        wavelength_col = name
                    elif name_upper in ['FLUX', 'F']:
                        flux_col = name
                    elif name_upper in ['SNR', 'SIGNAL_TO_NOISE']:
                        snr_col = name
                
                if wavelength_col is None or flux_col is None:
                    raise ValueError(f"Required columns not found. Available: {col_names}")
                
                wavelength = data[wavelength_col]
                flux = data[flux_col]
                
                if snr_col is not None:
                    effective_snr = data[snr_col]
                else:
                    effective_snr = np.full_like(flux, 5.0, dtype=float)
            else:
                # 普通数组，假设第一列是波长，第二列是流量
                wavelength = data[:, 0]
                flux = data[:, 1]
                effective_snr = np.full_like(flux, 5.0, dtype=float)
            
            # 确保是 numpy 数组
            wavelength = np.array(wavelength)
            flux = np.array(flux)
            effective_snr = np.array(effective_snr)

            snr_medium = np.median(effective_snr)
            
            # 计算统计量（简化版本）
            delta_flux = np.zeros_like(flux)
            std_flux = np.zeros_like(flux)
            
            # 构建 spectrum_dict，与 _convert_to_spectrum 输出格式一致
            spectrum_dict = {
                'flux': flux.tolist(),
                'wavelength': wavelength.tolist(),
                'new_wavelength': wavelength.tolist(),
                'weighted_flux': flux.tolist(),
                'max_unresolved_flux': flux.tolist(),
                'min_unresolved_flux': flux.tolist(),
                'delta_flux': delta_flux.tolist(),
                'std_flux': std_flux.tolist(),
                'effective_snr': effective_snr.tolist(),
                'snr_medium': snr_medium
            }
            
            state['spectrum'] = spectrum_dict
            
        return state

    async def detect_axis_ticks(self, state: SpectroState) -> SpectroState:
        """调用 VLM 检测坐标轴刻度"""
        function_name = "detect_axis_ticks"

        if not state['file_path'] or not os.path.exists(state['file_path']):
            print(state['file_path'])
            logging.error("No image provided or image path does not exist")
            raise

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name
        )

        axis_info = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=state['file_path'],
            parse_json=True,
            description="Axis information",
            want_tools=False
        )
        if axis_info == "非光谱图":
            logging.error("The input image is not a spectral plot. LLM output: %s", axis_info)
            raise
        state["axis_info"] = axis_info
        return state

    async def detect_axis_ticks_OCR(self, state: SpectroState) -> SpectroState:
        """调用 OCR 检测坐标轴刻度"""
        OCR = self.runtime.configs.params.ocr
        print(f"OCR: {OCR}")
        if OCR == 'paddle':
            state['OCR_detected_ticks'] = _detect_axis_ticks_paddle(state)
        else:
            state['OCR_detected_ticks'] = _detect_axis_ticks_tesseract(state)
        print(state["OCR_detected_ticks"])
        return state

    async def combine_axis_mapping(self, state: SpectroState) -> SpectroState:
        """结合视觉结果与 OCR 结果生成像素-数值映射"""
        function_name = "combine_axis_mapping"

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            axis_info=state['axis_info'],
            ocr=state['OCR_detected_ticks']
        )

        tick_pixel_raw = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=None,
            parse_json=True,
            description="Tick pixel mapping",
            want_tools=False
        )
        state["tick_pixel_raw"] = tick_pixel_raw
        return state

    async def revise_axis_mapping(self, state: SpectroState) -> SpectroState:
        """检查并修正刻度值与像素位置匹配关系"""
        function_name = "revise_axis_mapping"

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            axis_mapping=state['tick_pixel_raw']
        )

        tick_pixel_revised = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=None,
            parse_json=True,
            description="Revised tick mapping",
            want_tools=False
        )
        state["tick_pixel_raw"] = tick_pixel_revised
        return state

    async def check_border(self, state: SpectroState):
        """调用 LLM 判断裁剪边界是否干净"""
        function_name = "check_border"

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name
        )

        response = await self.call_llm_with_context(
            system_prompt,
            user_prompt,
            image_path=[state['file_path'], state['crop_path']],
            parse_json=True,
            description='Check cropping',
            want_tools=False
        )
        result = {}
        for key in ['top', 'right', 'bottom', 'left']:
            try:
                val = safe_to_bool(response.get(key, 'false'))
                result[key] = val == 'true'
            except Exception:
                logging.warning(f"Error parsing {key} border, defaulting to False: {response}")
                result[key] = False
        return result

    async def border_detection_and_cropping(self, state: SpectroState) -> SpectroState:
        """迭代调整边距，直到 LLM 确认四边干净"""
        state['margin'] = {
            'top': 20, 'right': 10, 'bottom': 15, 'left': 10,
        }
        MAX_MARGIN = 30
        INCREMENT = 2

        stop = False
        while not stop:
            state["chart_border"] = _detect_chart_border(state['file_path'], state['margin'])
            _crop_img(state['file_path'], state["chart_border"], state['crop_path'])

            box_new = await self.check_border(state)

            if all(box_new.values()):
                stop = True
                break
            elif any(state['margin'][k] >= MAX_MARGIN for k in state['margin']):
                stop = True
                logging.info(f"Reached maximum margin, stopping cropping: {state['margin']}")
                break
            else:
                for k, clean in box_new.items():
                    if not clean:
                        state['margin'][k] += INCREMENT

        return state

    async def select_line_center_with_llm(self, state: SpectroState):
        """使用 LLM 逐组选择最佳谱线中心"""
        function_name = "select_line_center_with_llm"
        num_peaks = self.runtime.configs.params.num_peaks
        num_troughs = self.runtime.configs.params.num_troughs
        sigma_list = sorted(set([0] + self.runtime.configs.params.sigma_list))

        approved_peaks = []
        approved_troughs = []
        resolved_peaks = []
        resolved_troughs = []

        # --- Peaks ---
        for pg in state['peak_groups'][:num_peaks]:
            pg_infos = group_features_for_llm(pg)

            system_prompt, user_prompt = self.runtime.prompt_manager.load(
                state=state,
                agent_name=self.agent_name,
                function_name=function_name,
                sigma_list=sigma_list,
                group_data=pg_infos,
                approved_peaks=approved_peaks,
                approved_troughs=[]
            )
            result = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=None,
                parse_json=True,
                description="Select line center with LLM",
                want_tools=False
            )
            approved_peaks.append(result)

            selected_idx = result.get("selected_index")
            candidate = next(
                (c for c in pg_infos["candidates"] if c["index"] == selected_idx),
                None
            )
            if candidate is not None:
                resolved = candidate.copy()
                resolved["reason"] = result.get("reason", "")
                # 获取 width
                candidate_evidence = candidate.get("evidence", {})
                evidence_global = candidate_evidence.get("global", [])
                evidence_local = candidate_evidence.get("local", [])
                # 优先使用 global，否则使用 local
                widths = [e.get("width", 0) for e in (evidence_global or evidence_local)]
                resolved["width"] = np.max(widths) if widths else 0
                prominence = [e.get("prominence", 0) for e in (evidence_global or evidence_local)]
                resolved["prominence"] = np.max(prominence) if prominence else 0

                resolved_peaks.append(resolved)

        # --- Troughs ---
        for tg in state['trough_groups'][:num_troughs]:
            tg_infos = group_features_for_llm(tg)
            system_prompt, user_prompt = self.runtime.prompt_manager.load(
                state=state,
                agent_name=self.agent_name,
                function_name=function_name,
                sigma_list=sigma_list,
                group_data=tg_infos,
                approved_peaks=[],
                approved_troughs=approved_troughs
            )
            result = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=None,
                parse_json=True,
                description="Select line center with LLM",
                want_tools=False
            )
            approved_troughs.append(result)

            selected_idx = result.get("selected_index")
            candidate = next(
                (c for c in tg_infos["candidates"] if c["index"] == selected_idx),
                None
            )
            if candidate is not None:
                resolved = candidate.copy()
                resolved["reason"] = result.get("reason", "")
                # 获取 width
                candidate_evidence = candidate.get("evidence", {})
                evidence_global = candidate_evidence.get("global", [])
                evidence_local = candidate_evidence.get("local", [])
                # 优先使用 global，否则使用 local
                widths = [e.get("width", 0) for e in (evidence_global or evidence_local)]
                resolved["width"] = np.max(widths) if widths else 0
                prominences = [e.get("prominence", 0) for e in (evidence_global or evidence_local)]
                resolved["prominence"] = np.max(prominences) if prominences else 0
                depth = [e.get("depth", 0) for e in (evidence_global or evidence_local)]
                resolved["depth"] = np.max(depth) if depth else 0
                resolved_troughs.append(resolved)

        state['approved_peaks'] = approved_peaks
        state['approved_troughs'] = approved_troughs
        state['peaks'] = resolved_peaks
        state['troughs'] = resolved_troughs

        save_resolved_features(state, resolved_peaks, resolved_troughs)
        return state

    async def _cleaning(self, state: SpectroState) -> SpectroState:
        """
        清理受 overlap 区域影响的峰值和谷值。
        如果峰值/谷值中心在 overlap_regions 区间内，或者中心在区间外但到区间端点的距离小于宽度，
        则将该特征标记为 artifact，存储到 wiped_peaks/wiped_troughs。
        同时为每个特征添加 describe 属性。
        """
        arm_name = self.runtime.configs.params.arm_name
        arm_wavelength_range = self.runtime.configs.params.arm_wavelength_range
        wavelength = np.array(state['spectrum']['new_wavelength'])

        def _get_describe(width: float, is_peak: bool) -> str:
            """根据宽度返回描述字符串"""
            line_type = "line" if is_peak else "valley"
            if width > 2000:
                return f"Broad {line_type}"
            elif width < 1000:
                return f"Narrow {line_type}"
            else:
                return f"Medium-width {line_type}"

        if not arm_name or not arm_wavelength_range:
            # 没有 arm 配置，不需要清理，但仍需添加 describe
            cleaned_peaks = []
            for p in state['peaks']:
                wl = p['wavelength']
                width = p.get('width', 0)
                if width is not None and wl > wavelength[0]:
                    p['describe'] = _get_describe(width, is_peak=True)
                    cleaned_peaks.append(p)
            cleaned_troughs = []
            for t in state['troughs']:
                wl = t['wavelength']
                width = t.get('width', 0)
                if width is not None and wl > wavelength[0]:
                    t['describe'] = _get_describe(width, is_peak=False)
                    cleaned_troughs.append(t)
            state['cleaned_peaks'] = cleaned_peaks
            state['cleaned_troughs'] = cleaned_troughs
            state['wiped_peaks'] = []
            state['wiped_troughs'] = []
            state['overlap_regions'] = {}
            return state

        # 获取 overlap 区域
        overlap_regions = find_overlap_regions(arm_name, arm_wavelength_range)
        state['overlap_regions'] = overlap_regions

        if not overlap_regions:
            # 没有 overlap 区域，不需要清理，但仍需添加 describe
            cleaned_peaks = []
            for p in state['peaks']:
                wl = p['wavelength']
                width = p.get('width', 0)
                if width is not None and wl > wavelength[0]:
                    p['describe'] = _get_describe(width, is_peak=True)
                    cleaned_peaks.append(p)
            cleaned_troughs = []
            for t in state['troughs']:
                wl = t['wavelength']
                width = t.get('width', 0)
                if width is not None and wl > wavelength[0]:
                    t['describe'] = _get_describe(width, is_peak=False)
                    cleaned_troughs.append(t)
            state['cleaned_peaks'] = cleaned_peaks
            state['cleaned_troughs'] = cleaned_troughs
            state['wiped_peaks'] = []
            state['wiped_troughs'] = []
            return state

        # --- 处理 Peaks ---
        peaks = state['peaks']
        cleaned_peaks = []
        wiped_peaks = []

        for p in peaks:
            wl = p['wavelength']
            width = p.get('width', 0)

            is_artifact = False
            for overlap_name, (left, right) in overlap_regions.items():
                # 检查峰值中心是否在 overlap 区间内
                if left <= wl <= right:
                    is_artifact = True
                    break
                # 检查峰值中心到区间端点的距离是否小于峰的宽度
                distance_to_left = abs(wl - left)
                distance_to_right = abs(wl - right)
                if distance_to_left <= width or distance_to_right <= width:
                    is_artifact = True
                    break

            if not is_artifact:
                if width is not None and wl > wavelength[0]:
                    p['describe'] = _get_describe(width, is_peak=True)
                    cleaned_peaks.append(p)
            else:
                wiped_peaks.append(p)

        state['cleaned_peaks'] = cleaned_peaks
        state['wiped_peaks'] = wiped_peaks

        # --- 处理 Troughs ---
        troughs = state['troughs']
        cleaned_troughs = []
        wiped_troughs = []

        for t in troughs:
            wl = t['wavelength']
            width = t.get('width', 0)

            is_artifact = False
            for overlap_name, (left, right) in overlap_regions.items():
                # 检查谷值中心是否在 overlap 区间内
                if left <= wl <= right:
                    is_artifact = True
                    break
                # 检查谷值中心到区间端点的距离是否小于谷的宽度
                distance_to_left = abs(wl - left)
                distance_to_right = abs(wl - right)
                if distance_to_left <= width or distance_to_right <= width:
                    is_artifact = True
                    break

            if not is_artifact:
                if width is not None and wl > wavelength[0]:
                    t['describe'] = _get_describe(width, is_peak=False)
                    cleaned_troughs.append(t)
            else:
                wiped_troughs.append(t)

        state['cleaned_troughs'] = cleaned_troughs
        state['wiped_troughs'] = wiped_troughs

        save_cleaned_features(state, cleaned_peaks, cleaned_troughs)

        logging.info(f"Cleaning: {len(wiped_peaks)} peaks wiped, {len(cleaned_peaks)} peaks kept; "
                     f"{len(wiped_troughs)} troughs wiped, {len(cleaned_troughs)} troughs kept")
        return state

    # ======================================================
    # 主流程
    # ======================================================

    async def run(self, state: SpectroState, plot: bool = True):
        """执行完整视觉分析流程"""
        params = self.runtime.configs.params
        try:
            if self.runtime.configs.io.input_format == 'fits':
                self._load_spectrum_from_fits(state)
            else:  
                # === Phase A: 坐标轴检测与校准 ===
                await self.detect_axis_ticks(state)
                await self.detect_axis_ticks_OCR(state)
                await self.combine_axis_mapping(state)
                await self.revise_axis_mapping(state)

                # === Phase B: 图像裁剪与像素映射 ===
                await self.border_detection_and_cropping(state)
                state["tick_pixel_remap"] = _remap_to_cropped_canvas(
                    state['tick_pixel_raw'], state["chart_border"]
                )
                state["pixel_to_value"] = _pixel_tickvalue_fitting(state['tick_pixel_remap'])

                # === Phase C: 光谱重建 ===
                input_format = getattr(self.runtime.configs.params, 'input_format', 'image')
                if input_format == 'fits':
                    # FITS 格式：直接从文件读取光谱数据
                    self._load_spectrum_from_fits(state)
                else:
                    # 图像格式：从图像中提取光谱
                    state["spectrum"] = _convert_to_spectrum(
                        state['crop_path'], state['pixel_to_value']
                    )
            plot_spec_extract(state)
            plot_spectrum_snr(state)

            # === Phase D: 初次峰谷检测 & Continuum 拟合 ===
            spec = state["spectrum"]
            wavelengths_orig = np.array(spec["new_wavelength"])
            flux_orig = np.array(spec["weighted_flux"])

            run_peak_trough_detection(
                state, wavelengths_orig, flux_orig,
                sigma_list=params.sigma_list,
                tol_wavelength=params.tol_wavelength,
                prom_peaks=params.prom_threshold_peaks,
                prom_troughs=params.prom_threshold_troughs,
            )
            if state.get('peak_groups') is None or state.get('trough_groups') is None:
                raise RuntimeError("Failed to detect peaks and troughs")
            print(f"Detected {len(state['peak_groups'])} peak groups and {len(state['trough_groups'])} trough groups.")

            run_mask_peaks_and_troughs(
                state,
                num_peaks=params.num_peaks,
                num_troughs=params.num_troughs,
            )
            plot_masked_spectrum(state)

            run_continuum_fitting(
                state,
                arm_name=params.arm_name,
                arm_wavelength_range=params.arm_wavelength_range,
                sigma_continuum=params.continuum_smoothing,
            )
            plot_continuum(state)
            plot_residual_spectrum(state)

            # === Phase E: 残差光谱上的精细峰谷检测 ===
            residual = state["residual_spectrum"]
            wavelengths_res = np.array(residual["wavelength"])
            flux_res = np.array(residual["flux"])

            run_peak_trough_detection(
                state, wavelengths_res, flux_res,
                sigma_list=params.sigma_list,
                tol_wavelength=params.tol_wavelength,
                prom_peaks=params.prom_threshold_peaks,
                prom_troughs=params.prom_threshold_troughs,
            )
            print(f"Detected {len(state['peak_groups'])} peak groups and {len(state['trough_groups'])} trough groups.")
            save_peak_trough_groups(state)

            # === Phase F: LLM 选择谱线中心 ===
            await self.select_line_center_with_llm(state)

            # === Phase G: 清理受 overlap 区域影响的峰值 ===
            await self._cleaning(state)

            plot_cleaned_features(state, params.sigma_list, params.label)

            return state
        except Exception as e:
            print(f"run pipeline terminated with error: {e}")
            raise
