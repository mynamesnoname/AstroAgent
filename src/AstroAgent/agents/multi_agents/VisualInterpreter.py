import json
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.ndimage import gaussian_filter1d

from AstroAgent.agents.common.state import SpectroState
from AstroAgent.agents.common.base_agent import BaseAgent
from AstroAgent.core.runtime.runtime_container import RuntimeContainer

from AstroAgent.agents.common.utils import (
    _detect_chart_border, _crop_img, 
    _remap_to_cropped_canvas, _pixel_tickvalue_fitting, 
    _process_and_extract_curve_points, _convert_to_spectrum,
    _find_features_multiscale, _plot_spectrum, 
    merge_features, safe_to_bool, 
    find_overlap_regions, 
    _detect_axis_ticks_tesseract,
    _detect_axis_ticks_paddle
)

# ---------------------------------------------------------
# 1. Visual Assistant — 负责图像理解与坐标阅读
# ---------------------------------------------------------
class VisualInterpreter(BaseAgent):
    """
    从科学光谱图中自动提取坐标轴刻度、边框、像素映射、峰/谷等信息
    Visual Interpreter: Automatically extract coordinate axis ticks, border, pixel mapping, peaks=valleys etc. information from scientific spectroscopic graphs
    """

    agent_name = "VisualInterpreter"

    def __init__(self, runtime: RuntimeContainer):
        super().__init__(runtime)

    # --------------------------
    # Step 1.1: Detect Axis Ticks
    # --------------------------
    async def detect_axis_ticks(self, state: SpectroState) -> SpectroState:
        """
        调用 VLM 检测坐标轴刻度，如果无图像或非光谱图报错
        Call VLM to detect axis ticks, if no image or not a spectral graph, raise error
        """
        function_name = "detect_axis_ticks"

        if not state['image_path'] or not os.path.exists(state['image_path']):
            print(state['image_path'])
            logging.error("❌ No image provided or image path does not exist")
            raise

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name
        )

        axis_info = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=state['image_path'],
            parse_json=True,
            description="Axis information",
            want_tools=False
        )
        if axis_info == "非光谱图":
            logging.error("❌ The input image is not a spectral plot. LLM output: %s", axis_info)
            raise
        state["axis_info"] = axis_info
        # print('detect_axis_ticks:')
        # print(axis_info)
        return state

    # --------------------------
    # Step 1.2: OCR Detect Axis Ticks
    # --------------------------
    async def detect_axis_ticks_OCR(self, state: SpectroState) -> SpectroState:
        """
        调用 OCR 检测坐标轴刻度，如果无图像或非光谱图报错
        Call OCR to detect axis ticks, if no image or not a spectral graph, raise error
        """
        OCR = self.runtime.configs.params.ocr
        print(f"OCR: {OCR}")
        if OCR == 'paddle':
            state['OCR_detected_ticks'] = _detect_axis_ticks_paddle(state)
        else:
            state['OCR_detected_ticks'] = _detect_axis_ticks_tesseract(state)
        print(state["OCR_detected_ticks"])
        return state
    
    # --------------------------
    # Step 1.3: Combine Visual + OCR Axis Mapping
    # --------------------------
    async def combine_axis_mapping(self, state: SpectroState) -> SpectroState:
        """
        结合视觉结果与 OCR 结果生成像素-数值映射
        Combine visual results and OCR results to generate pixel-value mapping
        """
        function_name = "combine_axis_mapping"

        axis_info = state['axis_info']
        ocr = state['OCR_detected_ticks']

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            axis_info=axis_info, 
            ocr=ocr
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
        # print('combine_axis_mapping:')
        # print(tick_pixel_raw)
        return state

    # --------------------------
    # Step 1.4: Revision and Correction
    # --------------------------
    async def revise_axis_mapping(self, state: SpectroState) -> SpectroState:
        """
        检查并修正刻度值与像素位置匹配关系
        Check and revise the relationship between tick values and pixel positions
        """
        function_name = "revise_axis_mapping"

        axis_mapping = state['tick_pixel_raw']

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            axis_mapping=axis_mapping
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
        # print('revise_axis_mapping:')
        # print(tick_pixel_revised)
        return state

    # --------------------------
    # Step 1.5: Image Cropping
    # --------------------------
    async def check_border(self, state: SpectroState):
        function_name = "check_border"

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name
        )

        response = await self.call_llm_with_context(
            system_prompt,
            user_prompt,
            image_path=[state['image_path'], state['crop_path']],
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
        state['margin'] = {
            'top': 20,
            'right': 10,
            'bottom': 15,
            'left': 10,
        }
        MAX_MARGIN = 30  # Maximum margin to prevent infinite growth
        INCREMENT = 2    # Increment per round

        stop = False
        while not stop:
            # Use OpenCV to predict edge positions
            state["chart_border"] = _detect_chart_border(state['image_path'], state['margin'])
            _crop_img(state['image_path'], state["chart_border"], state['crop_path'])

            # Call LLM to determine if all four edges are clean
            box_new = await self.check_border(state)

            # Stop conditions
            if all(box_new.values()):  # All edges are clean
                stop = True
                break
            elif any(state['margin'][k] >= MAX_MARGIN for k in state['margin']):
                stop = True
                logging.info(f"Reached maximum margin, stopping cropping: {state['margin']}")
                break
            else:
                # Only increase the margin for the edges that are not clean
                for k, clean in box_new.items():
                    if not clean:
                        state['margin'][k] += INCREMENT

        return state

    async def peak_trough_detection(self, state: SpectroState):
        try:
            sigma_list = self.runtime.configs.params.sigma_list
            tol_pixels = self.runtime.configs.params.tol_pixels
            prom_peaks = self.runtime.configs.params.prom_threshold_peaks
            prom_troughs = self.runtime.configs.params.prom_threshold_troughs

            spec = state["spectrum"]
            wavelengths = np.array(spec["new_wavelength"])
            flux = np.array(spec["weighted_flux"])

            state["peaks"] = _find_features_multiscale(
                wavelengths, flux,
                state, feature="peak", sigma_list=sigma_list,
                prom=prom_peaks, tol_pixels=tol_pixels,
                use_continuum_for_trough=True
            )
            state["troughs"] = _find_features_multiscale(
                wavelengths, flux,
                state, feature="trough", sigma_list=sigma_list,
                prom=prom_troughs, tol_pixels=tol_pixels, 
                use_continuum_for_trough=True,
                min_depth=0.08
            )
            # print(len(state["peaks"]), len(state["troughs"]))

            # Divide wavelengths into ROIs of 500 Angstroms each for peak/valley detection
            ROI_peaks = []
            ROI_troughs = []
            roi_size = 500  # Width of each ROI, in Angstroms
            roi_edges = np.arange(wavelengths[0], wavelengths[-1], roi_size)
            for i in range(len(roi_edges)-1):
                roi_start = roi_edges[i]
                roi_end = roi_edges[i+1]
                mask = (wavelengths >= roi_start) & (wavelengths < roi_end)
                roi_wavelengths = np.where(mask, wavelengths, 0)
                roi_flux = np.where(mask, flux, 0)
                # If roi_wavelengths length is non-zero
                if len(roi_wavelengths) == 0:
                    continue
                roi_peaks = _find_features_multiscale(
                    roi_wavelengths, roi_flux,
                    state, feature="peak", sigma_list=sigma_list,
                    prom=prom_peaks, tol_pixels=tol_pixels,
                    use_continuum_for_trough=True
                )
                roi_troughs = _find_features_multiscale(
                    roi_wavelengths, roi_flux,
                    state, feature="trough", sigma_list=sigma_list,
                    prom=prom_troughs, tol_pixels=tol_pixels, 
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
                # If roi_wavelengths length is non-zero
                roi_wavelengths = np.where(mask, wavelengths, 0)
                roi_flux = np.where(mask, flux, 0)
                if len(roi_wavelengths) == 0:
                    continue
                roi_peaks = _find_features_multiscale(
                    roi_wavelengths, roi_flux,
                    state, feature="peak", sigma_list=sigma_list,
                    prom=prom_peaks, tol_pixels=tol_pixels,
                    use_continuum_for_trough=True
                )
                roi_troughs = _find_features_multiscale(
                    roi_wavelengths, roi_flux,
                    state, feature="trough", sigma_list=sigma_list,
                    prom=prom_troughs, tol_pixels=tol_pixels,
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
        """
        简单的continuum拟合
        Simple continuum fitting
        """
        try:
            spec = state["spectrum"]
            wavelengths = np.array(spec["new_wavelength"])
            flux = np.array(spec["weighted_flux"])

            arm_name = self.runtime.configs.params.arm_name
            arm_wavelength_range = self.runtime.configs.params.arm_wavelength_range
            # print('cut continuum')
            if arm_name:
                overlap_regions = find_overlap_regions(arm_name, arm_wavelength_range)
                # Initialize mask as all False
                mask = np.zeros_like(wavelengths, dtype=bool)
                for key in overlap_regions:
                    low, high = overlap_regions[key]
                    region_mask = (wavelengths >= low) & (wavelengths <= high)
                    mask = mask | region_mask
                wavelengths = wavelengths[~mask]
                flux = flux[~mask]

            sigma_contunuum = self.runtime.configs.params.continuum_smoothing
            print(f'CONTINUUM_SMOOTHING_SIGMA: {sigma_contunuum}')

            if not sigma_contunuum:
                logging.error("CONTINUUM_SMOOTHING_SIGMA is not set, using 100")
                sigma_contunuum = 100

            continuum_flux = gaussian_filter1d(flux, sigma=sigma_contunuum)
            state['continuum'] = {
                'wavelength': wavelengths.tolist(),
                'flux': continuum_flux.tolist()
            }
        except Exception as e:
            print(f"❌ continuum_fitting: {e}")
        return state

    # --------------------------
    # Step 1.1~1.12: Main Workflow
    # --------------------------
    async def run(self, state: SpectroState, plot: bool = True):
        """执行完整视觉分析流程"""
        try:
            # Step 1.1: Visual LLM Extract Axis Ticks
            await self.detect_axis_ticks(state)
            # Step 1.2: OCR Detect Axis Ticks
            await self.detect_axis_ticks_OCR(state)
            # Step 1.3: Combine Visual + OCR Axis Mapping
            await self.combine_axis_mapping(state)
            # Step 1.4: Revision and Correction
            await self.revise_axis_mapping(state)
            # Step 1.5: Border Detection and Cropping
            await self.border_detection_and_cropping(state)
            # Step 1.6: Remap Pixels
            state["tick_pixel_remap"] = _remap_to_cropped_canvas(state['tick_pixel_raw'], state["chart_border"])
            # Step 1.7: Pixel to Value Fitting
            state["pixel_to_value"] = _pixel_tickvalue_fitting(state['tick_pixel_remap'])
            # Step 1.8: Extract Curve & Grayscale
            curve_points, curve_gray_values = _process_and_extract_curve_points(state['crop_path'])
            state["curve_points"] = curve_points
            state["curve_gray_values"] = curve_gray_values
            # Step 1.9: Spectrum Reconstruction
            state["spectrum"] = _convert_to_spectrum(state['curve_points'], state['curve_gray_values'], state['pixel_to_value'])
            # Step 1.10: Detect Peaks and Troughs
            await self.peak_trough_detection(state)
            max_attempts = 100
            attempts = 0
            while (state['merged_peaks'] is None or state['merged_troughs'] is None) and attempts < max_attempts:
                try:
                    print(f"Retry peak/trough detection, attempt {attempts + 1}")
                    await self.peak_trough_detection(state)
                    attempts += 1
                except Exception as e:
                    print(f"Peak/trough detection failed: {e}")
                    break
            if state['merged_peaks'] is None or state['merged_troughs'] is None:
                raise RuntimeError("Failed to detect peaks and troughs after maximum attempts")
            print(f"Detected {len(state['merged_peaks'])} peaks and {len(state['merged_troughs'])} troughs.")
            # Step 1.11: Continuum Fitting
            await self.continuum_fitting(state)
            # Step 1.12: Optional Plotting
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
