# import json
import os
import numpy as np
# import pandas as pd
import logging

from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

from AstroAgent.agents.common.state import SpectroState
from AstroAgent.agents.common.base_agent import BaseAgent
from AstroAgent.agents.common.result_writer import ResultWriter

from AstroAgent.core.runtime.runtime_container import RuntimeContainer

from AstroAgent.agents.multi_agents.utils.usage import find_overlap_regions
from AstroAgent.agents.multi_agents.utils.usage import safe_to_bool
from AstroAgent.agents.multi_agents.utils.VI import (
    _detect_chart_border, _crop_img,
    _remap_to_cropped_canvas, _pixel_tickvalue_fitting,
    _convert_to_spectrum,
    _detect_axis_ticks_tesseract,
    _detect_axis_ticks_paddle,
    run_continuum_fitting_masked,
    run_iterative_feature_detection,
    brute_force_line_matching
)
from AstroAgent.agents.multi_agents.utils.plot import (
    plot_spec_extract,
    plot_spectrum_snr,
    plot_continuum,
    plot_residual_spectrum,
    plot_features,
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
        支持两种格式：
        1. DESI 分波段格式（B/R/Z_WAVELENGTH, B/R/Z_FLUX 等 HDU）
        2. 单表格格式（包含 WAVELENGTH/FLUX 列的 BinTableHDU）
        """
        fits_path = state['file_path']
        
        with fits.open(fits_path) as hdul:
            # 获取所有 HDU 名称
            hdu_names = [hdu.name.upper() for hdu in hdul]
            logging.info(f"FITS HDU names: {hdu_names}")
            
            # 检测是否为 DESI 分波段格式
            is_desi_format = all(
                f'{band}_WAVELENGTH' in hdu_names and f'{band}_FLUX' in hdu_names
                for band in ['B', 'R', 'Z']
            )
            
            if is_desi_format:
                # === DESI 分波段格式 ===
                logging.info("Detected DESI multi-arm FITS format")
                
                # 读取元数据（如果存在）
                if 'METADATA' in hdu_names:
                    metadata_idx = hdu_names.index('METADATA')
                    metadata = hdul[metadata_idx].data[0]
                    logging.info(f"METADATA: TARGETID={metadata['TARGETID']}, "
                                f"VI_SPECTYPE={metadata['VI_SPECTYPE']}, VI_Z={metadata['VI_Z']}")
                
                # 读取三个波段的数据并记录波长范围
                band_names = ['B', 'R', 'Z']
                wavelength_list = []
                flux_list = []
                ivar_list = []
                mask_list = []  # SPECMASK 数据
                band_wavelengths = []  # 各波段的波长范围
                
                for band in band_names:
                    wave_hdu = hdul[f'{band}_WAVELENGTH']
                    flux_hdu = hdul[f'{band}_FLUX']
                    ivar_hdu = hdul[f'{band}_IVAR']
                    
                    wave_data = wave_hdu.data
                    wavelength_list.append(wave_data)
                    flux_list.append(flux_hdu.data)
                    ivar_list.append(ivar_hdu.data)
                    
                    # 读取 SPECMASK（如果存在）
                    mask_hdu_name = f'{band}_MASK'
                    if mask_hdu_name in hdu_names:
                        mask_data = hdul[mask_hdu_name].data
                        mask_list.append(mask_data)
                        # 统计该波段的 mask 情况
                        unique_masks = np.unique(mask_data)
                        if len(unique_masks) > 1 or unique_masks[0] != 0:
                            logging.info(f"  {band}_MASK has non-zero values: {unique_masks}")
                    else:
                        # 如果没有 mask，假设所有像素都是好的
                        mask_list.append(np.zeros(len(wave_data), dtype=np.uint32))
                        logging.info(f"  {band}_MASK not found, assuming all pixels are good")
                    
                    # 记录该波段的波长范围
                    band_wavelengths.append([float(wave_data.min()), float(wave_data.max())])
                
                
                # 合并数组（保留完整的原始数据）
                wavelength = np.concatenate(wavelength_list)
                flux = np.concatenate(flux_list)
                ivar = np.concatenate(ivar_list)
                specmask = np.concatenate(mask_list)  # 合并 mask
                
                logging.info(f"Combined {len(wavelength)} wavelength points from 3 arms")
                logging.info(f"Wavelength range: {wavelength.min():.2f} - {wavelength.max():.2f} Å")
                
                # 统计整体 mask 情况
                n_masked = np.sum(specmask != 0)
                if n_masked > 0:
                    logging.info(f"SPECMASK: {n_masked} pixels ({100*n_masked/len(specmask):.2f}%) have quality issues")
                    # 统计各种 mask 值的出现次数
                    unique_vals, counts = np.unique(specmask[specmask != 0], return_counts=True)
                    for val, cnt in zip(unique_vals, counts):
                        bits_set = [i for i in range(32) if val & (1 << i)]
                        logging.info(f"  Mask value {val} (bits {bits_set}): {cnt} pixels")
                else:
                    logging.info("SPECMASK: All pixels are clean (mask=0)")
                
                # === 构建 quality mask ===
                # quality_mask[i] = True 表示该像素质量良好（specmask == 0）
                quality_mask = (specmask == 0)
                
                # === 引入 arm overlap mask ===                
                overlap_regions = find_overlap_regions(band_names, band_wavelengths)
                
                # 构建 combined mask：True 表示保留
                # 1. 不在重叠区域内
                # 2. SPECMASK == 0（质量良好）
                keep_mask = quality_mask.copy()
                
                if overlap_regions:
                    logging.info(f"Detected {len(overlap_regions)} overlap regions: {overlap_regions}")
                    
                    for region_name, (ov_start, ov_end) in overlap_regions.items():
                        in_overlap = (wavelength >= ov_start) & (wavelength <= ov_end)
                        keep_mask &= ~in_overlap
                        logging.info(f"  Masking overlap '{region_name}': [{ov_start:.2f}, {ov_end:.2f}], "
                                    f"{in_overlap.sum()} points removed")
                
                # 应用 mask，只保留质量良好且非重叠区域的像素
                # 注意：wavelength 和 flux 保留完整数据（用于 spectrum_dict['flux'] 和 ['wavelength']）
                # new_wavelength、weighted_flux、ivar 只保留 mask 后的数据
                wavelength_masked = wavelength[keep_mask]
                flux_masked = flux[keep_mask]
                ivar_masked = ivar[keep_mask]
                
                n_removed = len(wavelength) - len(wavelength_masked)
                n_quality_removed = np.sum(~quality_mask)
                logging.info(f"After masking: {len(wavelength_masked)} points remain "
                            f"({n_removed} removed, including {n_quality_removed} from SPECMASK)")
                
                # DESI 格式只有 IVAR，不直接计算 SNR
                # effective_snr 设为 None，下游模块可从 ivar 计算
                effective_snr = None
                has_ivar = True
                
            else:
                # === 单表格格式 ===
                logging.info("Detected single-table FITS format")
                is_desi_format = False
                overlap_regions = None
                
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
                    ivar_col = None
                    
                    for name in col_names:
                        name_upper = name.upper()
                        if name_upper in ['WAVELENGTH', 'WAVE', 'LAMBDA']:
                            wavelength_col = name
                        elif name_upper in ['FLUX', 'F']:
                            flux_col = name
                        elif name_upper in ['SNR', 'SIGNAL_TO_NOISE']:
                            snr_col = name
                        elif name_upper in ['IVAR', 'INVERSE_VARIANCE']:
                            ivar_col = name
                    
                    if wavelength_col is None or flux_col is None:
                        raise ValueError(f"Required columns not found. Available: {col_names}")
                    
                    wavelength = np.array(data[wavelength_col])
                    flux = np.array(data[flux_col])
                    
                    if snr_col is not None:
                        effective_snr = np.array(data[snr_col])
                        ivar = None
                        has_ivar = False
                    elif ivar_col is not None:
                        ivar = np.array(data[ivar_col])
                        effective_snr = None  # 只有 IVAR 时，SNR 设为 None
                        has_ivar = True
                    else:
                        effective_snr = np.full_like(flux, 5.0, dtype=float)
                        ivar = None
                        has_ivar = False
                else:
                    # 普通数组，假设第一列是波长，第二列是流量
                    wavelength = data[:, 0]
                    flux = data[:, 1]
                    effective_snr = np.full_like(flux, 5.0, dtype=float)
                    ivar = None
                    has_ivar = False
            
            # 确保是 numpy 数组
            wavelength = np.array(wavelength, dtype=np.float64)
            flux = np.array(flux, dtype=np.float64)
            
            # 处理 ivar
            if has_ivar and ivar is not None:
                ivar = np.array(ivar, dtype=np.float64)
            
            # 对于 DESI 格式，使用 mask 后的数据作为 new_wavelength/weighted_flux/ivar
            # 对于其他格式，mask 后数据与原始数据相同
            # 注意：保持 numpy 数组格式用于后续计算
            # DESI 格式：应用 SPECMASK 和 overlap mask 后的数据
            if is_desi_format:
                new_wavelength = np.array(wavelength_masked, dtype=np.float64)
                weighted_flux = np.array(flux_masked, dtype=np.float64)
                ivar_final = np.array(ivar_masked, dtype=np.float64)
            else:
                new_wavelength = wavelength
                weighted_flux = flux
                ivar_final = ivar if has_ivar and ivar is not None else None

            # 处理 effective_snr
            if effective_snr is not None:
                effective_snr = np.array(effective_snr, dtype=np.float64)
                snr_medium = float(np.median(effective_snr))
            elif ivar_final is not None:
                # 从 IVAR 计算 SNR: SNR = FLUX * sqrt(IVAR)
                ivar_safe = np.maximum(ivar_final, 0)  # 确保 IVAR 非负
                effective_snr = weighted_flux * np.sqrt(ivar_safe)
                snr_medium = float(np.median(effective_snr))
            else:
                # 没有数据时使用默认值
                effective_snr = np.full_like(weighted_flux, 5.0, dtype=np.float64)
                snr_medium = 5.0

            
            # smooth the spectrum
            weighted_flux = gaussian_filter1d(weighted_flux, sigma=2)
            
            # 构建 spectrum_dict，与 _convert_to_spectrum 输出格式一致
            # 最后统一转换为 list
            spectrum_dict = {
                'flux': flux.tolist(),
                'wavelength': wavelength.tolist(),
                'new_wavelength': new_wavelength.tolist(),
                'weighted_flux': weighted_flux.tolist(),
                'max_unresolved_flux': None,
                'min_unresolved_flux': None,
                'delta_flux': None,
                'std_flux': None,
                'effective_snr': effective_snr.tolist(),
                'snr_medium': snr_medium,
                'ivar': ivar_final.tolist() if ivar_final is not None else None,
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
                    arm_name = self.runtime.configs.params.arm_name
                    arm_wavelength_range = self.runtime.configs.params.arm_wavelength_range
                    state["spectrum"] = _convert_to_spectrum(
                        state['crop_path'], state['pixel_to_value'], arm_name, arm_wavelength_range
                    )

            plot_spec_extract(state)
            plot_spectrum_snr(state)
            
            # === Phase D: 迭代特征检测（在 continuum fitting 之前）===
            spec = state["spectrum"]
                        
            # 优先使用 ivar，其次使用 effective_snr
            ivar_data = spec.get("ivar", None)
            effective_snr_data = spec.get("effective_snr", 7.0) if ivar_data is None else None
                        
            # ── 吸收线检测参数（覆盖 find_absorption_lines 默认值）──
            absorption_detection_params = {
                # 窗口设置
                'window_width': params.abs_window_width,
                'window_overlap': params.abs_window_overlap,
                # 显著性阈值
                'delta_chi2_base': params.abs_delta_chi2_base,
                'dynamic_threshold_factor': params.abs_dynamic_threshold_factor,
                'global_delta_chi2_threshold': params.abs_global_delta_chi2_threshold,
                # 谱线宽度限制
                'fwhm_min': params.abs_fwhm_min,
                'fwhm_max': params.abs_fwhm_max,
                # 信噪比深度阈值
                'snr_depth_threshold': params.abs_snr_depth_threshold,
                # Smoothed 候选谷参数（宽低振幅谷的先验）
                'smooth_sigma_trough': params.smooth_sigma_trough,
                'smooth_prominence_frac_trough': params.smooth_prominence_frac_trough,
            }
                        
            # ── 发射线检测参数（覆盖 find_emission_lines 默认值）──
            emission_detection_params = {
                # 窗口设置
                'window_width': params.em_window_width,
                'window_overlap': params.em_window_overlap,
                # 显著性阈值
                'delta_chi2_base': params.em_delta_chi2_base,
                'dynamic_threshold_factor': params.em_dynamic_threshold_factor,
                'global_delta_chi2_threshold': params.em_global_delta_chi2_threshold,
                # 系统误差分量（参考 DLA-Toolkit var_lss 思路）
                'sys_err_frac': params.em_sys_err_frac,
                # 谱线宽度限制
                'fwhm_min': params.em_fwhm_min,
                'fwhm_max': params.em_fwhm_max,
                # Smoothed 候选峰参数（宽低振幅峰的先验）
                'smooth_sigma': params.smooth_sigma,
                'smooth_prominence_frac': params.smooth_prominence_frac,
            }
                        
            feature_result = run_iterative_feature_detection(
                output_dir=state['output_dir'],
                file_name=state['file_name'],
                wavelength=spec["new_wavelength"],
                flux=spec["weighted_flux"],
                ivar=ivar_data,
                effective_snr=effective_snr_data,
                n_iterations=params.n_iterations if hasattr(params, 'n_iterations') else 3,
                absorption_detection_params=absorption_detection_params,
                emission_detection_params=emission_detection_params,
                verbose=True,
            )
                        
            # 保存结果到 state（发射线 → peaks，吸收线 → troughs）
            # 注意：返回値已按波长从小到大排序，包含 amplitude_rank 字段
            df_em = feature_result['df_emission']
            df_ab = feature_result['df_absorption']
                        
            state['peaks'] = df_em.to_dict('records') if len(df_em) > 0 else []
            state['troughs'] = df_ab.to_dict('records') if len(df_ab) > 0 else []
                        
            # records 同样已按波长排序
            state['absorption_records'] = feature_result['records_absorption']
            state['emission_records'] = feature_result['records_emission']
            
            # === Phase E: Continuum 拟合（将已检测峰/谷区域 mask掉后再拟合）===
            run_continuum_fitting_masked(
                state,
                peaks=state['emission_records'],
                troughs=state['absorption_records'],
                chebyshev_degree=params.chebyshev_degree if hasattr(params, 'chebyshev_degree') else None,
                chebyshev_min_degree=params.chebyshev_min_degree if hasattr(params, 'chebyshev_min_degree') else 1,
                chebyshev_max_degree=params.chebyshev_max_degree if hasattr(params, 'chebyshev_max_degree') else 10,
            )
            plot_continuum(state)
            plot_residual_spectrum(state)
            
            plot_features(state)

            tol_wavelength_qso_elg = self.runtime.configs.params.tol_wavelength_qso_elg
            brute_force_matching_qso_elg = brute_force_line_matching(
                state, tol_wavelength_qso_elg,
                min_qso_redshift=self.runtime.configs.params.min_qso_redshift,
                min_galaxy_redshift=self.runtime.configs.params.min_galaxy_redshift,
                mode='qso_elg'
            )
            state['brute_force_matching_qso_elg'] = brute_force_matching_qso_elg

            tol_wavelength_lrg_bgs = self.runtime.configs.params.tol_wavelength_lrg_bgs
            brute_force_matching_lrg_bgs = brute_force_line_matching(
                state, tol_wavelength_lrg_bgs,
                min_qso_redshift=self.runtime.configs.params.min_qso_redshift,
                min_galaxy_redshift=self.runtime.configs.params.min_galaxy_redshift,
                mode='lrg_bgs'
            )
            state['brute_force_matching_lrg_bgs'] = brute_force_matching_lrg_bgs

            ResultWriter().write_brute_force_matching(state)

            return state
        except Exception as e:
            print(f"run pipeline terminated with error: {e}")
            raise
