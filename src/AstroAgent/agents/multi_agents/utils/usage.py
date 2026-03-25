import cv2
import os
import pytesseract
from paddleocr import PaddleOCR
import json
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from typing import Any, List, Dict, Tuple, Optional, Union
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_widths
from scipy.stats import mode

from AstroAgent.agents.common.state import SpectroState

def safe_to_bool(value):
    """专门处理true/True相关值的转换"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ['true', '1', 't', 'yes', 'y']
    return bool(value)

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

# def _load_feature_params():
#     """安全读取峰值/谷值检测参数"""
#     sigma_list = parse_list(os.getenv("SIGMA_LIST"), [2, 4, 16])
#     tol_pixels = getenv_int("TOL_PIXELS", 10)
#     prom_peaks = getenv_float("PROM_THRESHOLD_PEAKS", 0.01)
#     prom_troughs = getenv_float("PROM_THRESHOLD_TROUGHS", 0.05)
#     # weight_original = getenv_float("WEIGHT_ORIGINAL", 0.5)
#     plot_peaks = getenv_int("PLOT_PEAKS_NUMBER", 10)
#     plot_troughs = getenv_int("PLOT_TROUGHS_NUMBER", 15)

#     return sigma_list, tol_pixels, prom_peaks, prom_troughs, plot_peaks, plot_troughs

# ===========================================================
# Wavelength Band Overlap
# ===========================================================

def find_overlap_regions(band_names, band_wavelengths):
    """
    找出所有波段之间的重叠区域

    参数:
    band_names: 波段名称列表
    band_wavelengths: 波段波长范围列表，每个元素为[start, end]

    返回:
    重叠区域的字典
    """
    result = {}
    n = len(band_names)

    # 遍历所有可能的两两组合
    for i in range(n):
        for j in range(i + 1, n):
            # 获取两个波段的波长范围
            start1, end1 = band_wavelengths[i]
            start2, end2 = band_wavelengths[j]

            # 计算重叠区域
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)

            # 如果存在重叠区域
            if overlap_start < overlap_end:
                overlap_name = f"{band_names[i]}-{band_names[j]}"
                result[overlap_name] = [overlap_start, overlap_end]

    return result

def get_wiped_lines(state, overlap_regions):
    wiped_peaks = state.get('wiped_peaks', [])
    if wiped_peaks:
        width_means = [wp.get('width_mean') for wp in wiped_peaks[:5] if wp.get('width_mean') is not None]
        if width_means:
            wws = np.max(width_means)
        else:
            wws = 0
    else:
        wws = 0

    for key in overlap_regions:
        range = overlap_regions[key]
        overlap_regions[key] = [range[0]-wws, range[1]+wws]

    wiped = [
        {
            "wavelength": wp.get('wavelength'),
            "flux": wp.get('mean_flux'),
            "width": wp.get('width_mean'),
        }
        for wp in state.get('wiped_peaks', [])[:5]
    ]
    return wiped


###########################################################################3

# def _detect_features_on_flux(
#     feature, flux, x_axis_slope, sigma, prominence=None, height=None,
#     wavelengths=None, continuum=None
# ):
#     """
#     平滑后检测峰/谷，返回平滑光谱和峰信息。
#     支持 trough 检测（通过 flux 取负或 continuum 归一化）。
#     """
#     # === Step 1: 数据准备 ===
#     if feature == "trough":
#         if continuum is not None:
#             flux_proc = flux / continuum
#             flux_proc = 1.0 - flux_proc  # 变成“吸收强度”，越大越深
#         else:
#             flux_proc = -flux.copy()
#     else:
#         flux_proc = flux - continuum if continuum is not None else flux.copy()

#     # === Step 2: 平滑 ===
#     flux_smooth = gaussian_filter1d(flux_proc, sigma=sigma) if sigma > 0 else flux_proc

#     # === Step 3: 峰检测 ===
#     peaks, props = find_peaks(flux_smooth, height=height, prominence=prominence)
#     widths_res = peak_widths(flux_smooth, peaks, rel_height=0.5)

#     peaks_info = []
#     for i, p in enumerate(peaks):
#         width_pix = widths_res[0][i]
#         info = {
#             "index": int(p),
#             "wavelength": float(wavelengths[p]) if wavelengths is not None else None,
#             "flux": float(flux[p]),  # 原始 flux（未反转）
#             "prominence": float(props.get("prominences", [None])[i]),
#             "width_wavelength": float(x_axis_slope * width_pix),
#         }

#         # === Step 4: 对 trough 增加 depth / EW 信息 ===
#         if feature == "trough":
#             depth = float(flux_smooth[p])
#             ew_pix = depth * width_pix
#             info.update({
#                 "depth": depth,
#                 "equivalent_width_pixels": ew_pix,
#                 "prominence": float(props.get("prominences", [None])[i])
#             })
#         peaks_info.append(info)

#     return flux_smooth, peaks_info
    
# def _merge_peaks_across_sigmas(
#     feature, wavelengths, flux, peaks_by_sigma,
#     tol_pixels=5
# ):
#     """
#     合并不同 scale 的峰/谷。
#     新版逻辑：
#       - 每个 group 代表同一物理特征；
#       - 每个 σ 内挑选出最可信的代表点；
#       - 不同 σ 的代表点再按权重加权平均。
#     """
#     merged = []
#     for scale_entry in peaks_by_sigma:
#         sigma = scale_entry["sigma"]
#         for p in scale_entry["peaks"]:
#             idx = p["index"]
#             matched = None
#             for g in merged:
#                 rep = int(np.round(np.mean(g["indices"])))
#                 if abs(rep - idx) <= tol_pixels:
#                     matched = g
#                     break
#             if matched is None:
#                 g = {"indices": [idx], "infos": [dict(p, sigma=sigma)]}
#                 merged.append(g)
#             else:
#                 matched["indices"].append(idx)
#                 matched["infos"].append(dict(p, sigma=sigma))

#     consensus = []
#     for g in merged:
#         infos = g["infos"]

#         # === Step 1: 按 σ 分组 ===
#         infos_by_sigma = {}
#         for inf in infos:
#             s = inf["sigma"]
#             infos_by_sigma.setdefault(s, []).append(inf)

#         # === Step 2: 每 σ 选出最可信代表 ===
#         sigma_reps = []
#         for s, lst in infos_by_sigma.items():
#             if feature == "peak":
#                 # 峰：选 flux 最大 或 prominence 最大
#                 best = max(lst, key=lambda x: (x.get("flux", 0), x.get("prominence", 0)))
#             else:
#                 # 谷：选 depth 最大（或 flux 最低）
#                 best = max(lst, key=lambda x: (
#                     x.get("depth", 0),
#                     -x.get("flux", 0)
#                 ))
#             sigma_reps.append(dict(best, sigma=s))

#         # === Step 3: 对代表点进行加权平均 ===
#         # weighted_sum, weight_total = 0.0, 0.0
#         max_sigma, min_sigma = 0.0, np.inf
#         fff = -np.inf
#         for rep in sigma_reps:
#             id = rep["index"]
#             if rep['flux'] > fff:
#                 fff = rep['flux']
#                 rep_idx = id
            
#             sigma = rep["sigma"]
#             idx = rep["index"]
#             max_sigma = max(max_sigma, sigma)
#             min_sigma = min(min_sigma, sigma)
#             # w = 0.5 if sigma == 0 else 1.0 / np.sqrt(sigma)
#             # weighted_sum += idx * w
#             # weight_total += w
#         # rep_idx_ = int(np.round(weighted_sum / weight_total))
#         wlen = float(wavelengths[rep_idx]) if rep_idx < len(wavelengths) else None

#         # === Step 4: 统计特征信息 ===
#         appearances = len(sigma_reps)
#         # widths = [r.get("width_wavelength", 0.0) for r in sigma_reps]
#         # 使用最小sigma对应的 width 作为代表
#         widths = [r.get("width_wavelength", 0.0) for r in sigma_reps if r["sigma"] == min_sigma]
#         mean_flux = float(np.mean([r["flux"] for r in sigma_reps]))
#         scales = [r["sigma"] for r in sigma_reps]

#         if feature == "peak":
#             max_prom = max(r.get("prominence", 0.0) for r in sigma_reps)
#             consensus.append({
#                 "rep_index": rep_idx,
#                 "wavelength": wlen,
#                 "appearances": appearances,
#                 "max_prominence": float(max_prom),
#                 "mean_flux": mean_flux,
#                 "width_mean": float(np.mean(widths)),
#                 "width_in_km_s": float(np.mean(widths)) / wlen * 3e5 if wlen else None,
#                 "seen_in_scales_of_sigma": scales,
#                 "max_sigma_seen": max_sigma,
#             })
#         else:
#             max_depth = max(r.get("depth", 0.0) for r in sigma_reps)
#             mean_ew = np.mean([r.get("equivalent_width_pixels", 0.0) for r in sigma_reps])
#             max_prom = max(r.get("prominence", 0.0) for r in sigma_reps)
#             consensus.append({
#                 "rep_index": rep_idx,
#                 "wavelength": wlen,
#                 "appearances": appearances,
#                 "max_depth": float(max_depth),
#                 "max_prominence": float(max_prom),
#                 "mean_equivalent_width_pixels": float(mean_ew),
#                 "mean_flux": mean_flux,
#                 "width_mean": float(np.mean(widths)),
#                 "width_in_km_s": float(np.mean(widths)) / wlen * 3e5 if wlen else None,
#                 "seen_in_scales_of_sigma": scales,
#                 "min_sigma_seen": min_sigma,
#             })

#     # === Step 5: 排序逻辑 ===
#     if feature == "peak":
#         consensus = sorted(
#             consensus,
#             key=lambda x: (x["max_sigma_seen"], x["max_prominence"], x["appearances"]),
#             reverse=True,
#         )
#     else:
#         consensus = sorted(
#             consensus,
#             key=lambda x: (x["max_depth"], x["mean_equivalent_width_pixels"], x["appearances"]),
#             reverse=True,
#         )
#     return consensus

# def _find_features_multiscale(
#     wavelengths, flux, 
#     state,
#     feature="peak", sigma_list=None,
#     prom=0.01, tol_pixels=10,
#     use_continuum_for_trough=True,
#     min_depth=0.1  # ✅ 新增：按 depth 过滤阈值
# ):
#     """
#     多尺度特征检测器。
#     - 自动支持 peaks / troughs；
#     - trough 可选 continuum 归一化；
#     - prom 同时控制峰和谷的显著性；
#     - min_depth 用于过滤过浅吸收线。
#     """
#     if sigma_list is None:
#         sigma_list = [2, 4, 16]

#     try:
#         x_axis_slope = state["pixel_to_value"]["x"]["a"]
#         wavelengths = np.array(wavelengths)
#         flux = np.array(flux)

#         # continuum 估计（仅 trough）
#         continuum = None
#         if feature == "trough" and use_continuum_for_trough:
#             cont_window = max(51, int(3 * max(sigma_list)))
#             continuum = median_filter(flux, size=cont_window, mode="reflect")
#             continuum = np.where(continuum == 0, 1.0, continuum)
#         # if feature == "peak":
#         #     continuum = gaussian_filter1d(flux, sigma=100)

#         sigma_list = [0] + sigma_list  # 原始光谱权重最高
#         # print(f"Using sigma list: {sigma_list}")

#         peaks_by_sigma = []
#         for s in sigma_list:
#             flux_smooth, peaks_info = _detect_features_on_flux(
#                 feature, flux, x_axis_slope, sigma=float(s),
#                 prominence=prom, height=None,
#                 wavelengths=wavelengths, continuum=continuum
#             )

#             # ✅ 对 troughs 进行 depth 过滤
#             if feature == "trough" and min_depth > 0:
#                 peaks_info = [
#                     t for t in peaks_info if t.get("depth", 0) >= min_depth
#                 ]

#             peaks_by_sigma.append({
#                 "sigma": float(s),
#                 "flux_smooth": flux_smooth.tolist(),
#                 "peaks": peaks_info
#             })

#         return _merge_peaks_across_sigmas(
#             feature, wavelengths, flux, peaks_by_sigma,
#             tol_pixels=tol_pixels
#         )

#     except Exception as e:
#         return json.dumps({"error": str(e)}, ensure_ascii=False)

def savefig_unique(fig, filename, **kwargs):
    """
    保存图片，如果文件已存在则自动添加数字后缀
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    
    fig.savefig(new_filename, bbox_inches='tight', **kwargs)
    print(f"图片已保存为: {new_filename}")

def _plot_spectrum(state: SpectroState):
    # if effective_SNR:
    wavelength = state['spectrum']['new_wavelength']
    flux = state['spectrum']['weighted_flux']
    flux_top = state['spectrum']['max_unresolved_flux']
    flux_bottom = state['spectrum']['min_unresolved_flux']
    effective_snr = state['spectrum']['effective_snr']
    delta_flux = state['spectrum']['delta_flux']
    # effective_snr = np.array(flux)/(np.array(flux_top) - np.array(flux_bottom))

    fig, axs = plt.subplots(2, 1, figsize=(10, 7))

    axs[0].plot(wavelength, flux, color='b', label=r'$\bar F$: signal extracted from picture')
    axs[0].fill_between(wavelength, flux_top, flux_bottom, alpha=0.4, color='gray', label='information lossed in Opencv processing')
    axs[0].set_ylabel('flux')
    axs[0].set_xlabel('wavelength')
    axs[0].legend()  # 设置字号为12

    axs[1].plot(wavelength, effective_snr, c='orange', label=r'$SNR=\frac{{\bar F}_i}{\sigma_{i,j}}$')
    axs[1].set_ylabel('Effective SNR')
    axs[1].set_xlabel('wavelength')
    axs[1].legend(fontsize=15)  # 设置字号为12

    # savefig_unique(fig, os.path.join(state['output_dir'], f'{state['file_name']}_spectrum.png'))
    fig.savefig(os.path.join(state['output_dir'], f'{state['file_name']}_spectrum.png'), bbox_inches='tight')

    # 创建 figure（可选，不创建也会自动生成）
    plt.figure(figsize=(10, 3))
    # 填充区域
    plt.fill_between(wavelength, flux_bottom, flux_top,
                    color='#FFB6A6', alpha=0.5, linewidth=0,
                    label='information lost in OpenCV processing (pink #FFB6A6)')
    # 信号曲线
    plt.plot(wavelength, flux, color='b', lw=1.5,
            label=r'$\bar{F}$: signal extracted from picture (blue)')
    # 坐标轴标签
    plt.xlabel('wavelength')
    plt.ylabel('flux')
    # 图例（字号12）
    plt.legend(fontsize=12)
    # 保存当前 figure
    plt.savefig(
        os.path.join(state['output_dir'], f"{state['file_name']}_spec_extract.png"),
        dpi=150,
        bbox_inches='tight'
    )
    # 关闭当前 figure，防止内存累积（尤其在循环中很重要）
    plt.close()

    try:
        plt.figure(figsize=(10, 3))
        # print(type(state['continuum']['wavelength']))
        # print(type(state['continuum']['flux']))
        contunuum_wavelength = np.array(state['continuum']['wavelength'])
        continuum_flux = np.array(state['continuum']['flux'])
        plt.plot(contunuum_wavelength, continuum_flux, color='orange', label='Continuum')
        # mask = np.isin(wavelength, contunuum_wavelength)
        # errorbar = abs((np.array(delta_flux)[mask]))
        # plt.errorbar(state['continuum']['wavelength'], state['continuum']['flux'], 
        #     yerr=errorbar, fmt='x', markersize=0,
        #     ecolor='red', 
        #     elinewidth=0.8,
        #     capsize=2,
        #     alpha=0.7,
        #     zorder=5,
        #     label='SNR')
        plt.xlabel('wavelength')
        plt.ylabel('flux')
        plt.legend(fontsize=12)
        plt.savefig(
            state['continuum_path'],
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()
    except Exception as e:
        print(f"❌ plot spectrum or features terminated with error: {e}")
        raise

    # img = cv2.imread(state.crop_path)
    # plt.figure(figsize=(10,3))
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image')

    return fig

def _plot_features(state: SpectroState, sigma_list=[2,4,16], feature_number=[10,15]):
    fig = plt.figure(figsize=(10,3))

    plt.plot(state['spectrum']['new_wavelength'], state['spectrum']['weighted_flux'], label='original', c='k', alpha=0.7)

    for sigma in sigma_list:
        sigma_smooth = gaussian_filter1d(state['spectrum']['weighted_flux'], sigma=sigma)
        plt.plot(state['spectrum']['new_wavelength'], sigma_smooth, alpha=0.7, label=rf'$\sigma={sigma}$')

    # 安全地绘制峰值线
    peaks_to_plot = min(feature_number[0], len(state['peaks']))
    for i in range(peaks_to_plot):
        plt.axvline(state['peaks'][i]['wavelength'], linestyle='-', c='red', alpha=0.5)
    
    # 安全地绘制谷值线
    troughs_to_plot = min(feature_number[1], len(state['troughs']))
    for i in range(troughs_to_plot):
        plt.axvline(state['troughs'][i]['wavelength'], linestyle=':', c='red', alpha=0.5)

    plt.plot([], [], linestyle='-', c='red', alpha=0.5, label='peaks')
    plt.plot([], [], linestyle=':', c='blue', alpha=0.5, label='troughs')
    plt.ylabel('flux')
    plt.xlabel('wavelength')
    plt.legend()

    print(f'Plot {peaks_to_plot} peaks and {troughs_to_plot} troughs.')

    # savefig_unique(fig, os.path.join(state['output_dir'], f'{state['file_name']}_features.png'))
    fig.savefig(os.path.join(state['output_dir'], f'{state['file_name']}_features.png'), bbox_inches='tight')
    return fig

def _ROI_features_finding(state: SpectroState):
    spec = state["spectrum"]
    wavelengths = np.array(spec["new_wavelength"])
    flux = np.array(spec["weighted_flux"])
    fig = plt.figure(figsize=(10,3))
    plt.plot(wavelengths, flux, label='original', c='k', alpha=0.7)

    ROI_peaks = []
    ROI_troughs = []
    sigma_list, tol_pixels, prom_peaks, prom_troughs, _, _ = _load_feature_params()
    state['sigma_list'] = sigma_list
    for roi in state['visual_interpretation']['roi']:
        range = roi['wave_range']
        mask = (wavelengths >= range[0]) & (wavelengths <= range[1])
        wave_cut = wavelengths[mask]
        flux_cut = flux[mask]
        if len(wave_cut) == 0:
            print(f"⚠️ ROI {range} out of spectrum range — skipped.")
            continue

        pe = _find_features_multiscale(
            wave_cut, flux_cut,
            state, feature="peak", sigma_list=sigma_list,
            prom=prom_peaks, tol_pixels=tol_pixels,
            use_continuum_for_trough=True
        )
        # print(pe)
        tr = _find_features_multiscale(
            wave_cut, flux_cut,
            state, feature="trough", sigma_list=sigma_list,
            prom=prom_troughs, tol_pixels=tol_pixels,
            use_continuum_for_trough=True,
            min_depth=0.08
        )

        pe_info = {
            'roi_range': range,
            'peaks': pe, 
            'n_peaks': len(pe)
        }

        tr_info = {
            'roi_range': range,
            'troughs': tr, 
            'n_troughs': len(tr)
        }

        ROI_peaks.append(pe_info)
        ROI_troughs.append(tr_info)

        # 使用更Pythonic的方式遍历列表
        for peak_ in pe_info['peaks']:
            plt.axvline(peak_['wavelength'], linestyle='-', c='red', alpha=0.5)
        
        for trough_ in tr_info['troughs']:
            plt.axvline(trough_['wavelength'], linestyle=':', c='blue', alpha=0.5)

    plt.plot([], [], linestyle='-', c='red', alpha=0.5, label='peaks')
    plt.plot([], [], linestyle=':', c='blue', alpha=0.5, label='troughs')  # 注意：图例颜色与实际线条颜色的一致性
    plt.ylabel('flux')
    plt.xlabel('wavelength')
    plt.legend()

    plt.savefig(os.path.join(state['output_dir'], f'{state['file_name']}_ROI.png'), bbox_inches='tight')
    print('ROI done')
    return ROI_peaks, ROI_troughs

from copy import deepcopy


def merge_features(wavelength, flux, global_peaks,
                   global_troughs,
                   ROI_peaks,
                   ROI_troughs,
                   tol_pixels):
    """
    主入口：融合 Global 和 ROI 的 peaks / troughs
    """
    merged_peaks = _process_feature_type(
        wavelength, flux,
        global_list=global_peaks,
        roi_list=ROI_peaks,
        tol_pixels=tol_pixels,
        feature_type="peak"
    )

    merged_troughs = _process_feature_type(
        wavelength, flux,
        global_list=global_troughs,
        roi_list=ROI_troughs,
        tol_pixels=tol_pixels,
        feature_type="trough"
    )

    return merged_peaks, merged_troughs



def _process_feature_type(wavelength, flux, global_list, roi_list, tol_pixels, feature_type):
    """
    处理 peaks 或 troughs
    按 tol_pixels 合并，拆分 global/ROI appearances，global/ROI sigma
    """

    # ---- Step 1：把全局 + ROI 统一展开成 flat list ----
    # 为了后续按 pixel rep_index 合并，我们需要记录 pixel 位置
    # 假设 rep_index 就是 pixel index
    try:
        flat = []

        # 全局
        for g in global_list:
            item = deepcopy(g)
            item["_is_global"] = True
            item["_global_sigma"] = item.get("max_sigma_seen", None)
            item["_roi_sigma"] = None
            item["_global_app"] = item.get("appearances", 0)
            item["_roi_app"] = 0
            item["_rep_index"] = item["rep_index"]
            flat.append(item)

        # ROI
        for roi in roi_list:
            # for r in roi[feature_type + "s"]:
            item = deepcopy(roi)
            item["_is_global"] = False
            item["_global_sigma"] = None
            item["_roi_sigma"] = item.get("max_sigma_seen", None)
            item["_global_app"] = 0
            item["_roi_app"] = item.get("appearances", 0)
            item["_rep_index"] = item["rep_index"]
            flat.append(item)

        # ---- Step 2：按照 pixel 距离 tol_pixels 进行合并 ----
        flat.sort(key=lambda x: x["_rep_index"])

        groups = []
        current_group = [flat[0]]

        for x in flat[1:]:
            # 检查当前元素与当前组内所有元素的差值
            can_add_to_current_group = True
            for item in current_group:
                if abs(x["_rep_index"] - item["_rep_index"]) > tol_pixels:
                    can_add_to_current_group = False
                    break
            
            if can_add_to_current_group:
                # 如果与当前组内所有元素的差值都不超过 tol_pixels，则加入当前组
                current_group.append(x)
            else:
                # 否则，保存当前组并开始新分组
                groups.append(current_group)
                current_group = [x]

        # 循环结束后，将最后一个分组加入结果
        groups.append(current_group)

        # ---- Step 3：对每个 group 做融合 ----
        merged = []
        for group in groups:

            # 把 group 中 global/ROI 的 sigma / appearance 分开统计
            all_global_sigma = [x["_global_sigma"] for x in group if x["_global_sigma"] is not None]
            all_roi_sigma =   [x["_roi_sigma"]    for x in group if x["_roi_sigma"]    is not None]

            # appearances
            total_global_app = sum(x["_global_app"] for x in group)
            total_roi_app    = sum(x["_roi_app"]    for x in group)

            # ---- Feature group 的代表：选 flux 最大的 wavelength ----
            if feature_type == "peak":
                # flux 越大越代表 peak
                rep = max(group, key=lambda x: flux[x["_rep_index"]])
            else:
                # trough：flux 越小越深（负值），代表性强 → 选 mean_flux 最小的
                rep = min(group, key=lambda x: flux[x["_rep_index"]])

            # ---- fusion 后的结构 ----
            fused = {
                "rep_index": rep["_rep_index"],
                "wavelength": rep["wavelength"],

                # --- appearances ---
                "global_appearances": total_global_app,
                "roi_appearances": total_roi_app,

                # --- sigma ---
                "max_global_sigma_seen": max(all_global_sigma) if all_global_sigma else None,
                "max_roi_sigma_seen":    max(all_roi_sigma)    if all_roi_sigma    else None,
            }

            # ---- feature-specific fields ----
            if feature_type == "peak":
                # 选择 group 中 max prominence 最大者
                fused["max_prominence"] = max(x.get("max_prominence", 0) for x in group)
                fused["mean_flux"] = rep["mean_flux"]
                fused["width_mean"] = rep["width_mean"]
                fused["width_in_km_s"] = rep["width_in_km_s"]

            else:  # trough
                fused["max_depth"] = max(x.get("max_depth", 0) for x in group)
                fused["max_prominence"] = max(x.get("max_prominence", 0) for x in group)
                fused["mean_equivalent_width_pixels"] = rep["mean_equivalent_width_pixels"]
                fused["mean_flux"] = rep["mean_flux"]
                fused["width_mean"] = rep["width_mean"]
                fused["width_in_km_s"] = rep["width_in_km_s"]

            merged.append(fused)

        # ---- Step 4：排序 ----

        if feature_type == "peak":
            merged.sort(
                key=lambda x: (
                    # 1. 全局 max sigma
                    -999 if x["max_global_sigma_seen"] is None else -x["max_global_sigma_seen"],
                    # 4. ROI sigma
                    -999 if x["max_roi_sigma_seen"] is None else -x["max_roi_sigma_seen"],
                    # 2. max prominence
                    -x["max_prominence"],
                    # 3. 全局 appearances
                    -x["mean_flux"],
                    # 5. max prominence again
                    -x["max_prominence"],
                    # 6. ROI appearances
                    -x["roi_appearances"],
                )
            )
        else:  # trough
            merged.sort(
                key=lambda x: (
                    -x["max_depth"],
                    -x["mean_equivalent_width_pixels"],
                    -x["global_appearances"],
                    -x["roi_appearances"],
                )
            )

        return merged
    except Exception as e:
        logging.error(f"Error in _process_feature_type: {e}")

