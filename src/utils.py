import cv2
import os
import pytesseract
import json
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from typing import Any, List, Dict, Tuple, Optional, Union
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_widths

def safe_to_bool(value):
    """专门处理true/True相关值的转换"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ['true', '1', 't', 'yes', 'y']
    return bool(value)

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def user_query(system_prompt, user_prompt, image_path=None):
    # 创建系统消息
    system_message = SystemMessage(content=system_prompt)
    
    if not image_path:
        # 如果没有图片，创建纯文本用户消息
        human_message = HumanMessage(content=[{"type": "text", "text": user_prompt}])
        return [system_message, human_message]
    
    # 处理单张图片或多张图片的情况
    base64_image = image_to_base64(image_path)
    
    # 构建包含文本和图片的用户消息内容
    content = [{"type": "text", "text": user_prompt}]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    })
    
    human_message = HumanMessage(content=content)
    return [system_message, human_message]
    
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

def _load_feature_params():
    """安全读取峰值/谷值检测参数"""
    sigma_list = parse_list(os.getenv("SIGMA_LIST"), [2, 4, 16])
    tol_pixels = getenv_int("TOL_PIXELS", 10)
    prom_peaks = getenv_float("PROM_THRESHOLD_PEAKS", 0.01)
    prom_troughs = getenv_float("PROM_THRESHOLD_TROUGHS", 0.05)
    weight_original = getenv_float("WEIGHT_ORIGINAL", 0.5)
    plot_peaks = getenv_int("PLOT_PEAKS_NUMBER", 10)
    plot_troughs = getenv_int("PLOT_TROUGHS_NUMBER", 15)

    return sigma_list, tol_pixels, prom_peaks, prom_troughs, weight_original, plot_peaks, plot_troughs

def _detect_axis_ticks(image_path, config=None):
    if config is None:
        # config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.-eE'
        config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.-eE+ '  # 注意末尾加空格

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    data = pytesseract.image_to_data(
        thresh, config=config, output_type=pytesseract.Output.DICT
    )

    tick_values = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text != "":
            try:
                value = float(text)
                (x, y, w, h) = (data['left'][i], data['top'][i],
                                data['width'][i], data['height'][i])
                cx, cy = x + w//2, y + h//2
                tick_values.append({
                    "value": value,
                    "position": [cx, cy],
                    "bounding-box-scale": [w,h]
                })
            except ValueError:
                pass

    return tick_values

def _detect_chart_border(image_path: str, margin: int = 10) -> dict:
    """
    检测图像中图表的外围边框，并微调尺寸。
    
    参数:
        image_path: 图像文件路径
        margin: 调整边框的像素量（正数表示收缩边框）
    
    返回:
        dict 包含边框位置: {"x": int, "y": int, "w": int, "h": int}
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 形态学操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未找到任何轮廓")
    
    # 找到最大轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # 微调边框
    x += margin
    y += margin
    w -= 2*margin
    h -= 2*margin

    return {"x": x, "y": y, "w": w, "h": h}

def _crop_img(image_path: str, border_info: dict, save_path: str) -> str:
    """
    裁剪图像指定区域并保存。
    
    参数:
        image_path: 输入图像路径
        save_path: 保存裁剪后图像路径
    
    返回:
        save_path
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    x = border_info['x']
    y = border_info['y']
    w = border_info['w']
    h = border_info['h']

    img_cropped = img[y:y+h, x:x+w]
    cv2.imwrite(save_path, img_cropped)

    print(f'cropped image is saved to {save_path}')
    
    return save_path

def _remap_to_cropped_canvas(old_info, chart_border):
    """
    将原图坐标映射到裁剪后的画布上，自动处理 None 坐标。
    """
    x0, y0, w, h = chart_border["x"], chart_border["y"], chart_border["w"], chart_border["h"]

    new_info = []
    for d in old_info:
        ox, oy = d.get("position_x") if d.get("position_x") is not None else None, d.get("position_y") if d.get("position_y") is not None else None

        new_d = d.copy()

        # if ox is None or oy is None:
        #     # 如果原始坐标缺失，保持原值或设为 None
        #     new_d["position_x"] = ox
        #     new_d["position_y"] = oy
        # else:
        # 重映射到裁剪画布
        nx = ox - x0 if ox is not None else None
        ny = oy - y0 if oy is not None else None
        # 越界裁剪
        nx = max(0, min(nx, w - 1)) if ox is not None else None
        ny = max(0, min(ny, h - 1)) if oy is not None else None
        new_d["position_x"] = nx
        new_d["position_y"] = ny

        new_info.append(new_d)

    return new_info

def linear_func(x, a, b):
    return a * x + b

def _pixel_tickvalue_fitting(arr: list) -> dict:
    """
    对刻度数据做加权线性拟合（支持 x/y 轴分开）。
    输入: Python list，每个元素为 dict
    输出: dict 包含各轴拟合结果
    """
    results = {}
    for axis in ["x","y"]:
        # 提取有效数据
        values, pixels, sigmas, confs = [], [], [], []
        for d in arr:
            if d["axis"] == axis and d[f'position_{axis}'] is not None:
                values.append(float(d["value"]))
                pixels.append(float(d["position_x"] if axis == 'x' else d["position_y"]))
                sigmas.append(float(d["sigma_pixel"]) if d["sigma_pixel"] is not None else np.inf)
                confs.append(float(d["conf_llm"]) if d["conf_llm"] is not None else 1.0)

        if len(values) < 2:
            continue

        values = np.array(values, dtype=float)
        pixels = np.array(pixels, dtype=float)
        sigmas = np.array(sigmas, dtype=float)
        confs = np.array(confs, dtype=float)

        # 有效 sigma
        sigma_eff = sigmas / np.sqrt(confs)

        # 拟合
        popt, _ = curve_fit(
            linear_func,
            pixels,
            values,
            sigma=sigma_eff,
            absolute_sigma=True
        )
        a_fit, b_fit = popt
        value_fit = linear_func(pixels, a_fit, b_fit)
        residual = values - value_fit
        rms = np.sqrt(np.mean(residual**2))

        results[axis] = {
            "a": float(a_fit),
            "b": float(b_fit),
            "rms": float(rms),
            "residuals": residual.tolist()
        }

    return results

def _process_and_extract_curve_points(input_path: str):
    """
    读取图像，去除背景并转换为二值图像，提取曲线的像素点云，保存到LOCAL_VARS
    
    参数：
    - input_path：原始图像文件路径
    - output_dir：处理后图像保存路径
    
    返回：
    - curve_points: 曲线像素点云（列表形式，包含每个点的(x, y)坐标）
    - curve_gray_values: 曲线像素灰度值（列表形式）
    """
    # 1. 读取原始图像
    img = cv2.imread(input_path)
    
    # 2. 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. 二值化处理，背景为白色，曲线为黑色
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 4. 提取白色曲线像素点（黑色曲线会变为白色，背景变为黑色）
    curve_points = []
    curve_gray_values = []

    # 遍历所有像素点，提取白色区域（即曲线部分）
    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            if binary[y, x] == 255:  # 白色区域（曲线）
                curve_points.append([x, y])
                curve_gray_values.append(gray[y, x])

    curve_gray_values = np.array(curve_gray_values, dtype=np.float64)
    
    # 返回曲线点数量（简化信息）
    return curve_points, curve_gray_values

def weighted_average_flux(wavelength, flux, gray):
    """
    根据灰度值对同一波长的flux进行加权平均。

    参数：
    - wavelength: 一维数组，表示波长
    - flux: 一维数组，表示光谱强度（flux）
    - gray: 一维数组，表示每个像素的灰度值（权重）

    返回：
    - new_wavelength: 每个唯一波长的数组
    - weighted_flux: 每个波长对应的加权平均flux值
    """
    df = pd.DataFrame({
        'wavelength': wavelength,
        'flux': flux,
        'gray': gray
    })

    # 对每个唯一的波长进行加权平均
    weighted_flux = df.groupby('wavelength', group_keys=False).apply(
        # lambda x: np.sum(x['flux'] * x['gray']) / np.sum(x['gray']),
        lambda x: np.average(x['flux']),
        include_groups=False
    )

    unique_wavelength = weighted_flux.index.to_numpy()
    return unique_wavelength, weighted_flux.to_numpy()

def _convert_to_spectrum(points, gray, axis_fitting_info):
    """
    转换曲线的像素坐标信息到波长（wavelength）和光谱强度（flux），
    并根据灰度值对同一波长的flux进行加权平均。

    输入：
    - points: 曲线像素坐标，形状为 (n, 2) 的数组，第一列为 x 坐标，第二列为 y 坐标
    - gray: 灰度值，形状为 (n,) 的数组，表示每个像素的灰度值（权重）
    - x_axis: x 轴像素到物理量的转换系数，字典形式 {'a': x_scale, 'b': x_offset}
    - y_axis: y 轴像素到物理量的转换系数，字典形式 {'a': y_scale, 'b': y_offset}

    输出：
    - spectrum_dict: 包含转换后的波长、flux 和加权平均后的波长与flux的字典
    """
    # 提取坐标
    points = np.array(points)
    xs = points[:, 0]
    ys = points[:, 1]

    # 提取 x 轴和 y 轴的物理量转换系数
    a_y = axis_fitting_info['y']['a']
    b_y = axis_fitting_info['y']['b']
    flux = a_y * ys + b_y

    a_x = axis_fitting_info['x']['a']
    b_x = axis_fitting_info['x']['b']
    wavelength = a_x * xs + b_x

    # 计算加权平均flux
    unique_wavelength, weighted_flux = weighted_average_flux(wavelength, flux, gray)

    max_unresolved_flux = []
    min_unresolved_flux = []
    for i, w in enumerate(unique_wavelength):
        unresolved_flux = flux[wavelength == w]
        max_unresolved_flux.append(np.max(unresolved_flux))
        min_unresolved_flux.append(np.min(unresolved_flux))
    
    denominator = np.array(max_unresolved_flux) - np.array(min_unresolved_flux)
    effective_snr = np.where(
        denominator != 0,
        np.array(weighted_flux) / denominator,
        np.nan  # 或 np.nan，取决于你希望如何表示“无效 SNR”
    )
    # effective_snr = np.array(weighted_flux.tolist())/(np.array(max_unresolved_flux) - np.array(min_unresolved_flux))

    # 构造最终结果
    spectrum_dict = {
        'flux': flux.tolist(),
        'wavelength': wavelength.tolist(),
        'new_wavelength': unique_wavelength.tolist(),
        'weighted_flux': weighted_flux.tolist(),
        'max_unresolved_flux': max_unresolved_flux,
        'min_unresolved_flux': min_unresolved_flux, 
        'effective_snr': effective_snr
    }

    return spectrum_dict

def _detect_features_on_flux(
    feature, flux, x_axis_slope, sigma, prominence=None, height=None,
    wavelengths=None, continuum=None
):
    """
    平滑后检测峰/谷，返回平滑光谱和峰信息。
    支持 trough 检测（通过 flux 取负或 continuum 归一化）。
    """
    # === Step 1: 数据准备 ===
    if feature == "trough":
        if continuum is not None:
            flux_proc = flux / continuum
            flux_proc = 1.0 - flux_proc  # 变成“吸收强度”，越大越深
        else:
            flux_proc = -flux.copy()
    else:
        flux_proc = flux - continuum if continuum is not None else flux.copy()

    # === Step 2: 平滑 ===
    flux_smooth = gaussian_filter1d(flux_proc, sigma=sigma) if sigma > 0 else flux_proc

    # === Step 3: 峰检测 ===
    peaks, props = find_peaks(flux_smooth, height=height, prominence=prominence)
    widths_res = peak_widths(flux_smooth, peaks, rel_height=0.5)

    peaks_info = []
    for i, p in enumerate(peaks):
        width_pix = widths_res[0][i]
        info = {
            "index": int(p),
            "wavelength": float(wavelengths[p]) if wavelengths is not None else None,
            "flux": float(flux[p]),  # 原始 flux（未反转）
            "prominence": float(props.get("prominences", [None])[i]),
            "width_wavelength": float(x_axis_slope * width_pix),
        }

        # === Step 4: 对 trough 增加 depth / EW 信息 ===
        if feature == "trough":
            depth = float(flux_smooth[p])
            ew_pix = depth * width_pix
            info.update({
                "depth": depth,
                "equivalent_width_pixels": ew_pix,
                "prominence": float(props.get("prominences", [None])[i])
            })
        peaks_info.append(info)

    return flux_smooth, peaks_info

# def _detect_features_on_flux(
#     feature, flux, x_axis_slope, sigma, prominence=None, height=None,
#     wavelengths=None, continuum=None
# ):
#     """
#     平滑后检测峰/谷，返回平滑光谱和峰信息。
#     """
#     # === Step 0: 输入验证 ===
#     if flux is None:
#         print(f"❌ ERROR: flux is None for sigma={sigma}")
#         return np.array([]), []
    
#     if wavelengths is not None and len(wavelengths) != len(flux):
#         print(f"❌ ERROR: wavelengths and flux length mismatch: {len(wavelengths)} != {len(flux)}")
#         return np.array([]), []
    
#     # DEBUG: 添加调用栈信息
#     import traceback
#     # print(f"=== Called _detect_features_on_flux (sigma={sigma}) ===")
#     # print(f"flux type: {type(flux)}, shape: {flux.shape if hasattr(flux, 'shape') else 'N/A'}")
#     # print(f"wavelengths type: {type(wavelengths)}, shape: {wavelengths.shape if wavelengths is not None and hasattr(wavelengths, 'shape') else 'N/A'}")
    
#     # === Step 1: 数据准备 ===
#     try:
#         if feature == "trough":
#             if continuum is not None:
#                 # 确保 continuum 不是全零
#                 if np.all(continuum == 0):
#                     flux_proc = -flux.copy()
#                 else:
#                     flux_proc = flux / continuum
#                     flux_proc = 1.0 - flux_proc  # 变成"吸收强度"，越大越深
#             else:
#                 flux_proc = -flux.copy()
#         else:
#             flux_proc = flux - continuum if continuum is not None else flux.copy()
        
#         # 确保 flux_proc 是有效的 numpy 数组
#         flux_proc = np.array(flux_proc, dtype=np.float64)
        
#         # 检查是否包含 NaN 或 Inf
#         if np.any(np.isnan(flux_proc)) or np.any(np.isinf(flux_proc)):
#             print(f"⚠️ WARNING: flux_proc contains NaN or Inf values for sigma={sigma}")
#             flux_proc = np.nan_to_num(flux_proc, nan=0.0, posinf=0.0, neginf=0.0)
        
#         # 打印长度信息（如果需要调试）
#         # print(f"sigma={sigma}: flux_proc length = {len(flux_proc)}")
        
#         l_data = len(flux_proc)
        
#         # === 镜像延拓 ===
#         # 使用安全的延拓方式
#         if l_data > 0:
#             # 取适当的延拓长度（不超过数据长度）
#             pad_len = min(l_data, 50)  # 限制延拓长度
            
#             # 左延拓
#             left_pad = flux_proc[pad_len:0:-1] if pad_len > 0 else np.array([])
            
#             # 右延拓
#             right_pad = flux_proc[-2:-pad_len-2:-1] if pad_len > 0 and l_data > 2 else np.array([])
            
#             # 拼接
#             flux_proc = np.concatenate([left_pad, flux_proc, right_pad])
#         else:
#             # 如果数据为空，直接返回
#             return np.array([]), []
        
#         # === Step 2: 平滑 ===
#         if sigma > 0 and len(flux_proc) > 0:
#             flux_smooth = gaussian_filter1d(flux_proc, sigma=sigma)
#         else:
#             flux_smooth = flux_proc
        
#         # === Step 3: 峰检测 ===
#         if len(flux_smooth) == 0:
#             return np.array([]), []
        
#         # 确保 prominence 是正数
#         if prominence is not None and prominence <= 0:
#             prominence = None
        
#         peaks, props = find_peaks(flux_smooth, height=height, prominence=prominence)
        
#         if len(peaks) == 0:
#             # 返回原始数据段的平滑结果
#             if l_data > 0 and len(flux_smooth) >= 2 * l_data:
#                 return flux_smooth[l_data:2*l_data], []
#             else:
#                 return flux_smooth, []
        
#         # 计算峰宽
#         widths_res = peak_widths(flux_smooth, peaks, rel_height=0.5)
        
#         peaks_info = []
#         for i, p in enumerate(peaks):
#             # 检查是否在中间段（原始数据）
#             if l_data <= p < 2 * l_data:
#                 orig_idx = p - l_data
                
#                 # 确保索引有效
#                 if 0 <= orig_idx < l_data:
#                     width_pix = widths_res[0][i]
                    
#                     # 获取波长值
#                     wlen = None
#                     if wavelengths is not None and orig_idx < len(wavelengths):
#                         wlen = float(wavelengths[orig_idx])
                    
#                     info = {
#                         "index": int(orig_idx),
#                         "wavelength": wlen,
#                         "flux": float(flux[orig_idx]),
#                         "prominence": float(props.get("prominences", [0])[i]) if "prominences" in props else None,
#                         "width_wavelength": float(x_axis_slope * width_pix) if x_axis_slope is not None else None,
#                         "width_pixels": float(width_pix),
#                     }
                    
#                     # 对于 trough
#                     if feature == "trough":
#                         depth = float(flux_smooth[p])
#                         ew_pix = depth * width_pix
#                         info.update({
#                             "depth": depth,
#                             "equivalent_width_pixels": ew_pix,
#                         })
                    
#                     peaks_info.append(info)
        
#         # 返回中间段（原始数据）的平滑结果
#         if l_data > 0 and len(flux_smooth) >= 2 * l_data:
#             return flux_smooth[l_data:2*l_data], peaks_info
#         else:
#             return flux_smooth, peaks_info
            
#     except Exception as e:
#         print(f"❌ ERROR in _detect_features_on_flux (sigma={sigma}): {e}")
#         import traceback
#         traceback.print_exc()
#         return np.array([]), []
    
def _merge_peaks_across_sigmas(
    feature, wavelengths, peaks_by_sigma,
    tol_pixels=5, weight_original=2.0
):
    """
    合并不同 scale 的峰/谷。
    新版逻辑：
      - 每个 group 代表同一物理特征；
      - 每个 σ 内挑选出最可信的代表点；
      - 不同 σ 的代表点再按权重加权平均。
    """
    merged = []
    for scale_entry in peaks_by_sigma:
        sigma = scale_entry["sigma"]
        for p in scale_entry["peaks"]:
            idx = p["index"]
            matched = None
            for g in merged:
                rep = int(np.round(np.mean(g["indices"])))
                if abs(rep - idx) <= tol_pixels:
                    matched = g
                    break
            if matched is None:
                g = {"indices": [idx], "infos": [dict(p, sigma=sigma)]}
                merged.append(g)
            else:
                matched["indices"].append(idx)
                matched["infos"].append(dict(p, sigma=sigma))

    consensus = []
    for g in merged:
        infos = g["infos"]

        # === Step 1: 按 σ 分组 ===
        infos_by_sigma = {}
        for inf in infos:
            s = inf["sigma"]
            infos_by_sigma.setdefault(s, []).append(inf)

        # === Step 2: 每 σ 选出最可信代表 ===
        sigma_reps = []
        for s, lst in infos_by_sigma.items():
            if feature == "peak":
                # 峰：选 flux 最大 或 prominence 最大
                best = max(lst, key=lambda x: (x.get("flux", 0), x.get("prominence", 0)))
            else:
                # 谷：选 depth 最大（或 flux 最低）
                best = max(lst, key=lambda x: (
                    x.get("depth", 0),
                    -x.get("flux", 0)
                ))
            sigma_reps.append(dict(best, sigma=s))

        # === Step 3: 对代表点进行加权平均 ===
        weighted_sum, weight_total = 0.0, 0.0
        max_sigma, min_sigma = 0.0, np.inf
        fff = -np.inf
        for rep in sigma_reps:
            # if rep['flux'] > fff:
            #     fff = rep['flux']
            # rep_idx = rep["index"]
            sigma = rep["sigma"]
            idx = rep["index"]
            max_sigma = max(max_sigma, sigma)
            min_sigma = min(min_sigma, sigma)
            w = weight_original if sigma == 0 else 1.0 / np.sqrt(sigma)
            weighted_sum += idx * w
            weight_total += w
        rep_idx = int(np.round(weighted_sum / weight_total))
        wlen = float(wavelengths[rep_idx]) if rep_idx < len(wavelengths) else None

        # === Step 4: 统计特征信息 ===
        appearances = len(sigma_reps)
        # widths = [r.get("width_wavelength", 0.0) for r in sigma_reps]
        # 使用最小sigma对应的 width 作为代表
        widths = [r.get("width_wavelength", 0.0) for r in sigma_reps if r["sigma"] == min_sigma]
        mean_flux = float(np.mean([r["flux"] for r in sigma_reps]))
        scales = [r["sigma"] for r in sigma_reps]

        if feature == "peak":
            max_prom = max(r.get("prominence", 0.0) for r in sigma_reps)
            consensus.append({
                "rep_index": rep_idx,
                "wavelength": wlen,
                "appearances": appearances,
                "max_prominence": float(max_prom),
                "mean_flux": mean_flux,
                "width_mean": float(np.mean(widths)),
                "width_in_km_s": float(np.mean(widths)) / wlen * 3e5 if wlen else None,
                "seen_in_scales_of_sigma": scales,
                "max_sigma_seen": max_sigma,
            })
        else:
            max_depth = max(r.get("depth", 0.0) for r in sigma_reps)
            mean_ew = np.mean([r.get("equivalent_width_pixels", 0.0) for r in sigma_reps])
            max_prom = max(r.get("prominence", 0.0) for r in sigma_reps)
            consensus.append({
                "rep_index": rep_idx,
                "wavelength": wlen,
                "appearances": appearances,
                "max_depth": float(max_depth),
                "max_prominence": float(max_prom),
                "mean_equivalent_width_pixels": float(mean_ew),
                "mean_flux": mean_flux,
                "width_mean": float(np.mean(widths)),
                "width_in_km_s": float(np.mean(widths)) / wlen * 3e5 if wlen else None,
                "seen_in_scales_of_sigma": scales,
                "min_sigma_seen": min_sigma,
            })

    # === Step 5: 排序逻辑 ===
    if feature == "peak":
        consensus = sorted(
            consensus,
            key=lambda x: (x["max_sigma_seen"], x["max_prominence"], x["appearances"]),
            reverse=True,
        )
    else:
        consensus = sorted(
            consensus,
            key=lambda x: (x["max_depth"], x["mean_equivalent_width_pixels"], x["appearances"]),
            reverse=True,
        )
    return consensus

def _find_features_multiscale(
    wavelengths, flux, 
    state,
    feature="peak", sigma_list=None,
    prom=0.01, tol_pixels=10, weight_original=1.0,
    use_continuum_for_trough=True,
    min_depth=0.1  # ✅ 新增：按 depth 过滤阈值
):
    """
    多尺度特征检测器。
    - 自动支持 peaks / troughs；
    - trough 可选 continuum 归一化；
    - prom 同时控制峰和谷的显著性；
    - min_depth 用于过滤过浅吸收线。
    """
    if sigma_list is None:
        sigma_list = [2, 4, 16]

    try:
        x_axis_slope = state["pixel_to_value"]["x"]["a"]
        wavelengths = np.array(wavelengths)
        flux = np.array(flux)

        # continuum 估计（仅 trough）
        continuum = None
        if feature == "trough" and use_continuum_for_trough:
            cont_window = max(51, int(3 * max(sigma_list)))
            continuum = median_filter(flux, size=cont_window, mode="reflect")
            continuum = np.where(continuum == 0, 1.0, continuum)
        # if feature == "peak":
        #     continuum = gaussian_filter1d(flux, sigma=300)

        sigma_list = [0] + sigma_list  # 原始光谱权重最高
        print(f"Using sigma list: {sigma_list}")

        peaks_by_sigma = []
        for s in sigma_list:
            flux_smooth, peaks_info = _detect_features_on_flux(
                feature, flux, x_axis_slope, sigma=float(s),
                prominence=prom, height=None,
                wavelengths=wavelengths, continuum=continuum
            )

            # ✅ 对 troughs 进行 depth 过滤
            if feature == "trough" and min_depth > 0:
                peaks_info = [
                    t for t in peaks_info if t.get("depth", 0) >= min_depth
                ]

            peaks_by_sigma.append({
                "sigma": float(s),
                "flux_smooth": flux_smooth.tolist(),
                "peaks": peaks_info
            })

        return _merge_peaks_across_sigmas(
            feature, wavelengths, peaks_by_sigma,
            tol_pixels=tol_pixels, weight_original=weight_original
        )

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

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

def _plot_spectrum(state):
    # if effective_SNR:
    wavelength = state['spectrum']['new_wavelength']
    flux = state['spectrum']['weighted_flux']
    flux_top = state['spectrum']['max_unresolved_flux']
    flux_bottom = state['spectrum']['min_unresolved_flux']
    effective_snr = state['spectrum']['effective_snr']
    # effective_snr = np.array(flux)/(np.array(flux_top) - np.array(flux_bottom))

    fig, axs = plt.subplots(2, 1, figsize=(10, 7))

    axs[0].plot(wavelength, flux, color='b', label=r'$\bar F$: signal extracted from picture')
    axs[0].fill_between(wavelength, flux_top, flux_bottom, alpha=0.4, color='gray', label='information lossed in Opencv processing')
    axs[0].set_ylabel('flux')
    axs[0].set_xlabel('wavelength')
    axs[0].legend()  # 设置字号为12

    axs[1].plot(wavelength, effective_snr, c='orange', label=r'$\frac{\bar F}{F_\mathrm{top}-F_\mathrm{bottom}}$')
    axs[1].set_ylabel('Effective SNR')
    axs[1].set_xlabel('wavelength')
    axs[1].legend(fontsize=15)  # 设置字号为12

    # savefig_unique(fig, os.path.join(state['output_dir'], f'{state['image_name']}_spectrum.png'))
    fig.savefig(os.path.join(state['output_dir'], f'{state['image_name']}_spectrum.png'), bbox_inches='tight')

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
        os.path.join(state['output_dir'], f"{state['image_name']}_spec_extract.png"),
        dpi=150,
        bbox_inches='tight'
    )
    # 关闭当前 figure，防止内存累积（尤其在循环中很重要）
    plt.close()

    try:
        plt.figure(figsize=(10, 3))
        # print(type(state['continuum']['wavelength']))
        # print(type(state['continuum']['flux']))
        plt.plot(state['continuum']['wavelength'], state['continuum']['flux'], color='orange', label='Continuum')
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

def _plot_features(state, sigma_list=[2,4,16], feature_number=[10,15]):
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
    plt.plot([], [], linestyle=':', c='blue', alpha=0.5, label='troughs')  # 注意：图例颜色与实际线条颜色的一致性
    plt.ylabel('flux')
    plt.xlabel('wavelength')
    plt.legend()

    print(f'Plot {peaks_to_plot} peaks and {troughs_to_plot} troughs.')

    # savefig_unique(fig, os.path.join(state['output_dir'], f'{state['image_name']}_features.png'))
    fig.savefig(os.path.join(state['output_dir'], f'{state['image_name']}_features.png'), bbox_inches='tight')
    return fig  # 建议返回 fig 对象

def _ROI_features_finding(state):
    spec = state["spectrum"]
    wavelengths = np.array(spec["new_wavelength"])
    flux = np.array(spec["weighted_flux"])
    fig = plt.figure(figsize=(10,3))
    plt.plot(wavelengths, flux, label='original', c='k', alpha=0.7)

    ROI_peaks = []
    ROI_troughs = []
    sigma_list, tol_pixels, prom_peaks, prom_troughs, weight_original, _, _ = _load_feature_params()
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
            prom=prom_peaks, tol_pixels=tol_pixels, weight_original=weight_original,
            use_continuum_for_trough=True
        )
        # print(pe)
        tr = _find_features_multiscale(
            wave_cut, flux_cut,
            state, feature="trough", sigma_list=sigma_list,
            prom=prom_troughs, tol_pixels=tol_pixels, weight_original=weight_original,
            use_continuum_for_trough=True,
            min_depth=0.08
        )

        pe_info = {
            'roi_range': range,
            'peaks': pe,          # list of dict: [{'wavelength':..., 'flux':..., ...}]
            'n_peaks': len(pe)
        }

        tr_info = {
            'roi_range': range,
            'troughs': tr,        # ← 关键：改名！
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

    plt.savefig(os.path.join(state['output_dir'], f'{state['image_name']}_ROI.png'), bbox_inches='tight')
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

def plot_merged_features(state):
    sigma_list, _, _, _, _, plot_peaks, plot_troughs = _load_feature_params()
    fig = plt.figure(figsize=(10,3))
    spec = state["spectrum"]
    wavelengths = np.array(spec["new_wavelength"])
    flux = np.array(spec["weighted_flux"])
    plt.plot(wavelengths, flux, label='original', c='k', alpha=0.7)

    for sigma in sigma_list:
        sigma_smooth = gaussian_filter1d(state['spectrum']['weighted_flux'], sigma=sigma)
        plt.plot(state['spectrum']['new_wavelength'], sigma_smooth, alpha=0.7, label=rf'$\sigma={sigma}$')

    peaks_to_plot = min(plot_peaks, len(state['cleaned_peaks']))
    troughs_to_plot = min(plot_troughs, len(state['cleaned_troughs']))
    for peak_ in state['cleaned_peaks'][:peaks_to_plot]:
        plt.axvline(peak_['wavelength'], linestyle='-', c='red', alpha=0.5)
    for trough_ in state['cleaned_troughs'][:troughs_to_plot]:
        if trough_['wavelength'] > 0:
            plt.axvline(trough_['wavelength'], linestyle=':', c='blue', alpha=0.5)

    plt.plot([], [], linestyle='-', c='red', alpha=0.5, label='peaks')
    plt.plot([], [], linestyle=':', c='blue', alpha=0.5, label='troughs')  # 注意：图例颜色与实际线条颜色的一致性
    
    plt.ylabel('flux')
    plt.xlabel('wavelength')
    plt.legend()
    print(f'Plot {peaks_to_plot} peaks and {troughs_to_plot} troughs.')

    # savefig_unique(fig, os.path.join(state['output_dir'], f'{state['image_name']}_features.png'))
    fig.savefig(os.path.join(state['output_dir'], f'{state['image_name']}_features.png'), bbox_inches='tight')

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
                # 生成重叠区域名称
                overlap_name = f"{band_names[i]}-{band_names[j]}"
                result[overlap_name] = [overlap_start, overlap_end]
    
    return result