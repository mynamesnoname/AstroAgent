import cv2
import os
import pytesseract
import json
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from typing import Any, List, Dict, Tuple, Optional, Union
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_widths


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def user_query(prompt, image_path=None):
    if not image_path:
        return [{"role": "user", "content": prompt}]
        # return HumanMessage(content=[{"type": "text", "text": prompt}])
    
    # 处理单张图片或多张图片的情况
    base64_image = image_to_base64(image_path)

#     prompt_ = prompt + f"""
# 光谱图为
# {base64_image}
# """
    # 构建消息内容
    content = [{"type": "text", "text": prompt}]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    })
    return [{"role": "user", "content": content}]
    
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


import pandas as pd

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
        lambda x: np.sum(x['flux'] * x['gray']) / np.sum(x['gray']),
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
        flux_proc = flux.copy()

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
            "wavelength": float(wavelengths[p]) if wavelengths is not None else float(p),
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
                "equivalent_width_pixels": ew_pix
            })
        peaks_info.append(info)

    return flux_smooth, peaks_info

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
                best = max(lst, key=lambda x: (x.get("prominence", 0), x.get("flux", 0)))
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
        for rep in sigma_reps:
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
        widths = [r.get("width_wavelength", 0.0) for r in sigma_reps]
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
                "seen_in_scales_of_sigma": scales,
                "max_sigma_seen": max_sigma,
            })
        else:
            max_depth = max(r.get("depth", 0.0) for r in sigma_reps)
            mean_ew = np.mean([r.get("equivalent_width_pixels", 0.0) for r in sigma_reps])
            consensus.append({
                "rep_index": rep_idx,
                "wavelength": wlen,
                "appearances": appearances,
                "max_depth": float(max_depth),
                "mean_equivalent_width_pixels": float(mean_ew),
                "mean_flux": mean_flux,
                "width_mean": float(np.mean(widths)),
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


# def _merge_peaks_across_sigmas(feature, wavelengths, peaks_by_sigma, tol_pixels=5, weight_original=2.0):
#     """
#     合并不同 scale 的峰/谷。
#     - 对 peaks：按 prominence 排序；
#     - 对 troughs：按 depth 和 EW 排序。
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

#         # 加权代表索引
#         weighted_sum, weight_total = 0.0, 0.0
#         max_sigma, min_sigma = 0.0, np.inf
#         for inf in infos:
#             sigma = inf["sigma"]
#             idx = inf["index"]
#             max_sigma = max(max_sigma, sigma)
#             min_sigma = min(min_sigma, sigma)
#             w = weight_original if sigma == 0 else 1.0 / np.sqrt(sigma)
#             weighted_sum += idx * w
#             weight_total += w
#         rep_idx = int(np.round(weighted_sum / weight_total))

#         wlen = float(wavelengths[rep_idx]) if rep_idx < len(wavelengths) else None
#         appearances = len(infos)
#         widths = [inf.get("width_wavelength", 0.0) for inf in infos]
#         mean_flux = float(np.mean([inf["flux"] for inf in infos]))
#         scales = list({inf["sigma"] for inf in infos})

#         if feature == "peak":
#             max_prom = max(inf.get("prominence", 0.0) for inf in infos)
#             consensus.append({
#                 "rep_index": rep_idx,
#                 "wavelength": wlen,
#                 "appearances": appearances,
#                 "max_prominence": float(max_prom),
#                 "mean_flux": mean_flux,
#                 "width_mean": float(np.mean(widths)),
#                 "seen_in_scales_of_sigma": scales,
#                 "max_sigma_seen": max_sigma,
#             })
#         else:
#             max_depth = max(inf.get("depth", 0.0) for inf in infos)
#             mean_ew = np.mean([inf.get("equivalent_width_pixels", 0.0) for inf in infos])
#             consensus.append({
#                 "rep_index": rep_idx,
#                 "wavelength": wlen,
#                 "appearances": appearances,
#                 "max_depth": float(max_depth),
#                 "mean_equivalent_width_pixels": float(mean_ew),
#                 "mean_flux": mean_flux,
#                 "width_mean": float(np.mean(widths)),
#                 "seen_in_scales_of_sigma": scales,
#                 "min_sigma_seen": min_sigma,
#             })

#     # 排序
#     if feature == "peak":
#         consensus = sorted(consensus,
#             key=lambda x: (x["max_sigma_seen"], x["max_prominence"], x["appearances"]),
#             reverse=True)
#     else:
#         consensus = sorted(consensus,
#             key=lambda x: (x["max_depth"], x["mean_equivalent_width_pixels"], x["appearances"]),
#             reverse=True)
#     return consensus


def _find_features_multiscale(
    state, feature="peak", sigma_list=None,
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
        spec = state["spectrum"]
        wavelengths = np.array(spec["new_wavelength"])
        flux = np.array(spec["weighted_flux"])

        # continuum 估计（仅 trough）
        continuum = None
        if feature == "trough" and use_continuum_for_trough:
            cont_window = max(51, int(3 * max(sigma_list)))
            continuum = median_filter(flux, size=cont_window, mode="reflect")
            continuum = np.where(continuum == 0, 1.0, continuum)

        sigma_list = [0] + sigma_list  # 原始光谱权重最高

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



# def _detect_features_on_flux(feature, flux, x_axis_slope, sigma, prominence=None, height=None):
#     """平滑后检测峰，返回峰索引及属性"""

#     if feature == "trough":
#         flux = -flux

#     if sigma > 0:
#         flux_smooth = gaussian_filter1d(flux, sigma=sigma)
#     else:
#         flux_smooth = flux.copy()

#     peaks, props = find_peaks(flux_smooth, height=height, prominence=prominence, distance=None)
#     widths_res = peak_widths(flux_smooth, peaks, rel_height=0.5)

#     peaks_info = []
#     for i, p in enumerate(peaks):
#         info = {
#             "index": int(p),
#             "wavelength": float(flux_smooth[p]),
#             "flux": float(flux_smooth[p]),
#             "prominence": float(props["prominences"][i]) if "prominences" in props else None,
#             "width_wavelength": float(x_axis_slope * widths_res[0][i])
#         }
#         peaks_info.append(info)
#     return flux_smooth, peaks_info


# def _merge_peaks_across_sigmas(feature, wavelengths, peaks_by_sigma, tol_pixels=5, weight_original=2.0):
#     """
#     合并不同scale的峰，原始光谱权重最高。
#     peaks_by_sigma: list of dicts [{"sigma":..., "peaks": [...]}]
#     返回 consensus 列表
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

#         # 计算加权平均 index
#         weighted_sum = 0.0
#         weight_total = 0.0
#         max_sigma = 0.0  # 最大平滑度
#         min_sigma = np.inf  # 最小平滑度
#         for inf in infos:
#             sigma = inf["sigma"]
#             idx = inf["index"]
#             max_sigma = max(max_sigma, sigma)
#             min_sigma = min(min_sigma, sigma)
#             if sigma == 0:  # 原始光谱，权重大
#                 w = weight_original
#             else:
#                 w = 1.0 / np.sqrt(sigma)
#             weighted_sum += idx * w
#             weight_total += w
#         rep_idx = int(np.round(weighted_sum / weight_total))

#         wlen = float(wavelengths[rep_idx]) if rep_idx < len(wavelengths) else None
#         appearances = len(infos)
#         max_prom = max([inf.get("prominence") or 0.0 for inf in infos])
#         mean_flux = float(np.mean([inf["flux"] for inf in infos]))
#         widths = [inf.get("width_wavelength") or 0.0 for inf in infos]
#         scales = list({inf["sigma"] for inf in infos})

#         if feature == "peak":
#             consensus.append({
#                 "rep_index": rep_idx,
#                 "wavelength": wlen,
#                 "appearances": appearances,
#                 "max_prominence": float(max_prom),
#                 "mean_flux": mean_flux,
#                 "width_mean": float(np.mean(widths)),
#                 "seen_in_scales_of_sigma": scales,
#                 "max_sigma_seen": max_sigma,  # 最大平滑度
#                 # "details": infos
#             })
#         else:  # trough
#             consensus.append({
#                 "rep_index": rep_idx,
#                 "wavelength": wlen,
#                 "appearances": appearances,
#                 "max_prominence": float(max_prom),
#                 "mean_flux": mean_flux,
#                 "width_mean": float(np.mean(widths)),
#                 "seen_in_scales_of_sigma": scales,
#                 "min_sigma_seen": min_sigma,  # 最大平滑度
#                 # "details": infos
#             })

#     # 排序：先按 max_sigma（平滑度）降序，再按 appearances 降序，再按 max_prominence 降序
#     if feature == "peak":
#         consensus = sorted(
#             consensus,
#             key=lambda x: (x["max_sigma_seen"], x["max_prominence"], x["appearances"]),
#             reverse=True
#         )
#     elif feature == "trough":
#         consensus = sorted(
#             consensus,
#             key=lambda x: (x["min_sigma_seen"], x["max_prominence"], x["appearances"]),
#             reverse=True
#         )
#     return consensus


# def _find_features_multiscale(
#         state, feature: str = "peak", sigma_list: str = [2,4,16], 
#         prom: float = 0.01, tol_pixels: int = 3, 
#         weight_original = 1.0
#         ) -> str:
#     """
#     Multiscale peak finder.
#     - sigma_list: JSON list of sigma values (in pixels), e.g. "[2,4,16]"
#     - prom: prominence threshold passed to find_peaks
#     - tol_pixels: tolerance in pixels for merging peaks across scales
#     - distance: minimal distance in points between peaks (optional)
#     - feature: "peak" or "trough"
#     """
#     try:
#         x_axis_slope = state['pixel_to_value']['x']['a']
#         spec = state['spectrum']
#         wavelengths = np.array(spec["new_wavelength"])
#         flux = np.array(spec["weighted_flux"])

#         if feature == "peak":
#             sigma_list = [0] + sigma_list  # 加入原始光谱

#         peaks_by_sigma = []
#         for s in sigma_list:
#             flux_smooth, peaks_info = _detect_features_on_flux(feature, flux, x_axis_slope, sigma=float(s), prominence=prom, height=None)

#             if feature == "trough":
#                 peaks_info = [
#                     t for t in peaks_info
#                     if (t["prominence"] > 0.02 and t["width_wavelength"] < 50)
#                 ]

#             peaks_by_sigma.append({
#                 "sigma": float(s),
#                 "flux_smooth": flux_smooth.tolist(),
#                 "peaks": peaks_info
#             })
        

#         consensus = _merge_peaks_across_sigmas(feature, wavelengths, peaks_by_sigma, tol_pixels=tol_pixels, weight_original=weight_original)

#         return consensus

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

def _plot_spectrum(state):
    # if effective_SNR:
    wavelength = state.spectrum['new_wavelength']
    flux = state.spectrum['weighted_flux']
    flux_top = state.spectrum['max_unresolved_flux']
    flux_bottom = state.spectrum['min_unresolved_flux']
    effective_snr = state.spectrum['effective_snr']
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
    fig.savefig(os.path.join(state.output_dir, f'{state.image_name}_spectrum.png'), bbox_inches='tight')

    # img = cv2.imread(state.crop_path)
    # plt.figure(figsize=(10,3))
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image')

    return fig

def _plot_features(state, sigma_list=[2,4,16], feature_number=[10,15]):
    fig = plt.figure(figsize=(10,3))

    plt.plot(state.spectrum['new_wavelength'], state.spectrum['weighted_flux'], label='original', c='k', alpha=0.7)

    for sigma in sigma_list:
        sigma_smooth = gaussian_filter1d(state.spectrum['weighted_flux'], sigma=sigma)
        plt.plot(state.spectrum['new_wavelength'], sigma_smooth, alpha=0.7, label=rf'$\sigma={sigma}$')

    # sigma_2 = gaussian_filter1d(state['spectrum']['weighted_flux'], sigma=2)
    # sigma_4 = gaussian_filter1d(state['spectrum']['weighted_flux'], sigma=4)
    # sigma_16 = gaussian_filter1d(state['spectrum']['weighted_flux'], sigma=16)
    
    # plt.plot(state['spectrum']['new_wavelength'], sigma_4, alpha=0.7, c='green', label=r'$\sigma=4$')
    # plt.plot(state['spectrum']['new_wavelength'], sigma_16, alpha=0.7, c='blue', label=r'$\sigma=16$')

    # 安全地绘制峰值线
    peaks_to_plot = min(feature_number[0], len(state.peaks))
    for i in range(peaks_to_plot):
        plt.axvline(state.peaks[i]['wavelength'], linestyle='-', c='red', alpha=0.5)
    
    # 安全地绘制谷值线
    troughs_to_plot = min(feature_number[1], len(state['troughs']))
    for i in range(troughs_to_plot):
        plt.axvline(state.troughs[i]['wavelength'], linestyle=':', c='red', alpha=0.5)

    plt.plot([], [], linestyle='-', c='red', alpha=0.5, label='peaks')
    plt.plot([], [], linestyle=':', c='blue', alpha=0.5, label='troughs')  # 注意：图例颜色与实际线条颜色的一致性
    plt.ylabel('flux')
    plt.xlabel('wavelength')
    plt.legend()

    print(f'Plot {peaks_to_plot} peaks and {troughs_to_plot} troughs.')

    # savefig_unique(fig, os.path.join(state['output_dir'], f'{state['image_name']}_features.png'))
    fig.savefig( os.path.join(state.output_dir, f'{state.image_name}_features.png'), bbox_inches='tight')
    return fig  # 建议返回 fig 对象