import os
import json
import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
from paddleocr import PaddleOCR
from typing import List, Dict, Any
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mode

from AstroAgent.agents.common.state import SpectroState
from AstroAgent.agents.common.feature_finder import _peak_trough_detection
from AstroAgent.agents.common.feature_masker import generate_clean_flux_mask
from AstroAgent.agents.multi_agents.utils.usage import find_overlap_regions


# ===========================================================
# Step 1.2: OCR / Axis Tick Detection
# ===========================================================

def _detect_axis_ticks_tesseract(state: SpectroState, config=None):
    # Tesseract is not good. I prefer paddle.
    if config is None:
        # config = r'--oem 3 --psm 5 -c tessedit_char_whitelist=0123456789.-eE+ '
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.-eE+ '
        # config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.-eE+ '
    image_path = state['file_path']
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    # # _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

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
                    "bounding-box-scale": [w, h]
                })
            except ValueError:
                pass

    return tick_values


def _detect_axis_ticks_paddle(state: SpectroState):
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    result = ocr.predict(state['file_path'])
    for res in result:
        # res.print()
        res.save_to_img(state['output_dir'])
        res.save_to_json(state['output_dir'])
    data = []
    for i in range(len(result[-1]['rec_texts'])):
        pos = result[-1]['rec_polys'][i]
        center = [
            int((pos[0][0] + pos[2][0]) / 2),
            int((pos[0][1] + pos[2][1]) / 2),
        ]
        width = int((pos[1][0] - pos[0][0] + pos[2][0] - pos[3][0]) / 2)
        height = int((pos[3][1] - pos[0][1] + pos[2][1] - pos[1][1]) / 2)
        info = {
            'value': result[-1]['rec_texts'][i],
            'position': center,
            'bounding-box-scale': [width, height]
        }
        data.append(info)
    return data


# ===========================================================
# Step 1.5: Border Detection & Image Cropping
# ===========================================================

def _detect_chart_border(
        image_path: str,
        margin: Dict = {
            'top': 10,
            'right': 10,
            'bottom': 10,
            'left': 10
        }
) -> dict:
    """
    检测图像中图表的外围边框，并微调尺寸。

    参数:
        image_path: 图像文件路径
        margin: 调整边框的像素量（正数表示收缩边框）

    返回:
        dict 包含边框位置: {"x": int, "y": int, "w": int, "h": int}
    """
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
    x += margin['left']
    y += margin['top']
    w -= (margin['left'] + margin['right'])
    h -= (margin['top'] + margin['bottom'])

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


# ===========================================================
# Step 1.6~1.7: Pixel Coordinate Remapping & Fitting
# ===========================================================

def _remap_to_cropped_canvas(old_info, chart_border):
    """
    将原图坐标映射到裁剪后的画布上，自动处理 None 坐标。
    """
    x0, y0, w, h = chart_border["x"], chart_border["y"], chart_border["w"], chart_border["h"]

    new_info = []
    for d in old_info:
        ox = d.get("position_x") if d.get("position_x") is not None else None
        oy = d.get("position_y") if d.get("position_y") is not None else None

        new_d = d.copy()

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


def _linear_func(x, a, b):
    return a * x + b


def _pixel_tickvalue_fitting(arr: list) -> dict:
    """
    对刻度数据做加权线性拟合（支持 x/y 轴分开）。
    输入: Python list，每个元素为 dict
    输出: dict 包含各轴拟合结果
    """
    results = {}
    for axis in ["x", "y"]:
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
            _linear_func,
            pixels,
            values,
            sigma=sigma_eff,
            absolute_sigma=True
        )
        a_fit, b_fit = popt
        value_fit = _linear_func(pixels, a_fit, b_fit)
        residual = values - value_fit
        rms = np.sqrt(np.mean(residual**2))

        results[axis] = {
            "a": float(a_fit),
            "b": float(b_fit),
            "rms": float(rms),
            "residuals": residual.tolist()
        }
        print(f"{axis}: {results[axis]}")

    return results


# ===========================================================
# Step 1.8: Curve Extraction & Spectrum Reconstruction
# ===========================================================

def _process_and_extract_curve_points(input_path: str):
    """
    读取图像，去除背景并转换为二值图像，提取曲线的像素点云。

    参数：
    - input_path：原始图像文件路径

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

    return curve_points, curve_gray_values


def average_flux_by_wavelength(wavelength, flux):
    """
    对同一波长的flux进行简单平均。

    参数：
    - wavelength: 一维数组，表示波长
    - flux: 一维数组，表示光谱强度（flux）

    返回：
    - unique_wavelength: 每个唯一波长的数组
    - mean_flux: 每个波长对应的平均flux值
    """
    df = pd.DataFrame({
        'wavelength': wavelength,
        'flux': flux
    })

    # 对每个唯一的波长进行简单平均
    mean_flux = df.groupby('wavelength', group_keys=False)['flux'].mean()

    unique_wavelength = mean_flux.index.to_numpy()
    return unique_wavelength, mean_flux.to_numpy()


def _convert_to_spectrum(crop_path, axis_fitting_info):
    """
    从裁剪后的图像中提取曲线并转换为波长（wavelength）和光谱强度（flux）。

    输入：
    - crop_path: 裁剪后的图像路径
    - axis_fitting_info: 包含 x/y 轴拟合系数的字典

    输出：
    - spectrum_dict: 包含转换后的波长、flux 和平均后的波长与flux的字典
    """
    # Step 1: 提取曲线像素点
    points, gray = _process_and_extract_curve_points(crop_path)

    # Step 2: 转换坐标到物理量
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

    # 计算每个波长的平均flux
    unique_wavelength, mean_flux = average_flux_by_wavelength(wavelength, flux)

    max_unresolved_flux = []
    min_unresolved_flux = []
    std_flux = []  # 存储每个波长的标准差

    # 计算每个波长的统计量
    for w in unique_wavelength:
        unresolved_flux = flux[wavelength == w]
        max_unresolved_flux.append(np.max(unresolved_flux))
        min_unresolved_flux.append(np.min(unresolved_flux))

        # 计算标准差
        if len(unresolved_flux) > 1:
            # 使用无偏估计（除以 n-1）计算样本标准差
            std_val = np.std(unresolved_flux, ddof=1)
        else:
            # 如果只有一个数据点，标准差设为 0
            std_val = 0.0
        std_flux.append(std_val)

    # 计算 delta_flux
    delta_flux = np.array(max_unresolved_flux) - np.array(min_unresolved_flux)

    # 计算信噪比 SNR = mean / std
    snr = np.where(
        np.array(std_flux) != 0,
        mean_flux / np.array(std_flux),
        np.inf
    )

    snr_medium = np.median(snr)
    # result = mode(snr)
    # print("SNR中位数:", snr_medium)
    # print("SNR众数:", result.mode)

    spectrum_dict = {
        'flux': flux.tolist(),
        'wavelength': wavelength.tolist(),
        'new_wavelength': unique_wavelength.tolist(),
        'weighted_flux': mean_flux.tolist(),
        'max_unresolved_flux': max_unresolved_flux,
        'min_unresolved_flux': min_unresolved_flux,
        'delta_flux': delta_flux.tolist(),
        'std_flux': std_flux,
        'effective_snr': snr.tolist(),
        'snr_medium': snr_medium
    }

    return spectrum_dict


# ===========================================================
# Step 1.11: Group Features for LLM Line Center Selection
# ===========================================================


def group_features_for_llm(group: List[Dict[str, Any]], max_candidates: int = 20) -> Dict[str, Any]:
    """
    将 peak / trough group 整理为适合 LLM 判断的结构
    
    Args:
        group: 特征组列表
        max_candidates: 最多返回的候选条目数，默认20
    """

    candidates = {}

    # 判断是否包含 trough 信息
    has_depth = any("depth" in g for g in group)
    has_ew = any("equivalent_width_pixels" in g for g in group)

    for item in group:

        idx = item["index"]

        if idx not in candidates:
            candidates[idx] = {
                "index": idx,
                "wavelength": float(item["wavelength"]),
                "flux": float(item["flux"]),
                "evidence": {
                    "global": [],
                    "local": []
                }
            }

        source = item.get("source", "global")

        sigma = item.get("global_sigma") if source == "global" else item.get("local_sigma")

        ev = {
            "sigma": sigma,
            "prominence": float(item["prominence"]),
            "width": float(item["width_wavelength"]),
        }

        if has_depth and "depth" in item:
            ev['prominence'] = -ev['prominence']
            ev["depth"] = float(item["depth"])

        if has_ew and "equivalent_width_pixels" in item:
            ev["equivalent_width"] = float(item["equivalent_width_pixels"])

        candidates[idx]["evidence"][source].append(ev)

    candidates_list = list(candidates.values())

    # ---------- 智能筛选：如果超过 max_candidates，优先保留重要特征 ----------
    if len(candidates_list) > max_candidates:
        # 为每个 candidate 计算优先级分数
        # 优先级：大 sigma > 高 prominence
        def get_priority_score(c):
            all_ev = c["evidence"]["global"] + c["evidence"]["local"]
            if not all_ev:
                return (0, 0)
            # 最大 sigma（优先保留大 sigma）
            max_sigma = max(e.get("sigma", 0) or 0 for e in all_ev)
            # 最大 prominence（在 sigma 相同时，优先保留高 prominence）
            max_prom = max(abs(e.get("prominence", 0) or 0) for e in all_ev)
            # 是否有 global 证据（优先保留有 global 的）
            has_global = len(c["evidence"]["global"]) > 0
            # 返回排序键：(是否有global, 最大sigma, 最大prominence)
            return (has_global, max_sigma, max_prom)
        
        # 按优先级降序排序
        candidates_list.sort(key=get_priority_score, reverse=True)
        # 截取前 max_candidates 个
        candidates_list = candidates_list[:max_candidates]

    # ---------- group center ----------
    wavelengths = [c["wavelength"] for c in candidates_list]
    group_center = float(np.mean(wavelengths))

    # ---------- summary ----------
    for c in candidates_list:

        global_ev = c["evidence"]["global"]
        local_ev = c["evidence"]["local"]

        all_ev = global_ev + local_ev

        prominences = [e["prominence"] for e in all_ev]
        widths = [e["width"] for e in all_ev]

        summary = {
            "n_global": len(global_ev),
            "n_local": len(local_ev),
            "max_prominence": float(np.max(prominences)) if prominences else None,
            "mean_width": float(np.mean(widths)) if widths else None
        }

        if has_depth:
            depths = [e.get("depth") for e in all_ev if "depth" in e]
            if depths:
                summary["max_depth"] = float(np.max(depths))

        if has_ew:
            ews = [e.get("equivalent_width") for e in all_ev if "equivalent_width" in e]
            if ews:
                summary["max_equivalent_width"] = float(np.max(ews))

        c["summary"] = summary

        c["distance_to_group_center"] = abs(c["wavelength"] - group_center)

    # 按波长排序
    candidates_list.sort(key=lambda x: x["wavelength"])

    return {
        "group_center": group_center,
        "candidates": candidates_list
    }


# ===========================================================
# Step 1.9 / 1.11: Peak & Trough Detection Wrapper
# ===========================================================

def run_peak_trough_detection(state, wavelengths, flux, sigma_list, tol_wavelength, prom_peaks, prom_troughs):
    """
    调用 _peak_trough_detection 并更新 state 中的 peak_groups / trough_groups。
    """
    result = _peak_trough_detection(
        wavelengths, flux, state,
        sigma_list=sigma_list,
        tol_wavelength=tol_wavelength,
        prom_peaks=prom_peaks,
        prom_troughs=prom_troughs,
        local_size=500
    )
    state["peak_groups"] = result["peak_groups"]
    state["trough_groups"] = result["trough_groups"]
    return state


# ===========================================================
# Step 1.10a: Mask Peaks & Troughs
# ===========================================================

def run_mask_peaks_and_troughs(state, num_peaks, num_troughs):
    """
    根据 peak/trough groups 对光谱进行 mask，只使用前 num_peaks 个 peak groups
    和前 num_troughs 个 trough groups。
    """
    spec = state.get("spectrum", {})
    wavelengths = spec.get("new_wavelength", [])
    flux = spec.get("weighted_flux", [])

    peak_groups = state.get("peak_groups", [])[:num_peaks]
    trough_groups = state.get("trough_groups", [])[:num_troughs]

    if not wavelengths or not flux:
        logging.warning("No spectrum data available for masking")
        return state

    result = generate_clean_flux_mask(
        wavelengths=wavelengths,
        flux=flux,
        peak_groups=peak_groups,
        trough_groups=trough_groups,
        extend_ratio=0.6,
        min_width=200
    )

    state["cleaned_spectrum"] = {
        "wavelength": result["cleaned_wavelength"],
        "flux": result["cleaned_flux"]
    }
    state["masked_regions"] = result["masked_regions"]

    logging.info(
        f"Masked {len(peak_groups)} peak groups and "
        f"{len(trough_groups)} trough groups"
    )
    return state


# ===========================================================
# Step 1.10b: Continuum Fitting
# ===========================================================

def run_continuum_fitting(state, arm_name, arm_wavelength_range, sigma_continuum):
    """
    简单的 continuum 拟合：高斯平滑 + 残差光谱计算。
    """
    spec = state['cleaned_spectrum']
    wavelengths = np.array(spec['wavelength'])
    flux = np.array(spec['flux'])

    if arm_name:
        overlap_regions = find_overlap_regions(arm_name, arm_wavelength_range)
        arm_mask = np.zeros_like(wavelengths, dtype=bool)
        for key in overlap_regions:
            low, high = overlap_regions[key]
            region_mask = (wavelengths >= low) & (wavelengths <= high)
            arm_mask = arm_mask | region_mask
        # 使用线性插值填充 arm overlap 区域
        if np.any(arm_mask):
            unmasked_indices = np.where(~arm_mask)[0]
            if len(unmasked_indices) > 1:
                flux[arm_mask] = np.interp(
                    wavelengths[arm_mask],
                    wavelengths[unmasked_indices],
                    flux[unmasked_indices]
                )

    if not sigma_continuum:
        logging.error("CONTINUUM_SMOOTHING_SIGMA is not set, using 100")
        sigma_continuum = 100

    print(f'CONTINUUM_SMOOTHING_SIGMA: {sigma_continuum}')

    continuum_flux = gaussian_filter1d(flux, sigma=sigma_continuum)
    state['continuum'] = {
        'wavelength': wavelengths.tolist(),
        'flux': continuum_flux.tolist()
    }

    residual_flux = np.array(state['spectrum']["weighted_flux"]) - np.array(state['continuum']["flux"])
    state["residual_spectrum"] = {
        "wavelength": wavelengths.tolist(),
        "flux": residual_flux.tolist()
    }
    return state


# ===========================================================
# File I/O Helpers
# ===========================================================

def save_peak_trough_groups(state):
    """将 peak/trough groups 保存到文本文件。"""
    peak_path = os.path.join(state["output_dir"], f"{state['file_name']}_peak_g.txt")
    trough_path = os.path.join(state["output_dir"], f"{state['file_name']}_trough_g.txt")
    with open(peak_path, "w") as f:
        f.write("Peak Groups:\n")
        for i in state['peak_groups']:
            f.write("---------------\n")
            for j in i:
                f.write(f"{j}\n")
    with open(trough_path, "w") as f:
        f.write("Trough Groups:\n")
        for i in state['trough_groups']:
            f.write("---------------\n")
            for j in i:
                f.write(f"{j}\n")


def save_resolved_features(state, resolved_peaks, resolved_troughs):
    """将 resolved peaks/troughs 保存到 JSON 文件。"""
    with open(os.path.join(state['output_dir'], f"{state['file_name']}_resolved_peaks.txt"), "w") as f:
        json.dump(resolved_peaks, f, ensure_ascii=False, indent=4)
    with open(os.path.join(state['output_dir'], f"{state['file_name']}_resolved_troughs.txt"), "w") as f:
        json.dump(resolved_troughs, f, ensure_ascii=False, indent=4)

def save_cleaned_features(state, cleaned_peaks, cleaned_troughs):
    with open(os.path.join(state['output_dir'], f"{state['file_name']}_cleaned_peaks.txt"), "w") as f:
        json.dump(cleaned_peaks, f, ensure_ascii=False, indent=4)
    with open(os.path.join(state['output_dir'], f"{state['file_name']}_cleaned_troughs.txt"), "w") as f:
        json.dump(cleaned_troughs, f, ensure_ascii=False, indent=4)
