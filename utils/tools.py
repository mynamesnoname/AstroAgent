import cv2
import pytesseract
import json
import numpy as np
import pandas as pd

from typing import Any, Dict, Union
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

import json
from typing import Any, List, Dict, Tuple
from langchain.tools import tool

# ---------------------------
# 本地变量字典
# ---------------------------
LOCAL_VARS = {}

# ---------------------------
# 存储变量
# ---------------------------
import json

def _set_var(name: str, value: Any):
    """
    将任意Python对象存储到本地变量字典。如果输入为 JSON 格式，会自动解析为 Python 对象。
    """
    # 如果 value 是一个 JSON 字符串，尝试解析它
    if isinstance(value, str):
        try:
            # 尝试将字符串解析为 JSON
            value = json.loads(value)
        except json.JSONDecodeError:
            # 如果解析失败，则保留原始字符串
            pass

    LOCAL_VARS[name] = value
    return f"Variable '{name}' stored."

# @tool("set_var", return_direct=True)
# def set_var(name: str, value: Any):
#     """
#     将任意Python对象存储到本地变量字典。如果输入为 JSON 格式，会自动解析为 Python 对象。
#     """
#     # 如果 value 是一个 JSON 字符串，尝试解析它
#     if isinstance(value, str):
#         try:
#             # 尝试将字符串解析为 JSON
#             value = json.loads(value)
#         except json.JSONDecodeError:
#             # 如果解析失败，则保留原始字符串
#             pass

#     LOCAL_VARS[name] = value
#     return f"Variable '{name}' stored."

# ---------------------------
# 读取变量（返回 JSON）
# ---------------------------
def _get_var(name: str):
    """
    从本地变量字典中读取变量，并返回 JSON 字符串
    """
    if name not in LOCAL_VARS:
        return json.dumps({"error": f"Variable '{name}' not found"})
    return json.dumps(LOCAL_VARS[name], ensure_ascii=False)

# @tool("get_var", return_direct=True)
# def get_var(name: str):
#     """
#     从本地变量字典中读取变量，并返回 JSON 字符串
#     """
#     if name not in LOCAL_VARS:
#         return json.dumps({"error": f"Variable '{name}' not found"})
#     return json.dumps(LOCAL_VARS[name], ensure_ascii=False)


#################################################################################


def _detect_axis_ticks(image_path, config=None):
    if config is None:
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.-eE'

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

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
                    "pixel": [cx, cy],
                    "bounding-box-scale": [w,h]
                })
            except ValueError:
                pass
    
    _set_var("OCR_detected_ticks", tick_values)
    return tick_values


# @tool("detect_axis_ticks", return_direct=True)
# def detect_axis_ticks(image_path: str) -> list:
#     """
#     识别坐标轴刻度值及其像素位置。
#     输入: 图像路径字符串
#     输出字典，形如: {"value": value, "pixel": [cx, cy], "bounding-box-scale": [w,h]}
#     """
#     tick_values = _detect_axis_ticks(image_path)
#     return json.dumps(tick_values, ensure_ascii=False)


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
    
    _set_var("chart_border", {"x": x, "y": y, "w": w, "h": h})
    return {"x": x, "y": y, "w": w, "h": h}


@tool("detect_chart_border", return_direct=True)
def detect_chart_border(image_path: str) -> dict:
    """
    LangChain 工具：检测图表边框
    输入：图片路径
    输出：字典{"x":x, "y":y, "w":w, "h":h}，其中x和y分别为矩形边框左上角的像素坐标。w和h分别为边框的宽度和高度。
    """
    return _detect_chart_border(image_path)


# @tool("crop_img", return_direct=True)
def crop_img(image_path: str, x: int, y: int, w: int, h: int, save_path: str) -> str:
    """
    裁剪图像指定区域并保存。
    
    参数:
        image_path: 输入图像路径
        x, y: 边框的左上角坐标
        w, h: 边框的宽高
        save_path: 保存裁剪后图像路径
    
    返回:
        save_path
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    img_cropped = img[y:y+h, x:x+w]
    cv2.imwrite(save_path, img_cropped)
    
    return save_path


def _remap_to_cropped_canvas(old_info, chart_border):
    """
    将原图坐标映射到裁剪后的画布上，自动处理 None 坐标。
    """
    x0, y0, w, h = chart_border["x"], chart_border["y"], chart_border["w"], chart_border["h"]

    new_info = []
    for d in old_info:
        ox, oy = d.get("position_x"), d.get("position_y")

        new_d = d.copy()

        if ox is None or oy is None:
            # 如果原始坐标缺失，保持原值或设为 None
            new_d["position_x"] = ox
            new_d["position_y"] = oy
        else:
            # 重映射到裁剪画布
            nx = ox - x0
            ny = oy - y0
            # 越界裁剪
            nx = max(0, min(nx, w - 1))
            ny = max(0, min(ny, h - 1))
            new_d["position_x"] = nx
            new_d["position_y"] = ny

        new_info.append(new_d)

    return new_info


@tool("remap_to_cropped_canvas", return_direct=True)
def remap_to_cropped_canvas(info_json: str, chart_border_json: str) -> str:
    """
    LangChain 工具：将原坐标信息映射到裁剪后的画布。

    参数:
        info_json (str): JSON 字符串，表示原始坐标信息列表。例如：
            [
              {"axis": 0, "value": -2.0, "position_x": 39, "position_y": 313,
               "bounding-box-scale_x": 17, "bounding-box-scale_y": 10,
               "sigma_pixel": 5, "conf_llm": 0.8}
            ]
        chart_border_json (str): JSON 字符串，裁剪框信息。例如：
            {"x": 65, "y": 23, "w": 916, "h": 294}

    返回:
        str: JSON 字符串，重映射后的坐标信息列表。
    """
    old_info = json.loads(info_json)
    chart_border = json.loads(chart_border_json)

    new_info = _remap_to_cropped_canvas(old_info, chart_border)

    # 保存全局变量，便于后续调用
    _set_var("remapped_info", new_info)

    return json.dumps(new_info, ensure_ascii=False, indent=2)


def linear_func(x, a, b):
    return a * x + b

def _pixel_tickvalue_fitting(arr: list) -> dict:
    """
    对刻度数据做加权线性拟合（支持 x/y 轴分开）。
    输入: Python list，每个元素为 dict
    输出: dict 包含各轴拟合结果
    """
    results = {}
    for axis in [0, 1]:
        # 提取有效数据
        values, pixels, sigmas, confs = [], [], [], []
        for d in arr:
            if d["axis"] == axis and d["position_x"] is not None and d["position_y"] is not None:
                values.append(float(d["value"]))
                pixels.append(float(d["position_x"] if axis == 1 else d["position_y"]))
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

        axis_ = "y" if axis == 0 else "x"
        results[axis_] = {
            "a": float(a_fit),
            "b": float(b_fit),
            "rms": float(rms),
            "residuals": residual.tolist()
        }
        _set_var(f"{axis_}", results[axis_])

    return results

@tool("pixel_tickvalue_fitting", return_direct=True)
def pixel_tickvalue_fitting(data: str) -> str:
    """
    LangChain Tool 封装。
    输入: JSON 字符串
    输出: JSON 字符串
    """
    try:
        arr = json.loads(data)   # 转成 Python list
        results = _pixel_tickvalue_fitting(arr)
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _process_and_extract_curve_points(input_path: str):
    """
    读取图像，去除背景并转换为二值图像，提取曲线的像素点云，保存到LOCAL_VARS
    
    参数：
    - input_path：原始图像文件路径
    - output_path：处理后图像保存路径
    
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
    
    # 存储曲线点和灰度值到 LOCAL_VARS
    _set_var("curve_points", curve_points)
    _set_var('curve_gray_values', curve_gray_values.tolist())
    
    # 返回曲线点数量（简化信息）
    return json.dumps({
        "curve_points_count": len(curve_points),
        "curve_gray_values_count": len(curve_gray_values),
    })

@tool("process_and_extract_curve_points", return_direct=True)
def process_and_extract_curve_points(input_path: str) -> str:
    """
    输入：图像文件路径
    输出：简化的 JSON 字符串，包含图像路径和曲线点数量
    
    同时把提取出的曲线点云在图上用红色圆圈标记成示意图，保存到 output_path。
    """
    try:
        return _process_and_extract_curve_points(input_path)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


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
    # 使用 pandas 来简化操作
    df = pd.DataFrame({
        'wavelength': wavelength,
        'flux': flux,
        'gray': gray
    })

    # 对每个唯一的波长进行加权平均
    weighted_flux = df.groupby('wavelength').apply(
        lambda x: np.sum(x['flux'] * x['gray']) / np.sum(x['gray'])
    )

    # 获取唯一的波长
    unique_wavelength = weighted_flux.index.to_numpy()
    return unique_wavelength, weighted_flux.to_numpy()


def _convert_to_spectrum(points, gray, x_axis, y_axis):
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
    xs = points[:, 0]
    ys = points[:, 1]

    # 提取 x 轴和 y 轴的物理量转换系数
    a_y = y_axis['a']
    b_y = y_axis['b']
    flux = a_y * ys + b_y

    a_x = x_axis['a']
    b_x = x_axis['b']
    wavelength = a_x * xs + b_x

    # 计算加权平均flux
    unique_wavelength, weighted_flux = weighted_average_flux(wavelength, flux, gray)

    max_unresolved_flux = []
    min_unresolved_flux = []
    for i, w in enumerate(unique_wavelength):
        unresolved_flux = flux[wavelength == w]
        max_unresolved_flux.append(np.max(unresolved_flux))
        min_unresolved_flux.append(np.min(unresolved_flux))

    # 构造最终结果
    spectrum_dict = {
        'flux': flux.tolist(),
        'wavelength': wavelength.tolist(),
        'new_wavelength': unique_wavelength.tolist(),
        'weighted_flux': weighted_flux.tolist(),
        'max_unresolved_flux': max_unresolved_flux,
        'min_unresolved_flux': min_unresolved_flux
    }

    return spectrum_dict


# @tool("convert_to_spectrum", return_direct=True)
def convert_to_spectrum() -> str:
    """
    LangChain 工具：将曲线的像素坐标和灰度值转换为波长和光谱强度（flux）。
    并根据灰度值对同一波长的flux进行加权平均。
    
    输入：
    - curve_points: 曲线像素坐标信息（JSON 格式，形如 [(x1, y1), (x2, y2), ...]）
    - curve_gray_values: 灰度值（JSON 格式，形如 [gray1, gray2, ...]）
    - x_axis: x 轴转换系数（JSON 格式，形如 {"a": x_scale, "b": x_offset}）
    - y_axis: y 轴转换系数（JSON 格式，形如 {"a": y_scale, "b": y_offset}）

    输出：
    - 返回一个 JSON 字符串，包含波长（wavelength）、光谱强度（flux）和加权平均后的波长及flux
    """
    # curve_points = _get_var('curve_points')
    # curve_gray_values = _get_var('curve_gray_values')

    # x_axis_info = _get_var('x')
    # y_axis_info = _get_var('y')

    try:
        # 解析输入数据
        points = np.array(LOCAL_VARS['curve_points'])
        gray = np.array(LOCAL_VARS['curve_gray_values']).astype(np.float64)
        x_axis_dict = LOCAL_VARS['x']
        y_axis_dict = LOCAL_VARS['y']

        # 调用核心函数进行转换
        spectrum_dict = _convert_to_spectrum(points, gray, x_axis_dict, y_axis_dict)

        _set_var('spectrum', spectrum_dict)

        # 返回结果
        return json.dumps(spectrum_dict, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _detect_features_on_flux(feature, flux, sigma, height=None, prominence=None):
    """平滑后检测峰，返回峰索引及属性"""

    if feature == "trough":
        flux = -flux

    if sigma > 0:
        flux_smooth = gaussian_filter1d(flux, sigma=sigma)
    else:
        flux_smooth = flux.copy()

    peaks, props = find_peaks(flux_smooth, height=height, prominence=prominence, distance=None)
    widths_res = peak_widths(flux_smooth, peaks, rel_height=0.5)
    peaks_info = []
    for i, p in enumerate(peaks):
        info = {
            "index": int(p),
            "wavelength": float(flux_smooth[p]),
            "flux": float(flux_smooth[p]),
            "prominence": float(props["prominences"][i]) if "prominences" in props else None,
            "width_pixels": float(widths_res[0][i])
        }
        peaks_info.append(info)
    return flux_smooth, peaks_info


def _merge_peaks_across_sigmas(feature, wavelengths, peaks_by_sigma, tol_pixels=5, weight_original=1.0):
    """
    合并不同scale的峰，原始光谱权重最高。
    peaks_by_sigma: list of dicts [{"sigma":..., "peaks": [...]}]
    返回 consensus 列表
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

        # 计算加权平均 index
        weighted_sum = 0.0
        weight_total = 0.0
        max_sigma = 0.0  # 最大平滑度
        min_sigma = np.inf  # 最小平滑度
        for inf in infos:
            sigma = inf["sigma"]
            idx = inf["index"]
            max_sigma = max(max_sigma, sigma)
            min_sigma = min(min_sigma, sigma)
            if sigma == 0:  # 原始光谱，权重大
                w = weight_original
            else:
                w = 1.0 / np.sqrt(sigma)
            weighted_sum += idx * w
            weight_total += w
        rep_idx = int(np.round(weighted_sum / weight_total))

        wlen = float(wavelengths[rep_idx]) if rep_idx < len(wavelengths) else None
        appearances = len(infos)
        max_prom = max([inf.get("prominence") or 0.0 for inf in infos])
        mean_flux = float(np.mean([inf["flux"] for inf in infos]))
        widths = [inf.get("width_pixels") or 0.0 for inf in infos]
        scales = list({inf["sigma"] for inf in infos})

        if feature == "peak":
            consensus.append({
                "rep_index": rep_idx,
                "wavelength": wlen,
                "appearances": appearances,
                "max_prominence": float(max_prom),
                "mean_flux": mean_flux,
                "width_mean": float(np.mean(widths)),
                "scales_seen": scales,
                "max_sigma_seen": max_sigma,  # 最大平滑度
                "details": infos
            })
        else:  # trough
            consensus.append({
                "rep_index": rep_idx,
                "wavelength": wlen,
                "appearances": appearances,
                "max_prominence": float(max_prom),
                "mean_flux": mean_flux,
                "width_mean": float(np.mean(widths)),
                "scales_seen": scales,
                "min_sigma_seen": min_sigma,  # 最大平滑度
                "details": infos
            })

    # 排序：先按 max_sigma（平滑度）降序，再按 appearances 降序，再按 max_prominence 降序
    if feature == "peak":
        consensus = sorted(
            consensus,
            key=lambda x: (x["max_sigma_seen"], x["max_prominence"], x["appearances"]),
            reverse=True
        )
    elif feature == "trough":
        consensus = sorted(
            consensus,
            key=lambda x: (x["min_sigma_seen"], x["max_prominence"], x["appearances"]),
            reverse=True
        )
    return consensus


@tool("find_features_multiscale", return_direct=True)
def find_features_multiscale(feature: str = "peak", sigma_list: str = "[2,4,16]", prom: float = 0.01, tol_pixels: int = 3) -> str:
    """
    Multiscale peak finder.
    - sigma_list: JSON list of sigma values (in pixels), e.g. "[2,4,16]"
    - prom: prominence threshold passed to find_peaks
    - tol_pixels: tolerance in pixels for merging peaks across scales
    - distance: minimal distance in points between peaks (optional)
    - feature: "peak" or "trough"
    """
    try:
        # spec_js = _get_var("spectrum")
        # spec = json.loads(spec_js)
        spec = LOCAL_VARS["spectrum"]
        wavelengths = np.array(spec["new_wavelength"])
        flux = np.array(spec["weighted_flux"])

        sigma_list = json.loads(sigma_list) if isinstance(sigma_list, str) else list(sigma_list)
        if feature == "peak":
            sigma_list = [0] + sigma_list  # 加入原始光谱

        peaks_by_sigma = []
        for s in sigma_list:
            flux_smooth, peaks_info = _detect_features_on_flux(feature, flux, sigma=float(s), prominence=prom, height=None)

            if feature == "trough":
                peaks_info = [
                    t for t in peaks_info
                    if (t["prominence"] > 0.02 and t["width_pixels"] < 50)
                ]

            peaks_by_sigma.append({
                "sigma": float(s),
                "flux_smooth": flux_smooth.tolist(),
                "peaks": peaks_info
            })
            _set_var(f"{feature}_spectrum_smooth_sigma_{int(s)}", {
                "wavelength": wavelengths.tolist(),
                "flux_smooth": flux_smooth.tolist()
            })
        

        consensus = _merge_peaks_across_sigmas(feature, wavelengths, peaks_by_sigma, tol_pixels=tol_pixels)

        _set_var(f"{feature}s_multiscale", peaks_by_sigma)
        _set_var(f"{feature}s_consensus", consensus)

        return json.dumps({
            "status": "ok",
            "sigma_list": sigma_list,
            "peaks_counts": {str(int(p["sigma"])): len(p["peaks"]) for p in peaks_by_sigma},
            "consensus_count": len(consensus)
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    

def _calculate_redshift(obs_wavelength: Union[float, List[float]], rest_wavelength: Union[float, List[float]]) -> Union[float, List[float]]:
    """
    根据观测波长和谱线本征波长计算红移
    输入: float 或 list
    输出: float 或 list
    """
    # 单值
    if isinstance(obs_wavelength, (int, float)):
        return obs_wavelength / rest_wavelength - 1
    # 列表
    return [o / r - 1 for o, r in zip(obs_wavelength, rest_wavelength)]


# LangChain 工具封装，接收 JSON 字符串
@tool("calculate_redshift", return_direct=True)
def calculate_redshift(obs_wavelength: str, rest_wavelength: str) -> str:
    """
    LangChain 工具封装
    输入 JSON:
        "obs_wavelength": float 或 [float, ...],
        "rest_wavelength": float 或 [float, ...]

    输出 JSON:
        "redshift": float 或 [float, ...]
    """
    try:
        obs = json.loads(obs_wavelength)
        rest = json.loads(rest_wavelength)
        z = _calculate_redshift(obs, rest)
        return json.dumps({"redshift": z}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _predict_obs_wavelength(redshift: Union[float, List[float]], rest_wavelength: Union[float, List[float]]) -> Union[float, List[float]]:
    """
    根据红移和谱线本征波长计算观测波长
    输入: float 或 list
    输出: float 或 list
    """
    return rest_wavelength * (redshift + 1)


# LangChain 工具封装，接收 JSON 字符串
@tool("predict_obs_wavelength", return_direct=True)
def predict_obs_wavelength(redshift: str, rest_wavelength: str) -> str:
    """
    LangChain 工具封装
    输入 JSON:
        "rest_wavelength": float 或 [float, ...]
        "redshift": float
    输出 JSON:
        "redshift": float 或 [float, ...]
    """
    try:
        redshift = json.loads(redshift)
        rest_wavelength = json.loads(rest_wavelength)
        z = _predict_obs_wavelength(redshift, rest_wavelength)
        return json.dumps({"redshift": z}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
# # 扁平化的常见天文发射线字典（来自修正版本）
# emission_lines = {
#     "Ly ε": 937.80,
#     "Ly δ": 949.74,
#     "CIII 977": 977.02,
#     "NIII 991": 990.69,
#     "Ly β": 1025.72,
#     "OVI 1033": 1033.83,
#     "ArI 1066": 1066.66,
#     "FeIII UV1": 1175.35,
#     "CIII* 1175": 1175.70,
#     "Ly α": 1215.67,
#     "NV 1240": 1240.14,
#     "SiII 1263": 1262.59,
#     "OI 1304": 1304.35,
#     "SiII 1307": 1306.82,
#     "CII 1335": 1335.30,
#     "SiIV 1393": 1393.76,
#     "SiIV 1402": 1402.77,
#     "OIV] 1401": 1401.0,
#     "CIV 1549": 1549.06,
#     "HeII 1640": 1640.42,
#     "OIII] 1663": 1663.48,
#     "AlII 1671": 1670.79,
#     "NIV 1718": 1718.55,
#     "NIV] 1721": 1721.89,
#     "NIII] 1750": 1750.26,
#     "Uncertain_1814": 1814.73,
#     "Uncertain_1817": 1816.98,
#     "AlIII 1857": 1857.40,
#     "SiIII] 1892": 1892.03,
#     "CIII] 1909": 1908.73,
#     "[NeIV] 2424": 2423.83,
#     "[OII] 2471": 2471.03,
#     "AlII] 2670": 2669.95,
#     "OIII 2672?": 2672.04,
#     "MgII 2799": 2798.75,
#     "OIII 3134": 3133.70,
#     "HeI 3189": 3188.67,
#     "[NeV] 3346": 3346.82,
#     "[NeV] 3426": 3426.84,
#     "[FeVII] 3587": 3587.34,
#     "HeI 3588": 3588.30,
#     "[OII] 3727": 3728.48,
#     "[FeVII] 3760": 3759.99,
#     "[NeIII] 3869": 3869.85,
#     "HeI 3889": 3889.74,
#     "[NeIII] 3967": 3968.0,
#     "H ε": 3970.07,
#     "[FeV] 4072?": 4072.39,
#     "[SII] 4074?": 4073.63,
#     "H δ": 4102.89,
#     "H γ": 4341.68,
#     "[OIII] 4363": 4364.44,
#     "HeI 4472": 4472.76,
#     "HeII 4686": 4687.02,
#     "H β": 4862.68,
#     "[OIII] 4959": 4960.30,
#     "[OIII] 5007": 5008.24,
#     "[FeVII] 5160": 5160.33,
#     "[FeVI] 5177": 5177.48,
#     "[NI] 5200": 5200.53,
#     "[FeVII] 5278": 5277.85,
#     "[FeXIV] 5303": 5304.34,
#     "[CaV] 5311": 5310.59,
#     "[ClIII] 5539": 5539.43,
#     "[FeVII] 5721": 5722.30,
#     "HeI 5876": 5876.0,
#     "[FeVII] 6088": 6087.98,
#     "[OI] 6300": 6300.30,
#     "[OI] 6364": 6365.54,
#     "[FeX] 6375": 6376.30,
#     "[NII] 6549": 6549.85,
#     "H α": 6564.61,
#     "[NII] 6585": 6585.28,
#     "[SII] 6716": 6716.0,
#     "[SII] 6731": 6731.0,
#     "HeI 7067": 7067.20,
#     "[ArIII] 7138": 7137.80,
#     "[OII] 7320+7330": 7321.48,
#     "[NiIII] 7892?": 7892.10,
#     "[FeXI] 7894": 7894.00
# }

# # 常见天文发射线字典（精简版）
# emission_lines = {
#     "Ly β": 1025.72,
#     "Ly α": 1215.67,
#     "CIII 977": 977.02,
#     "NV 1240": 1240.14,
#     "SiII 1263": 1262.59,
#     "OI 1304": 1304.35,
#     "SiII 1307": 1306.82,
#     "CII 1335": 1335.30,
#     "SiIV 1393": 1393.76,
#     "SiIV 1402": 1402.77,
#     "OIV] 1401": 1401.0,
#     "CIV 1549": 1549.06,
#     "HeII 1640": 1640.42,
#     "OIII] 1663": 1663.48,
#     "AlII 1671": 1670.79,
#     "NIV 1718": 1718.55,
#     "NIII] 1750": 1750.26,
#     "AlIII 1857": 1857.40,
#     "SiIII] 1892": 1892.03,
#     "CIII] 1909": 1908.73,
#     "[OII] 2471": 2471.03,
#     "MgII 2799": 2798.75,
#     "OIII 3134": 3133.70,
#     "HeI 3189": 3188.67,
#     "[NeV] 3426": 3426.84,
#     "[NeIII] 3869": 3869.85,
#     "HeI 3889": 3889.74,
#     "H δ": 4102.89,
#     "H γ": 4341.68,
#     "[OIII] 4363": 4364.44,
#     "HeI 4472": 4472.76,
#     "HeII 4686": 4687.02,
#     "H β": 4862.68,
#     "[OIII] 4959": 4960.30,
#     "[OIII] 5007": 5008.24,
#     "[NI] 5200": 5200.53,
#     "HeI 5876": 5876.0,
#     "[OI] 6300": 6300.30,
#     "[OI] 6364": 6365.54,
#     "[NII] 6549": 6549.85,
#     "H α": 6564.61,
#     "[NII] 6585": 6585.28,
#     "[SII] 6716": 6716.0,
#     "[SII] 6731": 6731.0,
#     "HeI 7067": 7067.20,
#     "[ArIII] 7138": 7137.80,
#     "[OII] 7320+7330": 7321.48,
# }


# def _calculate_redshifts_for_peaks(observed_wavelengths, delta_lambda, line_dict=emission_lines, z_min=0.0, z_max=10.0):
#     """
#     根据发射线表，对每个观测波长逐一计算可能的红移，并只保留 z 在 [z_min, z_max] 范围内的结果。
#     """
#     results = []

#     for lam_obs in observed_wavelengths:
#         peak_info = {"lambda_obs": lam_obs, "redshifts": []}

#         for line_name, lambda_rest in line_dict.items():
#             z = lam_obs / lambda_rest - 1
#             z_left = (lam_obs - delta_lambda) / lambda_rest - 1
#             z_right = (lam_obs + delta_lambda) / lambda_rest - 1
#             if z_min <= z <= z_max:
#                 peak_info["redshifts"].append({
#                     "line": line_name,
#                     "lambda_rest": lambda_rest,
#                     "z": z,
#                     "z_range": [z_left, z_right]
#                 })

#         results.append(peak_info)

#     _set_var("redshift_results", results)
#     return results

# # LangChain 工具封装
# # @tool("calculate_redshifts_for_peaks", return_direct=True)
# def calculate_redshifts_for_peaks(observed_wavelengths_json: str, delta_lambda: float, z_min: float = 0.0, z_max: float = 10.0) -> str:
#     """
#     LangChain 工具封装，输入 JSON 字符串的观测波长列表，输出 JSON 字符串的红移字典。
#     支持 z_min 和 z_max 参数，只保留在该范围内的红移结果。
#     """
#     try:
#         observed_wavelengths = json.loads(observed_wavelengths_json)
#         res = _calculate_redshifts_for_peaks(observed_wavelengths, delta_lambda, z_min=z_min, z_max=z_max)
#         return json.dumps(res, ensure_ascii=False)
#     except Exception as e:
#         return json.dumps({"error": str(e)}, ensure_ascii=False)













# 把你的工具放这里
TOOLS = [
    pixel_tickvalue_fitting, 
    detect_chart_border, 
    remap_to_cropped_canvas, process_and_extract_curve_points, 
    convert_to_spectrum, find_features_multiscale,
    calculate_redshift, predict_obs_wavelength
]

# 构建工具映射
tool_map: Dict[str, Any] = {}
for t in TOOLS:
    name = getattr(t, "name", None) or getattr(t, "__name__", None)
    func = getattr(t, "func", None) or t
    tool_map[name] = func


def _normalize_args(fn, args: dict):
    """
    自动把 LLM 生成的 args 键名修正成函数签名里的参数名
    - 如果缺少 '_json'，会自动补全
    - 如果参数名大小写不一致，也会自动适配
    """
    import inspect
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    new_args = {}
    for p in params:
        if p in args:
            new_args[p] = args[p]
        elif p.replace("_json", "") in args:
            new_args[p] = args[p.replace("_json", "")]
        elif p.upper() in args:
            new_args[p] = args[p.upper()]
        elif p.lower() in args:
            new_args[p] = args[p.lower()]
        # 如果没找到，就不加，让 fn 抛出 TypeError
    return new_args


def _call_tool_dynamic(tool_call: Dict[str, Any]) -> str:
    """
    接收 tool_call dict（LLM 返回的结构），执行对应工具并返回字符串结果（JSON string 优先）。
    支持自动存储到 LOCAL_VARS，并可以通过 tool_call["store_as"] 指定变量名。
    """
    tool_name = tool_call["name"]
    args = tool_call.get("args", {})

    if tool_name not in tool_map:
        return json.dumps({"error": f"tool '{tool_name}' not found"}, ensure_ascii=False)

    fn = tool_map[tool_name]

    try:
        if isinstance(args, dict):
            # 尝试自动修正参数名
            args = _normalize_args(fn, args)
            result = fn(**args)
        elif isinstance(args, str):
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    parsed = _normalize_args(fn, parsed)
                    result = fn(**parsed)
                else:
                    result = fn(args)
            except json.JSONDecodeError:
                result = fn(args)
        else:
            result = fn(args)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    # 规范化返回值
    if not isinstance(result, str):
        try:
            result_str = json.dumps(result, ensure_ascii=False)
        except TypeError:
            result_str = str(result)
    else:
        result_str = result

    return result_str


def run_with_tools(llm_bound, messages):
    """
    运行 LLM 并自动处理工具调用。
    messages: list[Message]，第一个通常是 HumanMessage(...)
    """
    response = llm_bound.invoke(messages)

    while getattr(response, "tool_calls", None):
        tool_call = response.tool_calls[0]

        # 执行工具并存储结果
        tool_output = _call_tool_dynamic(tool_call)

        # 回写给 LLM
        tool_message = ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
        messages = messages[:6] + [response, tool_message]

        # 继续让 LLM 处理新的上下文
        response = llm_bound.invoke(messages)

    return response





# # 构建工具映射：支持普通函数或用 @tool 装饰的对象
# tool_map: Dict[str, Any] = {}
# for t in TOOLS:
#     # 如果是 LangChain 的 Tool-like 对象，可能有 .name / .func
#     name = getattr(t, "name", None) or getattr(t, "__name__", None)
#     func = getattr(t, "func", None) or t
#     tool_map[name] = func

# def _call_tool_dynamic(tool_call: Dict[str, Any]) -> str:
#     """
#     接收 tool_call dict（llm 返回的结构），执行对应工具并返回字符串结果（JSON string 优先）。
#     """
#     tool_name = tool_call["name"]
#     args = tool_call.get("args", {})

#     if tool_name not in tool_map:
#         return json.dumps({"error": f"tool '{tool_name}' not found"}, ensure_ascii=False)

#     fn = tool_map[tool_name]

#     # 解析 args：它可能是 dict，也可能是一个单字符串，或 JSON 字符串
#     try:
#         # case: args already dict
#         if isinstance(args, dict):
#             # 试图按名字调用（kwargs）优先
#             try:
#                 result = fn(**args)
#             except TypeError:
#                 # 若函数不接受 kwargs，则传入第一个 positional 参数（常见）
#                 # 取第一个值或整个 dict 字符串
#                 if len(args) == 1:
#                     single = list(args.values())[0]
#                     result = fn(single)
#                 else:
#                     result = fn(args)
#         elif isinstance(args, str):
#             # 如果是 JSON 字符串，尝试解析为 dict
#             try:
#                 parsed = json.loads(args)
#                 if isinstance(parsed, dict):
#                     try:
#                         result = fn(**parsed)
#                     except TypeError:
#                         # fallback: 传入整个 dict
#                         result = fn(parsed)
#                 else:
#                     # 不是 dict 就当作单字符串参数
#                     result = fn(args)
#             except json.JSONDecodeError:
#                 # 不是 JSON，直接传字符串
#                 result = fn(args)
#         else:
#             # 其它类型直接尝试传入
#             result = fn(args)
#     except Exception as e:
#         # 不要抛到 LLM，返回结构化错误
#         return json.dumps({"error": str(e)}, ensure_ascii=False)

#     # 规范化返回：优先 JSON 字符串；若函数返回 Python 对象，serialize
#     if isinstance(result, str):
#         return result
#     try:
#         return json.dumps(result, ensure_ascii=False)
#     except TypeError:
#         # 退化为字符串
#         return str(result)

# def run_with_tools(llm_bound, messages):
#     """
#     messages: list[Message]，第一个通常是 HumanMessage(...)
#     llm_bound.invoke 返回一个包含 tool_calls 的消息对象
#     """
#     response = llm_bound.invoke(messages)

#     # 循环直到没有待执行的 tool_calls
#     while getattr(response, "tool_calls", None):
#         tool_call = response.tool_calls[0]  # 取第一个待执行的工具调用
#         # 执行工具（动态分发）
#         tool_output = _call_tool_dynamic(tool_call)

#         # 把工具输出包装成 ToolMessage 回写给 LLM（注意 tool_call['id']）
#         tool_message = ToolMessage(content=tool_output, tool_call_id=tool_call["id"])

#         # 更新上下文：把 LLM 的上一条响应和工具返回放到消息序列里
#         messages = messages + [response, tool_message]

#         # 让 LLM 在新的上下文中继续推理（它可能会再发出新的工具调用，或给出最终回答）
#         response = llm_bound.invoke(messages)

#     return response

