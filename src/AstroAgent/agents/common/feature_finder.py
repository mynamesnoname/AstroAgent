import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_widths
from copy import deepcopy

def _detect_features_on_flux(
    feature, flux, sigma, prominence=None, height=None,
    wavelengths=None, continuum=None
):
    """
    平滑后检测峰/谷，返回平滑光谱和峰信息。
    不再需要 x_axis_slope 参数，直接通过 wavelengths 数组计算波长单位的宽度。
    """
    if feature == "trough":
        if continuum is not None:
            flux_proc = flux / continuum
            flux_proc = 1.0 - flux_proc
        else:
            flux_proc = -flux.copy()
    else:
        flux_proc = flux - continuum if continuum is not None else flux.copy()

    flux_smooth = gaussian_filter1d(flux_proc, sigma=sigma) if sigma > 0 else flux_proc
    peaks, props = find_peaks(flux_smooth, height=height, prominence=prominence)
    widths_res = peak_widths(flux_smooth, peaks, rel_height=0.5)

    peaks_info = []
    for i, p in enumerate(peaks):
        width_pix = widths_res[0][i]
        left_ip = widths_res[2][i]   # 左边界插值索引
        right_ip = widths_res[3][i]  # 右边界插值索引
        prom = props.get("prominences")
        
        # 计算波长单位的宽度
        if wavelengths is not None:
            # 直接用波长数组计算宽度，支持非均匀采样
            left_wl = np.interp(left_ip, np.arange(len(wavelengths)), wavelengths)
            right_wl = np.interp(right_ip, np.arange(len(wavelengths)), wavelengths)
            width_wavelength = float(right_wl - left_wl)
        else:
            width_wavelength = None
        
        info = {
            "index": int(p),
            "wavelength": float(wavelengths[p]) if wavelengths is not None else None,
            "flux": float(flux[p]),
            "prominence": float(prom[i]) if prom is not None else None,
            "width_wavelength": width_wavelength,
        }
        if feature == "trough":
            prom = props.get("prominences")
            # depth = float(flux_smooth[p])
            depth = 1.0 - flux[p] / continuum[p]
            ew_pix = depth * width_pix
            info.update({
                "depth": depth,
                "equivalent_width_pixels": ew_pix,
                "prominence": float(prom[i]) if prom is not None else None
            })
        peaks_info.append(info)
    return flux_smooth, peaks_info


def _find_features_multiscale(
    wavelengths, flux,
    state,
    feature="peak", sigma_list=None,
    prom=0.01,
    mask=None,
    source_label="global"
):
    """
    多尺度特征检测器，返回各 σ 下的峰信息，带 source_label 标记。
    不再依赖 state["pixel_to_value"] 中的 x_axis_slope。
    """
    if sigma_list is None:
        sigma_list = [2, 4, 16]

    try:
        wavelengths = np.array(wavelengths)
        flux = np.array(flux)

        if mask is not None:
            mask = np.array(mask)
            global_indices = np.where(mask)[0]
            wavelengths_masked = wavelengths[mask]
            flux_masked = flux[mask]
        else:
            global_indices = np.arange(len(wavelengths))
            wavelengths_masked = wavelengths
            flux_masked = flux

        continuum = None
        if feature == "trough":
            cont_window = max(51, int(3 * max(sigma_list)))
            continuum = median_filter(flux_masked, size=cont_window, mode="reflect")
            continuum = np.where(continuum == 0, 1.0, continuum)

        sigma_list = sorted(set([0] + sigma_list))

        all_peaks = []
        for s in sigma_list:
            flux_smooth, peaks_info = _detect_features_on_flux(
                feature, flux_masked, sigma=float(s),
                prominence=prom, height=None,
                wavelengths=wavelengths_masked, continuum=continuum
            )

            # 转回全局索引并打上 source_label sigma/appearances
            for p in peaks_info:
                local_idx = p["index"]
                p["index"] = int(global_indices[local_idx])
                p["wavelength"] = float(wavelengths[p["index"]])
                p[f"{source_label}_sigma"] = s
                p[f"{source_label}_appearances"] = 1  # 每 σ 内单独出现计 1
                p["source"] = source_label

            all_peaks.extend(peaks_info)
        return all_peaks

    except Exception as e:
        logging.error(f"Error in _find_features_multiscale: {e}")
        return []


def _merge_and_group_features(
    all_peaks, wavelengths, flux, tol_wavelength=100, feature_type="peak"
):
    """
    根据 tol_wavelength 分组，不合并 sigma，仅保留 group 内所有条目。
    并按代表性质排序 group：最大 global_sigma > 最大 local_sigma > 最大 prominence > 最大 flux。
    """
    if not all_peaks:
        return []

    # 按 rep_index 排序
    all_peaks.sort(key=lambda x: x["index"])

    groups = []
    current_group = [all_peaks[0]]

    for candidate in all_peaks[1:]:
        can_add = True
        for item in current_group:
            if abs(candidate["wavelength"] - item["wavelength"]) > tol_wavelength:
                can_add = False
                break
        if can_add:
            current_group.append(candidate)
        else:
            groups.append(current_group)
            current_group = [candidate]
    groups.append(current_group)

    # 计算代表性质字典，用于排序
    def get_representative_properties(group):
        max_global_sigma = max(item.get("global_sigma", -999) for item in group)
        max_local_sigma = max(item.get("local_sigma", -999) for item in group)

        if feature_type == "peak":
            max_prominence = max(item.get("prominence", -999) for item in group)
            max_flux = max(flux[item["index"]] for item in group)
            return {
                "max_global_sigma": max_global_sigma,
                "max_local_sigma": max_local_sigma,
                "max_prominence": max_prominence,
                "max_flux": max_flux
            }
        else:  # trough
            max_depth = max(item.get("depth", -999) for item in group)
            max_ew = max(item.get("equivalent_width_pixels", -999) for item in group)
            # mean_ew = np.mean([item.get("equivalent_width_pixels", 0) for item in group])
            global_appearances = sum(item.get("global_appearances", 0) for item in group)
            local_appearances = sum(item.get("local_appearances", 0) for item in group)
            return {
                # "max_global_sigma": max_global_sigma,
                # "max_local_sigma": max_local_sigma,
                "max_depth": max_depth,
                "max_equivalent_width_pixels": max_ew,
                "global_appearances": global_appearances,
                "local_appearances": local_appearances
            }

    # 按代表性质排序
    groups_with_props = [(group, get_representative_properties(group)) for group in groups]

    if feature_type == "peak":
        # peak 排序：global_sigma > local_sigma > prominence > flux
        groups_with_props.sort(
            key=lambda x: (
                x[1]["max_global_sigma"],
                x[1]["max_local_sigma"],
                x[1]["max_prominence"],
                x[1]["max_flux"]
            ),
            reverse=True
        )
    else:  # trough
        # trough 排序：max_depth > mean_ew > global_appearances > local_appearances
        groups_with_props.sort(
            key=lambda x: (
                x[1]["max_depth"],
                x[1]["max_equivalent_width_pixels"],
                x[1]["global_appearances"],
                x[1]["local_appearances"]
            ),
            reverse=True
        )

    return [group for group, _ in groups_with_props]


def _peak_trough_detection(
    wavelengths, flux, state,
    sigma_list, tol_wavelength,
    prom_peaks, prom_troughs,
    local_size=500
):
    """
    完整峰/谷检测流程，直接生成 group，保留 global/local sigma。
    """
    wavelengths = np.array(wavelengths)
    flux = np.array(flux)

    all_peak_candidates = []
    all_trough_candidates = []

    # === Global 检测 ===
    global_peaks = _find_features_multiscale(
        wavelengths, flux, state,
        feature="peak", sigma_list=sigma_list,
        prom=prom_peaks, source_label="global"
    )
    global_troughs = _find_features_multiscale(
        wavelengths, flux, state,
        feature="trough", sigma_list=sigma_list,
        prom=prom_troughs, source_label="global"
    )
    all_peak_candidates.extend(global_peaks)
    all_trough_candidates.extend(global_troughs)

    # === Local 检测 ===
    local_edges_1 = np.arange(wavelengths[0], wavelengths[-1], local_size)
    local_edges_2 = local_edges_1 + local_size / 2

    for local_edges in [local_edges_1, local_edges_2]:
        for i in range(len(local_edges) - 1):
            local_start = local_edges[i]
            local_end = local_edges[i + 1]
            mask = (wavelengths >= local_start) & (wavelengths < local_end)
            if not np.any(mask):
                continue

            local_peaks = _find_features_multiscale(
                wavelengths, flux, state,
                feature="peak", sigma_list=sigma_list,
                prom=prom_peaks, mask=mask, source_label="local"
            )
            local_troughs = _find_features_multiscale(
                wavelengths, flux, state,
                feature="trough", sigma_list=sigma_list,
                prom=prom_troughs, mask=mask, source_label="local"
            )
            all_peak_candidates.extend(local_peaks)
            all_trough_candidates.extend(local_troughs)

    # === 分组，不合并 sigma ===
    peak_groups = _merge_and_group_features(
        all_peak_candidates, wavelengths, flux, tol_wavelength, "peak"
    )
    trough_groups = _merge_and_group_features(
        all_trough_candidates, wavelengths, flux, tol_wavelength, "trough"
    )

    return {
        "peak_groups": peak_groups,
        "trough_groups": trough_groups
    }
