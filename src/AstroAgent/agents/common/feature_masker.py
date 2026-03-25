import numpy as np
import logging
import os


def generate_clean_flux_mask(
    wavelengths,
    flux,
    peak_groups,
    trough_groups,
    extend_ratio=0.6,
    min_width=300
):
    """
    根据 peak/trough groups 生成 mask 并返回清洗后的光谱

    mask 规则：
    A. group span mask
        [group_min - extend_ratio * width , group_max + extend_ratio * width]

    B. feature mask
        [center - width/2 , center + width/2]

    最终 mask = A ∪ B

    参数
    -------
    wavelengths : list or array
    flux : list or array
    peak_groups : list
    trough_groups : list
    extend_ratio : float
    min_width : float
        feature width 下限

    返回
    -------
    dict
    """

    wavelengths = np.array(wavelengths)
    flux = np.array(flux)

    n_points = len(wavelengths)

    masked_regions = []

    all_groups = []
    if peak_groups:
        all_groups.extend(peak_groups)
    if trough_groups:
        all_groups.extend(trough_groups)

    for group in all_groups:

        if not group:
            continue

        group_wls = []
        feature_regions = []

        for item in group:

            center = item.get("wavelength")
            width = item.get("width")

            if center is None:
                continue

            if width is None:
                width = min_width

            width = max(width, min_width)

            group_wls.append(center)

            start = center - width / 2
            end = center + width / 2

            feature_regions.append((start, end))

        if not group_wls:
            continue

        group_min = min(group_wls)
        group_max = max(group_wls)

        group_width = group_max - group_min

        if group_width == 0:

            if len(wavelengths) > 1:
                group_width = np.median(np.diff(wavelengths))
            else:
                group_width = min_width

        extend = group_width * extend_ratio

        group_region = (
            group_min - extend,
            group_max + extend
        )

        feature_regions.append(group_region)

        masked_regions.extend(feature_regions)

    final_regions = _merge_regions(masked_regions)

    mask = np.zeros(n_points, dtype=bool)

    for start_wl, end_wl in final_regions:

        region_mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)

        mask |= region_mask

    cleaned_flux = flux.copy()

    if np.any(mask):

        unmasked_indices = np.where(~mask)[0]

        if len(unmasked_indices) > 1:

            # 先对所有 masked 区域做线性插值
            cleaned_flux[mask] = np.interp(
                wavelengths[mask],
                wavelengths[unmasked_indices],
                flux[unmasked_indices]
            )

            # 检查 masked 区域是否涉及开头或结尾，用该区域原始 flux 的中位数覆盖
            masked_indices = np.where(mask)[0]

            # 找到所有连续的 masked 段
            if len(masked_indices) > 0:
                breaks = np.where(np.diff(masked_indices) > 1)[0] + 1
                segments = np.split(masked_indices, breaks)

                for seg in segments:
                    if len(seg) == 0:
                        continue
                    # 判断是否触及开头或结尾
                    at_start = seg[0] == 0
                    at_end = seg[-1] == n_points - 1
                    if at_start or at_end:
                        seg_median = float(np.median(flux[seg]))
                        cleaned_flux[seg] = seg_median

        else:

            logging.warning(
                "Too few unmasked points for interpolation"
            )

    return {
        "cleaned_wavelength": wavelengths.tolist(),
        "cleaned_flux": cleaned_flux.tolist(),
        "mask": mask.tolist(),
        "masked_regions": final_regions
    }


def _merge_regions(regions):

    if not regions:
        return []

    regions = sorted(regions, key=lambda x: x[0])

    merged = [list(regions[0])]

    for current in regions[1:]:

        last = merged[-1]

        if current[0] <= last[1]:

            last[1] = max(last[1], current[1])

        else:

            merged.append(list(current))

    return [(r[0], r[1]) for r in merged]

