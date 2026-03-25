import numpy as np

from AstroAgent.agents.multi_agents.utils.usage import find_overlap_regions


def expand_overlap_window(overlap, scale_factor=2.0):
    left_raw, right_raw = overlap
    center = (left_raw + right_raw) / 2
    half_width = (right_raw - left_raw) * scale_factor / 2
    return center - half_width, center + half_width


def get_overlap_window(state, arm_name, arm_wavelength_range):
    spec = state['spectrum']
    wl = np.array(spec['new_wavelength'])
    fl = np.array(spec['weighted_flux'])
    d_f = np.array(spec['delta_flux'])
    overlap_regions = find_overlap_regions(arm_name, arm_wavelength_range)
    overlap_payload = []
    for region, overlap in overlap_regions.items():
        left, right = expand_overlap_window(overlap)

        mask = (wl >= left) & (wl <= right)

        overlap_payload.append({
            "region": region,
            "wavelength": wl[mask].tolist(),
            "flux": fl[mask].tolist(),
            "delta_flux": d_f[mask].tolist()
        })
    return overlap_payload


def check_candidate(peak):
    """检查CSST候选线条件"""
    if peak['width_in_km_s'] is None or peak['width_in_km_s'] < 2000:
        return False
    if (peak['seen_in_max_global_smoothing_scale_sigma'] is not None and 
        peak['seen_in_max_global_smoothing_scale_sigma'] > 2):
        return True
    return False


def get_Ly_alpha_candidates(state, peak_list):
    Lyalpha_candidate = []

    def check_local_snr_candidate(peak):
        """检查局部平滑尺度的信噪比条件（用于备选）"""
        if peak['width_in_km_s'] is None or peak['width_in_km_s'] < 2000:
            return False
        if (peak['seen_in_max_local_smoothing_scale_sigma'] is not None and 
            peak['seen_in_max_local_smoothing_scale_sigma'] > 2):
            return True
        return False

    for peak in peak_list:
        if check_candidate(peak):
            Lyalpha_candidate.append(peak['wavelength'])

    if not Lyalpha_candidate:
        for peak in peak_list:
            if check_local_snr_candidate(peak):
                Lyalpha_candidate.append(peak['wavelength'])

    state['Lyalpha_candidates'] = Lyalpha_candidate


def calculate_magnitude(state):
    pass

def calculate_color(state):
    pass