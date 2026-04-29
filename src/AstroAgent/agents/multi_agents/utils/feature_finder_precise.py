"""
feature_finder_precise.py

封存版：发射线找峰 + 吸收线找谷算法（对称设计）

包含：
  - 共享基础设施：质量标志、物理常数、高斯/倒高斯模型、continuum 自适应选择
  - find_emission_lines   : CWT 候选 + Δχ² 迭代窗口拟合 + 全局精修
  - find_absorption_lines : CWT 候选 + Δχ² 迭代窗口拟合 + 全局精修（倒高斯）
  - iterative_emission_detection   : 迭代掩膜找峰（每轮 mask 已检测峰）
  - iterative_absorption_detection : 迭代掩膜找谷（每轮 mask 已检测谷）
  - 绘图函数（手动调用）：
      - plot_spectrum_with_features : 绘制光谱和特征位置
      - plot_continuum_fit          : 绘制 continuum 拟合结果
      - plot_residual_with_features : 绘制残差光谱和特征位置
      - plot_feature_detection_summary : 综合图（光谱+continuum+特征+残差）

默认参数来源：0414.py（emission_params / detection_params）
"""

import numpy as np
import warnings
from scipy.stats import skew
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pywt
import pandas as pd


# =============================================================================
# 质量控制标志（发射线 & 吸收线共用）
# =============================================================================
FLAG_LOW_DELTA_CHI2 = 1   # Δχ² 低于低Δχ²警告阈值
FLAG_BOUNDARY       = 2   # 参数触碰边界（FWHM越界 / 峰高越界）
FLAG_BAD_ERROR      = 4   # 参数误差过大或拟合异常
FLAG_BLEND          = 8   # 与相邻线混叠
FLAG_LOW_SNRS       = 16  # 深度信噪比过低（仅吸收线）

# 物理常数
SIGMA_TO_FWHM = 2.355  # FWHM = 2.355 × σ
C_LIGHT = 299792.458   # 光速 km/s


# =============================================================================
# 输出格式化（符合 spectral_feature_catalog_schema.md）
# =============================================================================

def _classify_width(fwhm_km_s):
    """根据 FWHM (km/s) 分类宽窄性"""
    if fwhm_km_s < 1000:
        return "narrow"
    elif fwhm_km_s < 2000:
        return "intermediate"
    else:
        return "broad"


def _flags_to_quality_dict(flags):
    """
    将 bitwise flags 转换为布尔字典格式（LLM 友好）
    
    Parameters
    ----------
    flags : int or numpy integer
        质量标志位（bitwise）
    
    Returns
    -------
    dict : 质量控制布尔字典
    """
    # 确保转换为 Python int
    flags = int(flags)
    return {
        'low_delta_chi2': bool(flags & FLAG_LOW_DELTA_CHI2),
        'boundary_touch': bool(flags & FLAG_BOUNDARY),
        'large_error': bool(flags & FLAG_BAD_ERROR),
        'blended': bool(flags & FLAG_BLEND),
        'low_snr_depth': bool(flags & FLAG_LOW_SNRS),
    }


def format_feature_record(
    wavelength_val,
    wavelength_err,
    fwhm_a,
    amplitude,
    amplitude_err,
    integrated_flux,
    global_delta_chi2,
    local_delta_chi2,
    flags,
    flux_at_center=None,
    index=None,
    feature_type='emission',
    is_pseudo_peak=False,
    pseudo_reason=None,
    covered_troughs=0,
    trough_centers=''
):
    """
    格式化单条谱线特征为新规范格式。
    
    Parameters
    ----------
    wavelength_val : float
        中心波长 (Å)
    wavelength_err : float
        波长误差 (Å)
    fwhm_a : float
        FWHM (Å)
    amplitude : float
        幅度/深度
    amplitude_err : float
        幅度/深度误差
    integrated_flux : float
        积分流量/等效宽度
    global_delta_chi2 : float
        全局 Δχ²
    local_delta_chi2 : float
        局部 Δχ²
    flags : int
        质量标志位
    flux_at_center : float, optional
        原光谱在中心波长处的通量
    index : int, optional
        在光谱全局数组上的序号
    feature_type : str
        'emission' 或 'absorption'
    is_pseudo_peak : bool
        是否为伪峰候选（发射线专用）
    pseudo_reason : str or None
        伪峰判断原因
    covered_troughs : int
        覆盖的谷数量（发射线专用）
    trough_centers : str
        被覆盖谷的中心波长列表
    
    Returns
    -------
    dict : 符合 schema 的特征记录字典
    """
    # 计算 FWHM_km_s
    fwhm_km_s = fwhm_a / wavelength_val * C_LIGHT if wavelength_val > 0 else 0.0
    
    # 宽窄分类
    width_class = _classify_width(fwhm_km_s)
    
    # 质量控制布尔字典
    quality = _flags_to_quality_dict(flags)
    
    record = {
        # 基础标识
        'index': index,
        
        # 位置与形态
        'wavelength': float(wavelength_val),
        'wavelength_err': float(wavelength_err),
        'FWHM_A': float(fwhm_a),
        'FWHM_km_s': float(fwhm_km_s),
        
        # 强度
        'amplitude': float(amplitude),
        'amplitude_err': float(amplitude_err),
        'integrated_flux': float(integrated_flux),
        'flux_at_center': float(flux_at_center) if flux_at_center is not None else None,
        
        # 显著性
        'global_delta_chi2': float(global_delta_chi2),
        'local_delta_chi2': float(local_delta_chi2),
        
        # 质量控制
        'quality_low_delta_chi2': quality['low_delta_chi2'],
        'quality_boundary_touch': quality['boundary_touch'],
        'quality_large_error': quality['large_error'],
        'quality_blended': quality['blended'],
        'quality_low_snr_depth': quality['low_snr_depth'],
        
        # 分类摘要
        'width_class': width_class,
        'feature_type': feature_type,
    }
    
    # 发射线专用字段
    if feature_type == 'emission':
        record['is_pseudo_peak'] = is_pseudo_peak
        record['pseudo_reason'] = pseudo_reason
        record['covered_troughs'] = int(covered_troughs)
        record['trough_centers'] = trough_centers
    
    return record


def format_features_catalog(df_old, wavelength_array=None, flux_array=None,
                            feature_type='emission',
                            df_troughs=None, pseudo_trough_threshold=2):
    """
    将旧格式 DataFrame 转换为新规范格式。
    
    Parameters
    ----------
    df_old : DataFrame
        旧格式 DataFrame（含中文列名）
    wavelength_array : array, optional
        原始波长数组（用于计算 index 和 flux_at_center）
    flux_array : array, optional
        原始流量数组（用于计算 flux_at_center）
    feature_type : str
        'emission' 或 'absorption'
    df_troughs : DataFrame, optional
        吸收线结果（用于伪峰标注，仅发射线使用）
    pseudo_trough_threshold : int
        触发伪峰标注的最小谷数量
    
    Returns
    -------
    df_new : DataFrame
        新格式 DataFrame
    records : list of dict
        逐线字典列表
    """
    if len(df_old) == 0:
        return pd.DataFrame(), []
    
    records = []
    
    for i, (_, row) in enumerate(df_old.iterrows()):
        # 获取波长
        wl = row['波长(Å)']
        
        # 计算 index（在光谱数组上的位置）
        idx = None
        flux_at_center = None
        if wavelength_array is not None:
            idx = int(np.argmin(np.abs(wavelength_array - wl)))
            if flux_array is not None:
                flux_at_center = float(flux_array[idx])
        
        # 伪峰判断（发射线专用）
        is_pseudo = False
        pseudo_reason = None
        covered_troughs = 0
        trough_centers = ''
        
        if feature_type == 'emission' and df_troughs is not None and len(df_troughs) > 0:
            # 兼容新旧格式的列名
            fwhm_col = 'FWHM(Å)' if 'FWHM(Å)' in row else 'FWHM_A'
            sigma = row[fwhm_col] / SIGMA_TO_FWHM
            lo, hi = wl - 3 * sigma, wl + 3 * sigma
            
            # 检测 df_troughs 的列名格式
            trough_wl_col = '波长(Å)' if '波长(Å)' in df_troughs.columns else 'wavelength'
            trough_wl = df_troughs[trough_wl_col].values
            covered = trough_wl[(trough_wl >= lo) & (trough_wl <= hi)]
            covered_troughs = len(covered)
            trough_centers = ', '.join(f'{c:.1f}' for c in covered)
            if covered_troughs >= pseudo_trough_threshold:
                is_pseudo = True
                pseudo_reason = f"covers {covered_troughs} troughs at [{trough_centers}] Å"
        
        
        # 构建记录
        record = format_feature_record(
            wavelength_val=wl,
            wavelength_err=row['±波长误差'],
            fwhm_a=row['FWHM(Å)'],
            amplitude=row['幅度'] if feature_type == 'emission' else row['深度'],
            amplitude_err=row['±幅度误差'] if feature_type == 'emission' else row['±深度误差'],
            integrated_flux=row['积分流量(Jy·Å)'] if feature_type == 'emission' else row['等效宽度(Jy·Å)'],
            global_delta_chi2=row['全局Δχ²'],
            local_delta_chi2=row['局部Δχ²(窗口)'],
            flags=row['质量标志'],
            flux_at_center=flux_at_center,
            index=idx,
            feature_type=feature_type,
            is_pseudo_peak=is_pseudo,
            pseudo_reason=pseudo_reason,
            covered_troughs=covered_troughs,
            trough_centers=trough_centers,
        )
        records.append(record)
    
    df_new = pd.DataFrame(records)
    return df_new, records


def save_catalog_csv(df, filepath, feature_type='emission'):
    """
    将特征目录保存为 CSV 文件。
    
    Parameters
    ----------
    df : DataFrame
        新格式 DataFrame
    filepath : str
        输出文件路径
    feature_type : str
        'emission' 或 'absorption'
    """
    # 选择要保存的列（排除嵌套字典）
    cols_to_save = [
        'index', 'amplitude_rank',
        'wavelength', 'wavelength_err', 'FWHM_A', 'FWHM_km_s',
        'amplitude', 'amplitude_err', 'integrated_flux', 'flux_at_center',
        'global_delta_chi2', 'local_delta_chi2',
        'quality_low_delta_chi2', 'quality_boundary_touch', 'quality_large_error',
        'quality_blended', 'quality_low_snr_depth',
        'width_class', 'feature_type',
        'left_neighbor', 'right_neighbor',  # 近邻信息
    ]
    
    if feature_type == 'emission':
        cols_to_save.extend(['is_pseudo_peak', 'pseudo_reason', 'covered_troughs', 'trough_centers'])
    
    # 只保存存在的列
    cols_exist = [c for c in cols_to_save if c in df.columns]
    df[cols_exist].to_csv(filepath, index=False)
    return filepath


# =============================================================================
# 模型函数
# =============================================================================
def gaussian(wave, amplitude, center, sigma_width):
    """单个高斯发射线模型（amplitude > 0 为正峰）"""
    return amplitude * np.exp(-0.5 * ((wave - center) / sigma_width) ** 2)


def inverted_gaussian(wave, amplitude, center, sigma_width):
    """
    倒高斯吸收线模型（amplitude > 0，输出为负值向下凹陷）
    物理意义：amplitude = 吸收深度
    """
    return -amplitude * np.exp(-0.5 * ((wave - center) / sigma_width) ** 2)


def multi_gaussian_baseline(wave, *params):
    """
    多高斯 + 线性基线（发射线用）
    params: [c0, c1, amp1, cen1, sig1, ...]
    """
    c0, c1 = params[0], params[1]
    model = c0 + c1 * wave
    n = (len(params) - 2) // 3
    for i in range(n):
        idx = 2 + i * 3
        model += gaussian(wave, params[idx], params[idx+1], params[idx+2])
    return model


def multi_inverted_gaussian_baseline(wave, *params):
    """
    多倒高斯 + 线性基线（吸收线用）
    params: [c0, c1, amp1, cen1, sig1, ...]
    """
    c0, c1 = params[0], params[1]
    model = c0 + c1 * wave
    n = (len(params) - 2) // 3
    for i in range(n):
        idx = 2 + i * 3
        model += inverted_gaussian(wave, params[idx], params[idx+1], params[idx+2])
    return model


def _build_global_model_emission(wave, n_peaks):
    """构建全局发射线模型函数（用于 curve_fit）"""
    def model(wave, *params):
        c0, c1 = params[0], params[1]
        result = c0 + c1 * wave
        for i in range(n_peaks):
            idx = 2 + i * 3
            result += gaussian(wave, params[idx], params[idx+1], params[idx+2])
        return result
    return model


def _build_global_model_absorption(wave, n_troughs):
    """构建全局吸收线模型函数（用于 curve_fit）"""
    def model(wave, *params):
        c0, c1 = params[0], params[1]
        result = c0 + c1 * wave
        for i in range(n_troughs):
            idx = 2 + i * 3
            result += inverted_gaussian(wave, params[idx], params[idx+1], params[idx+2])
        return result
    return model


# =============================================================================
# 局部 Δχ² 计算（全局精修阶段使用）
# =============================================================================
def _local_delta_chi2_emission(wave_all, residual_all, sigma_all,
                                center, amplitude, sigma_width, halfwin=150.0):
    """
    计算发射线的局部 Δχ²（归一化：除以窗口数据点数）
    输入：全局残差（flux - global_continuum）
    """
    mask = (wave_all >= center - halfwin) & (wave_all <= center + halfwin)
    w, f, s = wave_all[mask], residual_all[mask], sigma_all[mask]
    if len(w) < 10:
        return 0.0

    def bl(wave, c0, c1):
        return c0 + c1 * wave

    def bl_gauss(wave, c0, c1, a, mu, sg):
        return c0 + c1 * wave + gaussian(wave, a, mu, sg)

    try:
        p0, _ = curve_fit(bl, w, f, p0=[0., 0.], sigma=s,
                          absolute_sigma=True, maxfev=2000)
        chi0 = np.sum(((f - bl(w, *p0)) / s) ** 2)
        blo = [-np.inf, -np.inf, 0., center - 5., sigma_width * 0.3]
        bhi = [np.inf,  np.inf,  np.inf, center + 5., sigma_width * 2.0]
        p1, _ = curve_fit(bl_gauss, w, f,
                          p0=[p0[0], p0[1], amplitude, center, sigma_width],
                          sigma=s, absolute_sigma=True,
                          bounds=(blo, bhi), maxfev=3000)
        chi1 = np.sum(((f - bl_gauss(w, *p1)) / s) ** 2)
        return float(chi0 - chi1) / max(len(w), 1)
    except Exception:
        return 0.0


def _local_delta_chi2_absorption(wave_all, flux_all, sigma_all,
                                  center, amplitude, sigma_width,
                                  halfwin=150.0, continuum_degree=2):
    """
    计算吸收线的局部 Δχ²（在局部 continuum 拟合后的残差上计算，归一化：除以窗口数据点数）
    输入：原始 flux（非全局残差），内部独立拟合局部 continuum
    """
    mask = (wave_all >= center - halfwin) & (wave_all <= center + halfwin)
    w, f, s = wave_all[mask], flux_all[mask], sigma_all[mask]
    if len(w) < 20:
        return 0.0

    _, local_res = _fit_local_continuum_window(
        w, f, s, degree=continuum_degree, n_iter=2, sigma_clip=2.0
    )

    def bl(wave, c0, c1):
        return c0 + c1 * wave

    def bl_inv(wave, c0, c1, a, mu, sg):
        return c0 + c1 * wave + inverted_gaussian(wave, a, mu, sg)

    try:
        p0, _ = curve_fit(bl, w, local_res, p0=[0., 0.],
                          sigma=s, absolute_sigma=True, maxfev=2000)
        chi0 = np.sum(((local_res - bl(w, *p0)) / s) ** 2)
        blo = [-np.inf, -np.inf, 0., center - 5., sigma_width * 0.3]
        bhi = [np.inf,  np.inf,  np.inf, center + 5., sigma_width * 2.0]
        p1, _ = curve_fit(bl_inv, w, local_res,
                          p0=[p0[0], p0[1], amplitude, center, sigma_width],
                          sigma=s, absolute_sigma=True,
                          bounds=(blo, bhi), maxfev=3000)
        chi1 = np.sum(((local_res - bl_inv(w, *p1)) / s) ** 2)
        return float(chi0 - chi1) / max(len(w), 1)  # 归一化：除以窗口数据点数
    except Exception:
        return 0.0


# =============================================================================
# 局部 continuum 拟合（吸收线窗口内使用）
# =============================================================================
def _fit_local_continuum_window(wave_win, flux_win, sigma_win,
                                 degree=2, n_iter=2, sigma_clip=2.0):
    """
    窗口内局部 continuum 拟合（带 σ-clipping 迭代，迭代 mask 吸收线点）

    Returns
    -------
    local_continuum : array
    local_residual  : array  (flux - continuum)
    """
    from specutils.fitting import fit_generic_continuum
    from specutils import Spectrum
    from astropy.modeling import models
    import astropy.units as u

    mask_fit = np.ones(len(wave_win), dtype=bool)
    continuum = None

    for _ in range(n_iter):
        if mask_fit.sum() < max(10, degree + 2):
            break
        try:
            sp = Spectrum(flux=flux_win[mask_fit] * u.Jy,
                          spectral_axis=wave_win[mask_fit] * u.AA)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cf = fit_generic_continuum(sp, model=models.Chebyshev1D(degree=degree))
            continuum = cf(wave_win * u.AA).value
            residual = flux_win - continuum
            mask_fit = (residual > -sigma_clip * sigma_win) | (residual > np.median(residual))
        except Exception:
            break

    if continuum is None:
        try:
            def lin(w, c0, c1): return c0 + c1 * w
            popt, _ = curve_fit(lin, wave_win, flux_win, sigma=sigma_win, absolute_sigma=True)
            continuum = lin(wave_win, *popt)
        except Exception:
            continuum = np.median(flux_win) * np.ones_like(flux_win)

    return continuum, flux_win - continuum


# =============================================================================
# 连续谱阶数自适应选择
# =============================================================================
def select_chebyshev_degree(sp, min_degree=1, max_degree=10, verbose=False):
    """
    自动选择最佳切比雪夫多项式阶数
    准则：残差 MAD 小 + 偏度接近理想值（0.5）+ 复杂度惩罚
    """
    from specutils.fitting import fit_generic_continuum
    from astropy.modeling import models

    flux = sp.flux.value
    best_degree = min_degree
    best_score = np.inf

    for deg in range(min_degree, max_degree + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cf = fit_generic_continuum(sp, model=models.Chebyshev1D(degree=deg))
            res = flux - cf(sp.spectral_axis).value
            mask = np.abs(res) < np.percentile(np.abs(res), 95)
            rm = res[mask]
            res_skew = abs(skew(rm))
            res_mad  = np.median(np.abs(rm - np.median(rm)))
            score = res_mad * 10.0 + (res_skew - 0.5) ** 2 * 5.0 + deg * 0.2
            if verbose:
                print(f"  degree={deg}: skew={res_skew:.4f}, MAD={res_mad:.4e}, score={score:.4f}")
            if score < best_score:
                best_score = score
                best_degree = deg
        except Exception as e:
            if verbose:
                print(f"  degree={deg}: 拟合失败 - {e}")
    if verbose:
        print(f"最佳阶数: {best_degree}")
    return best_degree


# =============================================================================
# 误差准备（发射线 & 吸收线共用）
# =============================================================================
def _prepare_sigma(flux, ivar=None, effective_snr=None,
                   continuum=None, sys_err_frac=0.0):
    """
    根据 ivar / effective_snr / 默认 SNR=7 准备误差数组，并可附加连续谱系统误差分量。

    Parameters
    ----------
    flux : array-like
    ivar : array-like, optional
    effective_snr : float or array-like, optional
        优先级：ivar > effective_snr > 默认 SNR=7
    continuum : array-like, optional
        连续谱模型值，用于计算系统误差分量。若为 None 则不附加。
    sys_err_frac : float, default 0.0
        连续谱相对系统误差系数 ε，附加方差 = (continuum * sys_err_frac)²。
        设为 0.0（默认）时完全不影响原有行为。
        典型取值 0.02~0.05，用于使强/弱流量光谱的有效噪声量级趋于一致。

    Returns
    -------
    sigma : ndarray, same shape as flux
    """
    flux = np.asarray(flux, dtype=np.float64)

    if ivar is not None:
        ivar = np.asarray(ivar, dtype=np.float64)
        # 处理 ivar 中的无效值（0 或负数）
        ivar_safe = np.maximum(ivar, 1e-10)
        sigma_pipe = 1.0 / np.sqrt(ivar_safe)
    elif effective_snr is not None:
        effective_snr = np.asarray(effective_snr, dtype=np.float64)
        # 处理无效值（NaN, Inf, 负数）
        mask = np.isfinite(effective_snr) & (effective_snr > 0)
        effective_snr = np.where(mask, effective_snr, 7.0)
        if effective_snr.ndim == 0:
            var = (np.median(flux) / float(effective_snr)) ** 2
            sigma_pipe = np.ones_like(flux) * np.sqrt(var)
        else:
            sigma_pipe = flux / effective_snr
    else:
        var = (np.median(flux) / 7.0) ** 2
        sigma_pipe = np.ones_like(flux, dtype=np.float64) * np.sqrt(var)

    # 附加连续谱相对系统误差分量：σ_eff² = σ_pipe² + (continuum * ε)²
    # 参考 DLA-Toolkit：nw = 1/(ivar⁻¹ + var_lss * f_model²)
    # [旧] 无系统误差分量，直接返回 sigma_pipe
    if sys_err_frac > 0.0 and continuum is not None:
        continuum = np.asarray(continuum, dtype=np.float64)
        sigma_sys = np.abs(continuum) * sys_err_frac
        return np.sqrt(sigma_pipe**2 + sigma_sys**2)

    return sigma_pipe


# =============================================================================
# 发射线找峰主函数
# =============================================================================
def find_emission_lines(
    wavelength,
    flux,
    ivar=None,
    effective_snr=None,
    # 连续谱拟合参数
    chebyshev_degree=None,
    chebyshev_min_degree=1,
    chebyshev_max_degree=10,
    # 窗口参数
    window_width=500,
    window_overlap=300,
    # Δχ² 参数（归一化）
    delta_chi2_base=0.05,
    dynamic_threshold_factor=10.0,
    global_delta_chi2_threshold=0.05,
    # FWHM 范围
    fwhm_min=3.0,
    fwhm_max=380.0,
    # 密度预筛
    enable_density_presift=True,
    density_presift_threshold=0.08,
    density_presift_topk=10,
    # 混叠剔除
    blend_k=1.0,
    # 全局精修参数
    narrow_sigma_shrink=0.5,
    narrow_sigma_expand=1.5,
    broad_sigma_shrink=0.5,
    broad_sigma_expand=2.0,
    amp_floor_frac=0.1,
    center_max_shift=5.0,
    # 系统误差分量（参考 DLA-Toolkit var_lss 思路）
    sys_err_frac=0.0,
    # 全局 smoothed 候选峰参数
    smooth_sigma=16.0,
    smooth_prominence_frac=0.1,
    # 控制参数
    verbose=False,
):
    """
    发射线找峰算法：CWT 初始候选 + Δχ² 迭代联合拟合 + 全局精修

    Parameters
    ----------
    wavelength : array-like, 波长数组 (Å)
    flux : array-like, 流量数组
    ivar : array-like, optional, 逆方差
    effective_snr : float or array-like, optional
        优先级：ivar > effective_snr > 默认 SNR=7
    chebyshev_degree : int or None, 指定阶数；None 则自动选择
    window_width : float, 滑动窗口宽度 (Å)
    window_overlap : float, 窗口重叠宽度 (Å)
    delta_chi2_base : float, Δχ² 基础阈值（归一化）
    dynamic_threshold_factor : float, 动态阈值增长因子
    global_delta_chi2_threshold : float, 全局精修 Δχ² 阈值（归一化）
    fwhm_min / fwhm_max : float, FWHM 范围 (Å)
    enable_density_presift : bool
    blend_k : float, 混叠剔除系数
    smooth_sigma : float
        对全局 residual 做高斯平滑的 σ（数据点单位，默认 16）。
        平滑后用 find_peaks 提取宽低振幅候选峰，作为窗口拟合的优先初始猜测。
    smooth_prominence_frac : float
        smoothed residual 找峰时的 prominence 阈值 = smooth_prominence_frac ×
        smoothed residual 的峰-峰范围，用于过滤旁瓣伪峰（默认 0.1）。
    verbose : bool

    Returns
    -------
    df_result : DataFrame
        列：['波长(Å)', '±波长误差', 'FWHM(Å)', '幅度', '±幅度误差',
              '积分流量(Jy·Å)', '全局Δχ²', '局部Δχ²(窗口)', '质量标志']
    """
    from specutils.fitting import fit_generic_continuum
    from specutils import Spectrum
    from astropy.modeling import models
    import astropy.units as u

    _COLS = ['波长(Å)', '±波长误差', 'FWHM(Å)', '幅度', '±幅度误差',
             '积分流量(Jy·Å)', '全局Δχ²', '局部Δχ²(窗口)', '质量标志']
    _EMPTY = pd.DataFrame(columns=_COLS)

    wavelength = np.asarray(wavelength)
    flux       = np.asarray(flux)

    if verbose:
        print(f"[峰] 波长范围: {wavelength.min():.1f} - {wavelength.max():.1f} Å")

    # ── 步骤1：连续谱拟合（仅执行一次，两遍共用） ───────────────────
    sp = Spectrum(flux=flux * u.Jy, spectral_axis=wavelength * u.AA)
    if chebyshev_degree is None:
        chebyshev_degree = select_chebyshev_degree(
            sp, min_degree=chebyshev_min_degree, max_degree=chebyshev_max_degree,
            verbose=verbose
        )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cf = fit_generic_continuum(sp, model=models.Chebyshev1D(degree=chebyshev_degree))
    continuum = cf(sp.spectral_axis).value
    residual  = flux - continuum
    if verbose:
        print(f"[峰] 连续谱: Chebyshev degree={chebyshev_degree}")

    # ── 步骤2：CWT 初始候选峰 ────────────────────────────────────────
    scales = np.arange(1, 80)
    coef, _ = pywt.cwt(residual, scales, 'mexh')
    max_response = np.max(coef, axis=0)
    peaks, _ = find_peaks(max_response, height=np.median(max_response))
    if verbose:
        print(f"[峰] CWT 候选峰: {len(peaks)} 个")
    if len(peaks) == 0:
        return _EMPTY

    # ── 步骤2b：全局 smoothed residual 候选峰（宽低振幅峰的先验） ───
    # 对 residual 做大核高斯平滑，在 smoothed 版本上用 find_peaks 提取宽特征候选。
    # 这些候选将在每个窗口内优先于 CWT 候选进行 Δχ² 验证，以提升 noisy 光谱
    # 上全局视觉明显但局部信噪比偏低的宽峰检出率。
    from scipy.ndimage import gaussian_filter1d as _gf1d
    smoothed_residual = _gf1d(residual, sigma=smooth_sigma)
    _sr_range = smoothed_residual.max() - smoothed_residual.min()
    _sr_prominence = smooth_prominence_frac * _sr_range if _sr_range > 0 else 0.0
    # 平均波长间距（用于将数据点单位的宽度换算为 Å）
    _mean_dwave = float(np.median(np.diff(wavelength))) if len(wavelength) > 1 else 1.0
    _smooth_peaks_idx, _smooth_peaks_props = find_peaks(
        smoothed_residual,
        height=0.0,           # 只找正峰（发射线在 residual 中为正）
        prominence=_sr_prominence,
        width=1,              # 启用宽度输出
    )
    # 构建 smoothed 候选峰信息表：{全局索引 -> (中心波长, 幅度, sigma初始估计)}
    # sigma_init = scipy 输出的半高宽（点数）/ SIGMA_TO_FWHM × 平均波长间距
    # 乘以退卷积修正系数 0.85，补偿 smoothing kernel 对宽度的展宽
    _smooth_peaks_info = {}
    for _si, _pi in enumerate(_smooth_peaks_idx):
        _fwhm_pts = float(_smooth_peaks_props['widths'][_si])
        _sigma_init_a = (_fwhm_pts / SIGMA_TO_FWHM) * _mean_dwave * 0.85
        _sigma_init_a = max(_sigma_init_a, fwhm_min / SIGMA_TO_FWHM)
        _sigma_init_a = min(_sigma_init_a, fwhm_max / SIGMA_TO_FWHM)
        _smooth_peaks_info[_pi] = {
            'wave': float(wavelength[_pi]),
            'amp':  float(smoothed_residual[_pi]),
            'sigma_init': _sigma_init_a,
        }
    if verbose:
        print(f"[峰] Smoothed 候选峰: {len(_smooth_peaks_info)} 个 "
              f"(smooth_sigma={smooth_sigma}, prominence≥{_sr_prominence:.4f})")
    # 找到的 smooth peaks 的中心波长
    smooth_peaks_waves = [info['wave'] for info in _smooth_peaks_info.values()]
    print(f"[峰] Smoothed 候选峰 {'found' if _smooth_peaks_info else 'NOT found'}: {len(_smooth_peaks_info)} 个，波长为 {smooth_peaks_waves}")

    # ── 内部辅助：窗口拟合 + 全局精修，支持两种精修模式 ────────────
    def _window_fit_and_refine(sigma_arr, use_snr_fallback=False):
        """
        use_snr_fallback=False : 初始算法（mask_keep = mask_dchi2 & ...，copy 版逻辑）
        use_snr_fallback=True  : 增强算法（mask_keep = (mask_dchi2 | mask_snr_fit) & ...）
        """
        # ── 步骤3：分块滑动窗口迭代拟合 ─────────────────────────────
        wave_min_l, wave_max_l = wavelength.min(), wavelength.max()
        win_starts = np.arange(wave_min_l, wave_max_l - window_width,
                               window_width - window_overlap)
        all_results = []

        for win_start in win_starts:
            win_end  = win_start + window_width
            mask_win = (wavelength >= win_start) & (wavelength < win_end)
            wave_win  = wavelength[mask_win]
            flux_win  = residual[mask_win]
            sigma_win = sigma_arr[mask_win]

            if len(wave_win) < 20:
                continue

            # ── Smoothed 候选峰（优先）：按 smoothed amplitude 降序 ──
            # 筛选中心波长落在当前窗口内的 smoothed 候选
            _sp_in_win = [
                (gidx, info) for gidx, info in _smooth_peaks_info.items()
                if win_start <= info['wave'] < win_end
            ]
            # 按 smoothed amplitude 降序排列
            _sp_in_win.sort(key=lambda x: x[1]['amp'], reverse=True)
            # 构建 smoothed 候选的 (中心波长, sigma初始估计) 列表
            smoothed_cands = [
                (info['wave'], info['sigma_init']) for _, info in _sp_in_win
            ]

            # ── CWT 候选峰（其次）：与原逻辑相同 ───────────────────
            peaks_in_win = peaks[(peaks >= np.argwhere(mask_win).min()) &
                                 (peaks <= np.argwhere(mask_win).max())]
            peaks_in_win = [p for p in peaks_in_win if mask_win[p]]

            # 如果 smoothed 和 CWT 均无候选，跳过本窗口
            if not peaks_in_win and not smoothed_cands:
                continue

            if peaks_in_win:
                peak_heights = [flux_win[np.argmin(np.abs(wave_win - wavelength[p]))]
                                for p in peaks_in_win]
                order = np.argsort(peak_heights)[::-1]
                cwt_cands_sorted = [peaks_in_win[i] for i in order]
                cwt_cands_wave   = [wavelength[p]   for p in cwt_cands_sorted]

                if enable_density_presift:
                    if len(cwt_cands_wave) / window_width > density_presift_threshold:
                        cwt_resp = [max_response[p] for p in cwt_cands_sorted]
                        topk = sorted(np.argsort(cwt_resp)[::-1][:density_presift_topk])
                        cwt_cands_sorted = [cwt_cands_sorted[k] for k in topk]
                        cwt_cands_wave   = [cwt_cands_wave[k]   for k in topk]
            else:
                cwt_cands_wave = []

            def _bl(wave, c0, c1): return c0 + c1 * wave
            try:
                popt_null, _ = curve_fit(_bl, wave_win, flux_win,
                                         p0=[0., 0.], sigma=sigma_win, absolute_sigma=True)
                chisq_prev = np.sum(((flux_win - _bl(wave_win, *popt_null)) / sigma_win) ** 2)
            except Exception:
                continue

            accepted_lines = []
            popt_prev      = popt_null

            # ── 候选处理循环：先 smoothed，后 CWT ────────────────────
            # smoothed 候选使用 scipy 估计的 sigma_init 作为初始猜测；
            # CWT 候选沿用原有固定 sigma=5.0 初始值。
            # 两组均通过统一的 Δχ²/FWHM/幅度/误差检验，混叠由 blend_k 自然处理。

            # 将两组候选统一为 (中心波长, sigma初始猜测, 是否来自_smoothed) 的迭代序列
            all_cands_iter = [(w, s, True)  for w, s in smoothed_cands] + \
                             [(w, 5.0, False) for w in cwt_cands_wave]

            for cand_wave, cand_sigma_init, is_smoothed_cand in all_cands_iter:
                n_acc     = len(accepted_lines)
                threshold = delta_chi2_base * (1 + dynamic_threshold_factor * n_acc)

                if n_acc == 0:
                    amp_g = flux_win[np.argmin(np.abs(wave_win - cand_wave))]
                    p0  = [popt_prev[0], popt_prev[1], amp_g, cand_wave, cand_sigma_init]
                    blo = [-np.inf, -np.inf, 0.,  win_start, fwhm_min / SIGMA_TO_FWHM]
                    bhi = [ np.inf,  np.inf, np.inf, win_end, fwhm_max / SIGMA_TO_FWHM]
                else:
                    p0  = [popt_prev[0], popt_prev[1]]
                    blo = [-np.inf, -np.inf]
                    bhi = [ np.inf,  np.inf]
                    for ln in accepted_lines:
                        p0.extend([ln['amplitude'], ln['center'], ln['sigma']])
                        blo.extend([0., ln['center'] - 10, fwhm_min / SIGMA_TO_FWHM])
                        bhi.extend([np.inf, ln['center'] + 10, fwhm_max / SIGMA_TO_FWHM])
                    amp_g = flux_win[np.argmin(np.abs(wave_win - cand_wave))]
                    p0.extend([amp_g, cand_wave, cand_sigma_init])
                    blo.extend([0., win_start, fwhm_min / SIGMA_TO_FWHM])
                    bhi.extend([np.inf, win_end, fwhm_max / SIGMA_TO_FWHM])

                try:
                    popt_new, pcov_new = curve_fit(
                        multi_gaussian_baseline, wave_win, flux_win,
                        p0=p0, sigma=sigma_win, absolute_sigma=True,
                        bounds=(blo, bhi), maxfev=5000
                    )
                    chisq_new  = np.sum(((flux_win - multi_gaussian_baseline(wave_win, *popt_new)) / sigma_win) ** 2)
                    n_pts_win  = max(len(wave_win), 1)
                    delta_chi2 = (chisq_prev - chisq_new) / n_pts_win
                    amp_new, center_new, sigma_new = popt_new[-3:]
                    fwhm_new = sigma_new * SIGMA_TO_FWHM
                    perr_new = np.sqrt(np.diag(pcov_new))[-3:]
                except Exception as _fit_exc:
                    if is_smoothed_cand:
                        print(f"  [峰|smoothed 拒绝] 波长={cand_wave:.1f}Å  原因: curve_fit 失败 ({_fit_exc})")
                    continue

                accept = True
                flags  = 0
                reject_reason = ''
                residual_at_center = residual[np.argmin(np.abs(wavelength - center_new))]
                amp_upper = abs(residual_at_center) * 1.5
                amp_lower = abs(residual_at_center) * 0.8

                if delta_chi2 < threshold:
                    accept = False
                    reject_reason = (f"Δχ²={delta_chi2:.4f} < threshold={threshold:.4f} "
                                     f"(base={delta_chi2_base}, n_acc={n_acc})")
                elif amp_new > amp_upper:
                    accept = False; flags |= FLAG_BOUNDARY
                    reject_reason = (f"amp={amp_new:.4f} > upper={amp_upper:.4f} "
                                     f"(residual@center={residual_at_center:.4f})")
                elif amp_new < amp_lower and not is_smoothed_cand:
                    # smoothed 候选峰展宽低平的特性导致单点参考值偏高，跳过下界检验
                    accept = False; flags |= FLAG_BOUNDARY
                    reject_reason = (f"amp={amp_new:.4f} < lower={amp_lower:.4f} "
                                     f"(residual@center={residual_at_center:.4f})")
                elif not (fwhm_min <= fwhm_new <= fwhm_max):
                    accept = False; flags |= FLAG_BOUNDARY
                    reject_reason = f"FWHM={fwhm_new:.2f}Å 超出范围 [{fwhm_min}, {fwhm_max}]"
                elif len(perr_new) == 3:
                    if np.any(perr_new / np.abs(popt_new[-3:]) > 0.5):
                        accept = False; flags |= FLAG_BAD_ERROR
                        rel_errs = perr_new / np.abs(popt_new[-3:])
                        reject_reason = (f"参数相对误差过大: "
                                         f"amp_rel={rel_errs[0]:.3f}, "
                                         f"cen_rel={rel_errs[1]:.3f}, "
                                         f"sig_rel={rel_errs[2]:.3f}")

                if accept:
                    drop_idxs = []
                    for pi, pl in enumerate(accepted_lines):
                        sep = abs(center_new - pl['center'])
                        if sep < blend_k * min(fwhm_new, pl['fwhm']):
                            if fwhm_new > pl['fwhm']:
                                drop_idxs.append(pi)
                            else:
                                accept = False; flags |= FLAG_BLEND
                                reject_reason = (f"混叠: 与已接受线 center={pl['center']:.1f}Å "
                                                 f"距离={sep:.1f}Å < blend_k*min_fwhm={blend_k*min(fwhm_new,pl['fwhm']):.1f}Å")
                                break
                    if accept and drop_idxs:
                        for di in sorted(drop_idxs, reverse=True):
                            del accepted_lines[di]

                if is_smoothed_cand and not accept:
                    print(f"  [峰|smoothed 拒绝] 波长={cand_wave:.1f}Å  原因: {reject_reason}")

                if accept:
                    if delta_chi2 < 0.01:
                        flags |= FLAG_LOW_DELTA_CHI2
                    accepted_lines.append({
                        'center': center_new, 'center_err': perr_new[1],
                        'amplitude': amp_new, 'amplitude_err': perr_new[0],
                        'sigma': sigma_new,   'sigma_err': perr_new[2],
                        'fwhm': fwhm_new,
                        'flux': amp_new * sigma_new * np.sqrt(2 * np.pi),
                        'delta_chi2': delta_chi2, 'flags': flags,
                        'window': (win_start, win_end)
                    })
                    chisq_prev = chisq_new
                    popt_prev  = popt_new

            all_results.extend(accepted_lines)

        if not all_results:
            return pd.DataFrame()

        # ── 步骤4：合并去重 ──────────────────────────────────────────
        df_lines = pd.DataFrame(all_results).sort_values('center').reset_index(drop=True)
        to_rm = []
        for i in range(len(df_lines) - 1):
            if i in to_rm: continue
            for j in range(i + 1, len(df_lines)):
                if j in to_rm: continue
                if abs(df_lines.loc[i,'center'] - df_lines.loc[j,'center']) < 5.0:
                    if df_lines.loc[i,'delta_chi2'] >= df_lines.loc[j,'delta_chi2']:
                        to_rm.append(j)
                    else:
                        to_rm.append(i); break
        df_windowed = df_lines.drop(to_rm).reset_index(drop=True)
        if len(df_windowed) == 0:
            return pd.DataFrame()

        # ── 步骤5：全局精修（迭代剔除） ──────────────────────────────
        fwhm_median_l = df_windowed['fwhm'].median()
        df_iter       = df_windowed.copy()
        anchor        = df_windowed['center'].values.copy()

        while True:
            n_lines = len(df_iter)
            if n_lines == 0:
                break

            p0_g = [0., 0.]; blo_g = [-np.inf, -np.inf]; bhi_g = [np.inf, np.inf]
            for i, (_, row) in enumerate(df_iter.iterrows()):
                is_broad = row['fwhm'] >= fwhm_median_l
                shrink   = broad_sigma_shrink if is_broad else narrow_sigma_shrink
                expand   = broad_sigma_expand if is_broad else narrow_sigma_expand
                p0_g.extend([row['amplitude'], row['center'], row['sigma']])
                blo_g.extend([row['amplitude'] * amp_floor_frac,
                               anchor[i] - center_max_shift, row['sigma'] * shrink])
                bhi_g.extend([np.inf, anchor[i] + center_max_shift, row['sigma'] * expand])

            gm = _build_global_model_emission(wavelength, n_lines)
            try:
                popt_g, pcov_g = curve_fit(gm, wavelength, residual,
                                            p0=p0_g, sigma=sigma_arr, absolute_sigma=True,
                                            bounds=(blo_g, bhi_g), maxfev=20000)
                perr_g = np.sqrt(np.diag(pcov_g))
            except Exception:
                break

            updated = []
            for i, (_, row) in enumerate(df_iter.iterrows()):
                idx = 2 + i * 3
                ag, cg, sg = popt_g[idx], popt_g[idx+1], popt_g[idx+2]
                dchi2 = _local_delta_chi2_emission(wavelength, residual, sigma_arr, cg, ag, sg)
                updated.append({**row.to_dict(),
                                 'center': cg,  'center_err': perr_g[idx+1],
                                 'amplitude': ag, 'amplitude_err': perr_g[idx],
                                 'sigma': sg,    'sigma_err': perr_g[idx+2],
                                 'fwhm': sg * SIGMA_TO_FWHM,
                                 'flux': ag * sg * np.sqrt(2 * np.pi),
                                 'global_delta_chi2': dchi2})

            df_res = pd.DataFrame(updated).reset_index(drop=True)
            mask_dchi2 = df_res['global_delta_chi2'] >= global_delta_chi2_threshold
            mask_fwhm  = df_res['fwhm'].between(fwhm_min, fwhm_max)

            _amp_upper_frac = 1.5
            _amp_lower_frac = 0.5
            def _amp_ok_em(row):
                ic = np.argmin(np.abs(wavelength - row['center']))
                res_val = abs(residual[ic])
                return (row['amplitude'] <= res_val * _amp_upper_frac and
                        row['amplitude'] >= res_val * _amp_lower_frac)
            mask_amp = df_res.apply(_amp_ok_em, axis=1)
            mask_err = (
                df_res['amplitude_err'].abs() / df_res['amplitude'].abs()
            ).fillna(np.inf) <= 1.0

            if use_snr_fallback:
                # 增强模式：A/σ_A ≥ 3 与 Δχ² 取并集，降低弱信号漏检率
                mask_snr_fit = (
                    df_res['amplitude'].abs() / df_res['amplitude_err'].abs()
                ).fillna(0.0) >= 3.0
                mask_keep = (mask_dchi2 | mask_snr_fit) & mask_fwhm & mask_amp & mask_err
            else:
                # 初始模式（copy 版）：严格 Δχ² 判据
                mask_keep = mask_dchi2 & mask_fwhm & mask_amp & mask_err

            blend_rm = set()
            surv = df_res[mask_keep]; sidx = list(surv.index)
            for ii in range(len(sidx)):
                for jj in range(ii + 1, len(sidx)):
                    ri, rj = surv.loc[sidx[ii]], surv.loc[sidx[jj]]
                    if abs(ri['center'] - rj['center']) < blend_k * min(ri['fwhm'], rj['fwhm']):
                        blend_rm.add(sidx[ii] if ri['fwhm'] < rj['fwhm'] else sidx[jj])
            if blend_rm:
                mask_keep = mask_keep & ~df_res.index.isin(blend_rm)

            n_removed = (~mask_keep).sum()
            if n_removed > 0:
                df_iter = df_res[mask_keep].reset_index(drop=True)
                anchor  = anchor[mask_keep.values]
            else:
                df_iter = df_res; break

        return df_iter

    # ── 第一遍：初始算法（copy 版，无 sys_err_frac，严格 Δχ² 判据） ──
    sigma_base = _prepare_sigma(flux, ivar, effective_snr)
    if verbose:
        print(f"[峰] 第一遍：初始算法")
    df_pass1 = _window_fit_and_refine(sigma_base, use_snr_fallback=False)

    if len(df_pass1) > 1:
        if verbose:
            print(f"[峰] 全局精修完成（第一遍）: {len(df_pass1)} 条发射线")
        df_out = df_pass1[['center','center_err','fwhm','amplitude','amplitude_err',
                            'flux','global_delta_chi2','delta_chi2','flags']].copy()
        df_out.columns = _COLS
        return df_out

    # ── 第二遍：增强算法（sys_err_frac + A/σ_A 并集判据） ────────────
    if verbose:
        print(f"[峰] 第一遍未找到峰，启动第二遍：增强算法（sys_err_frac={sys_err_frac}）")
    sigma_enhanced = _prepare_sigma(flux, ivar, effective_snr,
                                    continuum=continuum, sys_err_frac=sys_err_frac)
    df_pass2 = _window_fit_and_refine(sigma_enhanced, use_snr_fallback=True)

    if verbose:
        print(f"[峰] 全局精修完成（第二遍）: {len(df_pass2)} 条发射线")
    if len(df_pass2) == 0:
        return _EMPTY

    df_out = df_pass2[['center','center_err','fwhm','amplitude','amplitude_err',
                        'flux','global_delta_chi2','delta_chi2','flags']].copy()
    df_out.columns = _COLS
    return df_out


# =============================================================================
# 吸收线找谷主函数
# =============================================================================
def find_absorption_lines(
    wavelength,
    flux,
    ivar=None,
    effective_snr=None,
    # 连续谱拟合参数
    chebyshev_degree=None,
    chebyshev_min_degree=1,
    chebyshev_max_degree=10,
    # 窗口参数
    window_width=100,
    window_overlap=60,
    # Δχ² 参数（归一化）
    delta_chi2_base=0.02,
    dynamic_threshold_factor=0.05,
    global_delta_chi2_threshold=0.05,  # 归一化后阈值，与发射线一致
    # FWHM 范围
    fwhm_min=5.0,
    fwhm_max=100.0,
    # 密度预筛
    enable_density_presift=False,
    density_presift_threshold=0.08,
    density_presift_topk=10,
    # 混叠剔除
    blend_k=1.0,
    # 深度信噪比检验
    snr_depth_threshold=3.0,
    # 局部 continuum 拟合参数
    local_continuum_degree=2,
    local_continuum_n_iter=2,
    local_continuum_sigma_clip=2.0,
    # 全局精修参数
    narrow_sigma_shrink=0.5,
    narrow_sigma_expand=1.5,
    broad_sigma_shrink=0.5,
    broad_sigma_expand=2.0,
    amp_floor_frac=0.1,
    center_max_shift=5.0,
    # 全局 smoothed 候选谷参数
    smooth_sigma_trough=16.0,
    smooth_prominence_frac_trough=0.1,
    # 控制参数
    verbose=False,
    return_intermediate=False,
):
    """
    吸收线找谷算法：CWT 初始候选 + Δχ² 迭代联合拟合 + 全局精修

    与 find_emission_lines 的主要区别：
    - CWT 在 -residual 上找谷（吸收线在残差中为负值）
    - 拟合使用倒高斯模型（inverted_gaussian）
    - 窗口内先拟合局部 continuum，再在局部残差上计算 Δχ²
    - 接受条件额外包含：中心处残差必须为负 + 深度信噪比检验

    Parameters
    ----------
    wavelength : array-like, 波长数组 (Å)
    flux : array-like, 流量数组
    ivar : array-like, optional, 逆方差
    effective_snr : float or array-like, optional
        优先级：ivar > effective_snr > 默认 SNR=7
    chebyshev_degree : int or None, 指定阶数；None 则自动选择
    window_width : float, 滑动窗口宽度 (Å)
    window_overlap : float, 窗口重叠宽度 (Å)
    delta_chi2_base : float, Δχ² 基础阈值（归一化）
    dynamic_threshold_factor : float, 动态阈值增长因子
    global_delta_chi2_threshold : float, 全局精修 Δχ² 阈值（归一化）
    fwhm_min / fwhm_max : float, FWHM 范围 (Å)
    enable_density_presift : bool
    blend_k : float, 混叠剔除系数（吸收线：保窄去宽）
    snr_depth_threshold : float, 深度信噪比阈值（默认 3.0）
    local_continuum_degree/n_iter/sigma_clip : 局部 continuum 拟合参数
    smooth_sigma_trough : float
        对全局 -residual 做高斯平滑的 σ（数据点单位，默认 16）。
        平滑后用 find_peaks 提取宽低振幅候选谷，作为窗口拟合的优先初始猜测。
    smooth_prominence_frac_trough : float
        smoothed -residual 找谷时的 prominence 阈值 = smooth_prominence_frac_trough ×
        smoothed -residual 的峰-峰范围（默认 0.1）。
    verbose : bool
    return_intermediate : bool, 是否额外返回窗口阶段中间结果

    Returns
    -------
    df_result : DataFrame
        列：['波长(Å)', '±波长误差', 'FWHM(Å)', '深度', '±深度误差',
              '等效宽度(Jy·Å)', '全局Δχ²', '局部Δχ²(窗口)', '质量标志']
    如果 return_intermediate=True，返回 (df_result, df_windowed_fmt)
    """
    from specutils.fitting import fit_generic_continuum
    from specutils import Spectrum
    from astropy.modeling import models
    import astropy.units as u

    _COLS = ['波长(Å)', '±波长误差', 'FWHM(Å)', '深度', '±深度误差',
             '等效宽度(Jy·Å)', '全局Δχ²', '局部Δχ²(窗口)', '质量标志']
    _EMPTY = pd.DataFrame(columns=_COLS)

    wavelength = np.asarray(wavelength)
    flux       = np.asarray(flux)

    if verbose:
        print(f"[谷] 波长范围: {wavelength.min():.1f} - {wavelength.max():.1f} Å")

    # ── 步骤1：连续谱拟合 ────────────────────────────────────────────
    sp = Spectrum(flux=flux * u.Jy, spectral_axis=wavelength * u.AA)
    if chebyshev_degree is None:
        chebyshev_degree = select_chebyshev_degree(
            sp, min_degree=chebyshev_min_degree, max_degree=chebyshev_max_degree,
            verbose=verbose
        )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cf = fit_generic_continuum(sp, model=models.Chebyshev1D(degree=chebyshev_degree))
    continuum = cf(sp.spectral_axis).value
    residual  = flux - continuum
    sigma = _prepare_sigma(flux, ivar, effective_snr)
    if verbose:
        print(f"[谷] 连续谱: Chebyshev degree={chebyshev_degree}")

    # ── 步骤2：CWT 初始候选谷（对 -residual 找峰） ──────────────────
    scales = np.arange(1, 80)
    coef, _ = pywt.cwt(-residual, scales, 'mexh')
    max_response = np.max(coef, axis=0)
    troughs, _ = find_peaks(max_response, height=np.median(max_response))
    if verbose:
        print(f"[谷] CWT 候选谷: {len(troughs)} 个")
    if len(troughs) == 0:
        if return_intermediate:
            return _EMPTY, pd.DataFrame()
        return _EMPTY

    # ── 步骤2b：全局 smoothed -residual 候选谷（宽低振幅谷的先验） ──
    # 对 -residual 做大核高斯平滑，在 smoothed 版本上用 find_peaks 提取宽特征候选。
    # 这些候选将在每个窗口内优先于 CWT 候选进行 Δχ² 验证，以提升 noisy 光谱
    # 上全局视觉明显但局部信噪比偏低的宽吸收谷检出率。
    from scipy.ndimage import gaussian_filter1d as _gf1d_trough
    _neg_residual = -residual
    smoothed_neg_residual = _gf1d_trough(_neg_residual, sigma=smooth_sigma_trough)
    _snr_range = smoothed_neg_residual.max() - smoothed_neg_residual.min()
    _snr_prominence = smooth_prominence_frac_trough * _snr_range if _snr_range > 0 else 0.0
    # 平均波长间距（用于将数据点单位的宽度换算为 Å）
    _mean_dwave_t = float(np.median(np.diff(wavelength))) if len(wavelength) > 1 else 1.0
    _smooth_troughs_idx, _smooth_troughs_props = find_peaks(
        smoothed_neg_residual,
        height=0.0,           # 只找正峰（吸收谷在 -residual 中为正）
        prominence=_snr_prominence,
        width=1,              # 启用宽度输出
    )
    # 构建 smoothed 候选谷信息表：{全局索引 -> (中心波长, 幅度, sigma初始估计)}
    # sigma_init = scipy 输出的半高宽（点数）/ SIGMA_TO_FWHM × 平均波长间距
    # 乘以退卷积修正系数 0.85，补偿 smoothing kernel 对宽度的展宽
    _smooth_troughs_info = {}
    for _si, _pi in enumerate(_smooth_troughs_idx):
        _fwhm_pts = float(_smooth_troughs_props['widths'][_si])
        _sigma_init_t = (_fwhm_pts / SIGMA_TO_FWHM) * _mean_dwave_t * 0.85
        _sigma_init_t = max(_sigma_init_t, fwhm_min / SIGMA_TO_FWHM)
        _sigma_init_t = min(_sigma_init_t, fwhm_max / SIGMA_TO_FWHM)
        _smooth_troughs_info[_pi] = {
            'wave': float(wavelength[_pi]),
            'amp':  float(smoothed_neg_residual[_pi]),
            'sigma_init': _sigma_init_t,
        }
    if verbose:
        print(f"[谷] Smoothed 候选谷: {len(_smooth_troughs_info)} 个 "
              f"(smooth_sigma_trough={smooth_sigma_trough}, prominence≥{_snr_prominence:.4f})")
    smooth_troughs_waves = [info['wave'] for info in _smooth_troughs_info.values()]
    print(f"[谷] Smoothed 候选谷 {'found' if _smooth_troughs_info else 'NOT found'}: "
          f"{len(_smooth_troughs_info)} 个，波长为 {smooth_troughs_waves}")

    # ── 步骤3：分块滑动窗口迭代拟合 ─────────────────────────────────
    wave_min, wave_max = wavelength.min(), wavelength.max()
    window_starts   = np.arange(wave_min, wave_max - window_width, window_width - window_overlap)
    all_window_results = []

    for win_start in window_starts:
        win_end  = win_start + window_width
        mask_win = (wavelength >= win_start) & (wavelength < win_end)
        wave_win      = wavelength[mask_win]
        flux_win_orig = flux[mask_win]
        sigma_win     = sigma[mask_win]

        if len(wave_win) < 20:
            continue

        # 窗口内局部 continuum 拟合
        _, flux_win = _fit_local_continuum_window(
            wave_win, flux_win_orig, sigma_win,
            degree=local_continuum_degree,
            n_iter=local_continuum_n_iter,
            sigma_clip=local_continuum_sigma_clip
        )

        # ── Smoothed 候选谷（优先）：按 smoothed amplitude 降序 ──
        # 筛选中心波长落在当前窗口内的 smoothed 候选
        _st_in_win = [
            (gidx, info) for gidx, info in _smooth_troughs_info.items()
            if win_start <= info['wave'] < win_end
        ]
        # 按 smoothed amplitude 降序排列
        _st_in_win.sort(key=lambda x: x[1]['amp'], reverse=True)
        # 构建 smoothed 候选的 (中心波长, sigma初始估计) 列表
        smoothed_trough_cands = [
            (info['wave'], info['sigma_init']) for _, info in _st_in_win
        ]

        # ── CWT 候选谷（其次）：与原逻辑相同 ────────────────────────
        troughs_in_win = troughs[(troughs >= np.argwhere(mask_win).min()) &
                                 (troughs <= np.argwhere(mask_win).max())]
        troughs_in_win = [p for p in troughs_in_win if mask_win[p]]

        # 如果 smoothed 和 CWT 均无候选，跳过本窗口
        if not troughs_in_win and not smoothed_trough_cands:
            continue

        if troughs_in_win:
            # 按谷深度升序排列（最负在前）
            trough_depths = [flux_win[np.argmin(np.abs(wave_win - wavelength[p]))]
                             for p in troughs_in_win]
            order = np.argsort(trough_depths)
            cands_sorted = [troughs_in_win[i] for i in order]
            cands_wave   = [wavelength[p]      for p in cands_sorted]

            if enable_density_presift:
                if len(cands_wave) / window_width > density_presift_threshold:
                    cwt_resp = [max_response[p] for p in cands_sorted]
                    topk = sorted(np.argsort(cwt_resp)[::-1][:density_presift_topk])
                    cands_sorted = [cands_sorted[k] for k in topk]
                    cands_wave   = [cands_wave[k]   for k in topk]
        else:
            cands_wave = []

        def _bl(wave, c0, c1): return c0 + c1 * wave
        try:
            popt_null, _ = curve_fit(_bl, wave_win, flux_win,
                                     p0=[0., 0.], sigma=sigma_win, absolute_sigma=True)
            chisq_prev = np.sum(((flux_win - _bl(wave_win, *popt_null)) / sigma_win) ** 2)
        except Exception:
            continue

        accepted_lines = []
        popt_prev      = popt_null

        # ── 候选处理循环：先 smoothed，后 CWT ──────────────────────
        # smoothed 候选使用 scipy 估计的 sigma_init 作为初始猜测；
        # CWT 候选沿用原有固定 sigma=5.0 初始值。
        # 两组均通过统一的 Δχ²/FWHM/深度/误差检验，混叠由 blend_k 自然处理。

        # 将两组候选统一为 (中心波长, sigma初始猜测, 是否来自_smoothed) 的迭代序列
        all_cands_trough_iter = [(w, s, True)  for w, s in smoothed_trough_cands] + \
                                [(w, 5.0, False) for w in cands_wave]

        for cand_wave, cand_sigma_init_t, is_smoothed_trough in all_cands_trough_iter:
            n_acc     = len(accepted_lines)
            threshold = delta_chi2_base * (1 + dynamic_threshold_factor * n_acc)

            flux_at_cand = flux_win[np.argmin(np.abs(wave_win - cand_wave))]
            amp_guess    = abs(flux_at_cand)

            if n_acc == 0:
                p0  = [popt_prev[0], popt_prev[1], amp_guess, cand_wave, cand_sigma_init_t]
                blo = [-np.inf, -np.inf, 0.,  win_start, fwhm_min / SIGMA_TO_FWHM]
                bhi = [ np.inf,  np.inf, np.inf, win_end, fwhm_max / SIGMA_TO_FWHM]
            else:
                p0  = [popt_prev[0], popt_prev[1]]
                blo = [-np.inf, -np.inf]
                bhi = [ np.inf,  np.inf]
                for ln in accepted_lines:
                    p0.extend([ln['amplitude'], ln['center'], ln['sigma']])
                    blo.extend([0., ln['center'] - 10, fwhm_min / SIGMA_TO_FWHM])
                    bhi.extend([np.inf, ln['center'] + 10, fwhm_max / SIGMA_TO_FWHM])
                p0.extend([amp_guess, cand_wave, cand_sigma_init_t])
                blo.extend([0., win_start, fwhm_min / SIGMA_TO_FWHM])
                bhi.extend([np.inf, win_end, fwhm_max / SIGMA_TO_FWHM])

            try:
                popt_new, pcov_new = curve_fit(
                    multi_inverted_gaussian_baseline, wave_win, flux_win,
                    p0=p0, sigma=sigma_win, absolute_sigma=True,
                    bounds=(blo, bhi), maxfev=5000
                )
                chisq_new  = np.sum(((flux_win - multi_inverted_gaussian_baseline(wave_win, *popt_new)) / sigma_win) ** 2)
                n_pts_win  = max(len(wave_win), 1)
                delta_chi2 = (chisq_prev - chisq_new) / n_pts_win
                amp_new, center_new, sigma_new = popt_new[-3:]
                fwhm_new = sigma_new * SIGMA_TO_FWHM
                perr_new = np.sqrt(np.diag(pcov_new))[-3:]
            except Exception as _fit_exc_t:
                if is_smoothed_trough:
                    print(f"  [谷|smoothed 拒绝] 波长={cand_wave:.1f}Å  原因: curve_fit 失败 ({_fit_exc_t})")
                continue

            accept = True
            flags  = 0
            reject_reason_t = ''

            # 条件1：Δχ² 显著性
            if delta_chi2 < threshold:
                accept = False
                reject_reason_t = (f"Δχ²={delta_chi2:.4f} < threshold={threshold:.4f} "
                                   f"(base={delta_chi2_base}, n_acc={n_acc})")
            # 条件2：FWHM 范围
            elif not (fwhm_min <= fwhm_new <= fwhm_max):
                accept = False; flags |= FLAG_BOUNDARY
                reject_reason_t = f"FWHM={fwhm_new:.2f}Å 超出范围 [{fwhm_min}, {fwhm_max}]"
            # 条件3：参数误差
            elif len(perr_new) == 3:
                if np.any(perr_new / np.abs(popt_new[-3:]) > 0.5):
                    accept = False; flags |= FLAG_BAD_ERROR
                    rel_errs_t = perr_new / np.abs(popt_new[-3:])
                    reject_reason_t = (f"参数相对误差过大: "
                                       f"amp_rel={rel_errs_t[0]:.3f}, "
                                       f"cen_rel={rel_errs_t[1]:.3f}, "
                                       f"sig_rel={rel_errs_t[2]:.3f}")

            # 条件4：中心处局部残差必须为负（吸收线是向下的谷）
            if accept:
                local_ci = np.argmin(np.abs(wave_win - center_new))
                res_at_center = flux_win[local_ci]
                if res_at_center >= 0:
                    accept = False; flags |= FLAG_BAD_ERROR
                    reject_reason_t = (f"中心处残差非负: res={res_at_center:.4f} >= 0")
                elif amp_new > 2 * abs(res_at_center) and not is_smoothed_trough:
                    # smoothed 候选谷展宽低平导致单点残差偏小，跳过上界检验
                    accept = False; flags |= FLAG_BAD_ERROR
                    reject_reason_t = (f"amp={amp_new:.4f} > 2×|res|={2*abs(res_at_center):.4f}")

            # 条件5：深度信噪比
            if accept:
                local_noise = np.median(np.abs(flux_win - np.median(flux_win))) * 1.4826
                if local_noise > 0 and amp_new / local_noise < snr_depth_threshold:
                    accept = False; flags |= FLAG_LOW_SNRS
                    reject_reason_t = (f"深度信噪比={amp_new/local_noise:.2f} < {snr_depth_threshold}")

            # 混叠检查（保窄去宽）
            if accept:
                drop_idxs = []
                for pi, pl in enumerate(accepted_lines):
                    sep = abs(center_new - pl['center'])
                    if sep < blend_k * min(fwhm_new, pl['fwhm']):
                        if fwhm_new > pl['fwhm']:
                            accept = False; flags |= FLAG_BLEND
                            reject_reason_t = (f"混叠: 与已接受谷 center={pl['center']:.1f}Å "
                                               f"距离={sep:.1f}Å < blend_k*min_fwhm={blend_k*min(fwhm_new,pl['fwhm']):.1f}Å")
                            break
                        else:
                            drop_idxs.append(pi)
                if accept and drop_idxs:
                    for di in sorted(drop_idxs, reverse=True):
                        del accepted_lines[di]

            if is_smoothed_trough and not accept:
                print(f"  [谷|smoothed 拒绝] 波长={cand_wave:.1f}Å  原因: {reject_reason_t}")

            if accept:
                if delta_chi2 < 0.01:
                    flags |= FLAG_LOW_DELTA_CHI2
                accepted_lines.append({
                    'center': center_new, 'center_err': perr_new[1],
                    'amplitude': amp_new, 'amplitude_err': perr_new[0],
                    'sigma': sigma_new,   'sigma_err': perr_new[2],
                    'fwhm': fwhm_new,
                    'flux': amp_new * sigma_new * np.sqrt(2 * np.pi),
                    'delta_chi2': delta_chi2, 'flags': flags,
                    'window': (win_start, win_end)
                })
                chisq_prev = chisq_new
                popt_prev  = popt_new

        all_window_results.extend(accepted_lines)

    if verbose:
        print(f"[谷] 窗口拟合完成: {len(all_window_results)} 条候选吸收线")
    if not all_window_results:
        if return_intermediate:
            return _EMPTY, pd.DataFrame()
        return _EMPTY

    # ── 步骤4：合并去重 ──────────────────────────────────────────────
    df_lines = pd.DataFrame(all_window_results).sort_values('center').reset_index(drop=True)
    to_rm = []
    for i in range(len(df_lines) - 1):
        if i in to_rm: continue
        for j in range(i + 1, len(df_lines)):
            if j in to_rm: continue
            if abs(df_lines.loc[i,'center'] - df_lines.loc[j,'center']) < 5.0:
                if df_lines.loc[i,'delta_chi2'] >= df_lines.loc[j,'delta_chi2']:
                    to_rm.append(j)
                else:
                    to_rm.append(i); break
    df_windowed = df_lines.drop(to_rm).reset_index(drop=True)
    df_windowed_copy = df_windowed.copy()
    if verbose:
        print(f"[谷] 去重后: {len(df_windowed)} 条")
    if len(df_windowed) == 0:
        if return_intermediate:
            return _EMPTY, pd.DataFrame()
        return _EMPTY

    # ── 步骤5：全局精修（迭代剔除） ──────────────────────────────────
    fwhm_median    = df_windowed['fwhm'].median()
    df_global_iter = df_windowed.copy()
    center_anchor  = df_windowed['center'].values.copy()

    while True:
        n_lines = len(df_global_iter)
        if n_lines == 0:
            break

        p0_g = [0., 0.]; blo_g = [-np.inf, -np.inf]; bhi_g = [np.inf, np.inf]
        for i, (_, row) in enumerate(df_global_iter.iterrows()):
            is_broad = row['fwhm'] >= fwhm_median
            shrink   = broad_sigma_shrink if is_broad else narrow_sigma_shrink
            expand   = broad_sigma_expand if is_broad else narrow_sigma_expand
            p0_g.extend([row['amplitude'], row['center'], row['sigma']])
            blo_g.extend([row['amplitude'] * amp_floor_frac,
                           center_anchor[i] - center_max_shift, row['sigma'] * shrink])
            bhi_g.extend([np.inf, center_anchor[i] + center_max_shift, row['sigma'] * expand])

        gm = _build_global_model_absorption(wavelength, n_lines)
        try:
            popt_g, pcov_g = curve_fit(gm, wavelength, residual,
                                        p0=p0_g, sigma=sigma, absolute_sigma=True,
                                        bounds=(blo_g, bhi_g), maxfev=20000)
            perr_g = np.sqrt(np.diag(pcov_g))
        except Exception:
            break

        updated = []
        for i, (_, row) in enumerate(df_global_iter.iterrows()):
            idx = 2 + i * 3
            ag, cg, sg = popt_g[idx], popt_g[idx+1], popt_g[idx+2]
            dchi2 = _local_delta_chi2_absorption(
                wavelength, flux, sigma, cg, ag, sg,
                halfwin=150.0, continuum_degree=local_continuum_degree
            )
            updated.append({**row.to_dict(),
                             'center': cg,  'center_err': perr_g[idx+1],
                             'amplitude': ag, 'amplitude_err': perr_g[idx],
                             'sigma': sg,    'sigma_err': perr_g[idx+2],
                             'fwhm': sg * SIGMA_TO_FWHM,
                             'flux': ag * sg * np.sqrt(2 * np.pi),
                             'global_delta_chi2': dchi2})

        df_res = pd.DataFrame(updated).reset_index(drop=True)
        mask_dchi2 = df_res['global_delta_chi2'] >= global_delta_chi2_threshold
        mask_fwhm  = df_res['fwhm'].between(fwhm_min, fwhm_max)
        mask_keep  = mask_dchi2 & mask_fwhm

        # 混叠剔除（吸收线保窄去宽）
        blend_rm = set()
        surv = df_res[mask_keep]; sidx = list(surv.index)
        for ii in range(len(sidx)):
            for jj in range(ii + 1, len(sidx)):
                ri, rj = surv.loc[sidx[ii]], surv.loc[sidx[jj]]
                if abs(ri['center'] - rj['center']) < blend_k * min(ri['fwhm'], rj['fwhm']):
                    blend_rm.add(sidx[ii] if ri['fwhm'] > rj['fwhm'] else sidx[jj])
        if blend_rm:
            mask_keep = mask_keep & ~df_res.index.isin(blend_rm)

        n_removed = (~mask_keep).sum()
        if n_removed > 0:
            df_global_iter = df_res[mask_keep].reset_index(drop=True)
            center_anchor  = center_anchor[mask_keep.values]
        else:
            df_global_iter = df_res; break

    if verbose:
        print(f"[谷] 全局精修完成: {len(df_global_iter)} 条吸收线")
    if len(df_global_iter) == 0:
        if return_intermediate:
            return _EMPTY, pd.DataFrame()
        return _EMPTY

    df_out = df_global_iter[['center','center_err','fwhm','amplitude','amplitude_err',
                              'flux','global_delta_chi2','delta_chi2','flags']].copy()
    df_out.columns = _COLS

    if return_intermediate:
        df_wf = df_windowed_copy[['center','center_err','fwhm','amplitude','amplitude_err',
                                   'flux','delta_chi2','flags']].copy()
        df_wf.columns = ['波长(Å)', '±波长误差', 'FWHM(Å)', '深度', '±深度误差',
                         '等效宽度(Jy·Å)', '局部Δχ²(窗口)', '质量标志']
        return df_out, df_wf

    return df_out


# =============================================================================
# 迭代找峰（每轮 mask 已检测的发射线）
# =============================================================================
def iterative_emission_detection(
    wavelength,
    flux,
    n_iterations=3,
    detection_params=None,
    verbose=True,
    output_format='new',
    df_troughs=None,
    pseudo_trough_threshold=2,
):
    """
    迭代发射线检测：每轮检测后将已找到的发射线区域（中心 ±3σ）从波长数组中删除，
    在剩余光谱上继续下一轮检测。

    Parameters
    ----------
    wavelength : array-like, 波长数组
    flux : array-like, 流量数组
    n_iterations : int, 迭代次数（默认 3）
    detection_params : dict, 传给 find_emission_lines 的参数（覆盖默认值）
    verbose : bool
    output_format : str, 输出格式 'new'（新规范）或 'old'（兼容旧格式）
    df_troughs : DataFrame, optional, 吸收线结果（用于伪峰标注）
    pseudo_trough_threshold : int, 触发伪峰标注的最小谷数量

    Returns
    -------
    all_results : list of DataFrame, 每次迭代的结果
        - output_format='new': 新规范格式 DataFrame
        - output_format='old': 旧格式 DataFrame（含 'emission_iteration' 列）
    all_records : list of dict, 逐线字典列表（仅 output_format='new' 时返回）
    final_wavelength : array, mask 后剩余波长
    final_flux : array, mask 后剩余流量
    """
    _default = dict(
        ivar=None, effective_snr=7,
        chebyshev_degree=None, chebyshev_min_degree=1, chebyshev_max_degree=10,
        window_width=500, window_overlap=300,
        delta_chi2_base=0.05, dynamic_threshold_factor=1000.0,
        global_delta_chi2_threshold=0.05,
        fwhm_min=3.0, fwhm_max=380.0,
        enable_density_presift=True, density_presift_threshold=0.08, density_presift_topk=10,
        blend_k=1.0,
        narrow_sigma_shrink=0.5, narrow_sigma_expand=1.5,
        broad_sigma_shrink=0.5, broad_sigma_expand=2.0,
        amp_floor_frac=0.1, center_max_shift=5.0,
        smooth_sigma=16.0, smooth_prominence_frac=0.1,
        verbose=False,
    )
    if detection_params:
        _default.update(detection_params)
    _default['verbose'] = False  # 迭代时关闭内层输出

    em_wave = np.asarray(wavelength).copy()
    em_flux = np.asarray(flux).copy()
    orig_wave = np.asarray(wavelength).copy()  # 保存原始数组用于计算 index
    orig_flux = np.asarray(flux).copy()
    all_results = []

    # 将 ivar 和 effective_snr 单独提取，以便随迭代同步 mask
    _ivar = _default.pop('ivar', None)
    _snr = _default.pop('effective_snr', None)
    
    # 优先使用 ivar
    if _ivar is not None:
        _ivar = np.asarray(_ivar)
        if _ivar.ndim == 0:
            em_ivar = None  # 标量 ivar 无意义，跳过
        else:
            em_ivar = _ivar.copy()
        em_snr = None
    elif _snr is not None:
        _snr = np.asarray(_snr)
        if _snr.ndim == 0:
            em_snr = _snr  # 标量，不需要 mask
        else:
            em_snr = _snr.copy()
        em_ivar = None
    else:
        em_snr = None
        em_ivar = None

    for it in range(n_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[峰] 发射线迭代 {it+1}/{n_iterations}  数据点: {len(em_wave)}")
            print('='*60)

        df_em = find_emission_lines(wavelength=em_wave, flux=em_flux, ivar=em_ivar, effective_snr=em_snr, **_default)

        if verbose:
            print(f"本轮检测到 {len(df_em)} 条发射线")
        if len(df_em) == 0:
            if verbose: print("未检测到新发射线，停止迭代")
            break

        all_results.append(df_em)

        if it < n_iterations - 1:
            keep = np.ones(len(em_wave), dtype=bool)
            for _, row in df_em.iterrows():
                cen = row['波长(Å)']
                sig = row['FWHM(Å)'] / SIGMA_TO_FWHM
                keep &= ~((em_wave >= cen - 3 * sig) & (em_wave <= cen + 3 * sig))
            n_masked = (~keep).sum()
            em_wave = em_wave[keep]
            em_flux = em_flux[keep]
            # 同步 mask ivar 或 effective_snr（仅当为数组时）
            if em_ivar is not None and em_ivar.ndim > 0:
                em_ivar = em_ivar[keep]
            if em_snr is not None and em_snr.ndim > 0:
                em_snr = em_snr[keep]
            if verbose:
                print(f"已 mask {len(df_em)} 条发射线（{n_masked} 个数据点），剩余 {len(em_wave)} 点")

    # ── 输出格式转换 ──────────────────────────────────────────────────────
    if output_format == 'new' and len(all_results) > 0:
        # 合并所有迭代结果
        df_old = pd.concat(all_results, ignore_index=True)
        # 转换为新格式
        df_new, all_records = format_features_catalog(
            df_old, wavelength_array=orig_wave, flux_array=orig_flux,
            feature_type='emission',
            df_troughs=df_troughs, pseudo_trough_threshold=pseudo_trough_threshold
        )
        return df_new, all_records, em_wave, em_flux
    elif output_format == 'new' and len(all_results) == 0:
        return pd.DataFrame(), [], em_wave, em_flux
    else:
        # 兼容旧格式
        return all_results, em_wave, em_flux


# =============================================================================
# 迭代找谷（每轮 mask 已检测的吸收线）
# =============================================================================
def iterative_absorption_detection(
    wavelength,
    flux,
    n_iterations=3,
    detection_params=None,
    verbose=True,
    output_format='new',
):
    """
    迭代吸收线检测：每轮检测后将已找到的吸收线区域（中心 ±3σ）从波长数组中删除，
    在剩余光谱上继续下一轮检测。

    Parameters
    ----------
    wavelength : array-like, 波长数组
    flux : array-like, 流量数组
    n_iterations : int, 迭代次数（默认 3）
    detection_params : dict, 传给 find_absorption_lines 的参数（覆盖默认值）
    verbose : bool
    output_format : str, 输出格式 'new'（新规范）或 'old'（兼容旧格式）

    Returns
    -------
    all_results : DataFrame, 所有迭代结果合并
        - output_format='new': 新规范格式 DataFrame
        - output_format='old': list of 旧格式 DataFrame
    all_records : list of dict, 逐线字典列表（仅 output_format='new' 时返回）
    final_wavelength : array, mask 后剩余波长
    final_flux : array, mask 后剩余流量
    """
    _default = dict(
        ivar=None, effective_snr=7,
        chebyshev_degree=None, chebyshev_min_degree=1, chebyshev_max_degree=10,
        window_width=100, window_overlap=60,
        delta_chi2_base=0.02, dynamic_threshold_factor=0.05,
        global_delta_chi2_threshold=0.05,  # 归一化后阈值
        fwhm_min=5.0, fwhm_max=100.0,
        enable_density_presift=False, density_presift_threshold=0.08, density_presift_topk=10,
        blend_k=1, snr_depth_threshold=3.0,
        local_continuum_degree=2, local_continuum_n_iter=2, local_continuum_sigma_clip=2.0,
        narrow_sigma_shrink=0.5, narrow_sigma_expand=1.5,
        broad_sigma_shrink=0.5, broad_sigma_expand=2.0,
        amp_floor_frac=0.1, center_max_shift=5.0,
        smooth_sigma_trough=16.0, smooth_prominence_frac_trough=0.1,
        verbose=False,
    )
    if detection_params:
        _default.update(detection_params)
    _default['verbose'] = False

    ab_wave = np.asarray(wavelength).copy()
    ab_flux = np.asarray(flux).copy()
    orig_wave = np.asarray(wavelength).copy()  # 保存原始数组用于计算 index
    orig_flux = np.asarray(flux).copy()
    all_results = []

    # 将 ivar 和 effective_snr 单独提取，以便随迭代同步 mask
    _ivar = _default.pop('ivar', None)
    _snr = _default.pop('effective_snr', None)
    
    # 优先使用 ivar
    if _ivar is not None:
        _ivar = np.asarray(_ivar)
        if _ivar.ndim == 0:
            ab_ivar = None  # 标量 ivar 无意义，跳过
        else:
            ab_ivar = _ivar.copy()
        ab_snr = None
    elif _snr is not None:
        _snr = np.asarray(_snr)
        if _snr.ndim == 0:
            ab_snr = _snr  # 标量，不需要 mask
        else:
            ab_snr = _snr.copy()
        ab_ivar = None
    else:
        ab_snr = None
        ab_ivar = None

    for it in range(n_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[谷] 吸收线迭代 {it+1}/{n_iterations}  数据点: {len(ab_wave)}")
            print('='*60)

        df_ab = find_absorption_lines(wavelength=ab_wave, flux=ab_flux, ivar=ab_ivar, effective_snr=ab_snr, **_default)

        if verbose:
            print(f"本轮检测到 {len(df_ab)} 条吸收线")
        if len(df_ab) == 0:
            if verbose: print("未检测到新吸收线，停止迭代")
            break

        all_results.append(df_ab)

        if it < n_iterations - 1:
            keep = np.ones(len(ab_wave), dtype=bool)
            for _, row in df_ab.iterrows():
                cen = row['波长(Å)']
                sig = row['FWHM(Å)'] / SIGMA_TO_FWHM
                keep &= ~((ab_wave >= cen - 3 * sig) & (ab_wave <= cen + 3 * sig))
            n_masked = (~keep).sum()
            ab_wave = ab_wave[keep]
            ab_flux = ab_flux[keep]
            # 同步 mask ivar 或 effective_snr（仅当为数组时）
            if ab_ivar is not None and ab_ivar.ndim > 0:
                ab_ivar = ab_ivar[keep]
            if ab_snr is not None and ab_snr.ndim > 0:
                ab_snr = ab_snr[keep]
            if verbose:
                print(f"已 mask {len(df_ab)} 条吸收线（{n_masked} 个数据点），剩余 {len(ab_wave)} 点")

    # ── 输出格式转换 ──────────────────────────────────────────────────────
    if output_format == 'new' and len(all_results) > 0:
        # 合并所有迭代结果
        df_old = pd.concat(all_results, ignore_index=True)
        # 转换为新格式
        df_new, all_records = format_features_catalog(
            df_old, wavelength_array=orig_wave, flux_array=orig_flux,
            feature_type='absorption'
        )
        return df_new, all_records, ab_wave, ab_flux
    elif output_format == 'new' and len(all_results) == 0:
        return pd.DataFrame(), [], ab_wave, ab_flux
    else:
        # 兼容旧格式
        return all_results, ab_wave, ab_flux


# =============================================================================
# 绘图函数（手动调用）
# =============================================================================

def plot_spectrum_with_features(
    wavelength,
    flux,
    df_features,
    save_path,
    feature_type='emission',
    figsize=(12, 5),
    dpi=150,
    show_gaussian=True,
    color_flux='blue',
    color_features='red',
    alpha_line=0.7,
    title=None
):
    """
    绘制光谱曲线和检测到的特征位置。
    
    Parameters
    ----------
    wavelength : array-like
        波长数组
    flux : array-like
        流量数组
    df_features : DataFrame
        检测结果 DataFrame（旧格式，含 '波长(Å)', 'FWHM(Å)', '幅度'/'深度' 列）
    save_path : str
        图像保存路径
    feature_type : str
        'emission' 或 'absorption'
    figsize : tuple
        图像尺寸
    dpi : int
        分辨率
    show_gaussian : bool
        是否绘制高斯轮廓
    color_flux : str
        光谱曲线颜色
    color_features : str
        特征标记颜色
    alpha_line : float
        特征线透明度
    title : str, optional
        图像标题
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    wavelength = np.asarray(wavelength)
    flux = np.asarray(flux)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制原始光谱
    ax.plot(wavelength, flux, color=color_flux, linewidth=0.8, alpha=0.9, label='Spectrum')
    
    if len(df_features) > 0:
        # 兼容新旧格式列名
        wl_col = '波长(Å)' if '波长(Å)' in df_features.columns else 'wavelength'
        fwhm_col = 'FWHM(Å)' if 'FWHM(Å)' in df_features.columns else 'FWHM_A'
        
        if feature_type == 'emission':
            amp_col = '幅度' if '幅度' in df_features.columns else 'amplitude'
        else:
            amp_col = '深度' if '深度' in df_features.columns else 'amplitude'
        
        for _, row in df_features.iterrows():
            center = row[wl_col]
            fwhm = row[fwhm_col]
            amplitude = row[amp_col]
            sigma = fwhm / SIGMA_TO_FWHM
            
            # 绘制特征位置垂直线
            ax.axvline(center, color=color_features, linestyle='-', alpha=alpha_line, linewidth=1.5)
            
            # 绘制高斯轮廓
            if show_gaussian:
                wave_local = wavelength[(wavelength >= center - 3*sigma) & (wavelength <= center + 3*sigma)]
                if len(wave_local) > 0:
                    if feature_type == 'emission':
                        gaussian_profile = amplitude * np.exp(-0.5 * ((wave_local - center) / sigma) ** 2)
                    else:
                        gaussian_profile = -amplitude * np.exp(-0.5 * ((wave_local - center) / sigma) ** 2)
                    ax.plot(wave_local, gaussian_profile, color=color_features, linewidth=1.0, alpha=0.5)
    
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux')
    
    legend_elements = [
        plt.Line2D([0], [0], color=color_flux, linewidth=1, label='Spectrum'),
        plt.Line2D([0], [0], color=color_features, linewidth=1.5, label=f'{feature_type.capitalize()} lines'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Detected {feature_type} features: {len(df_features)}')
    
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return fig


def plot_continuum_fit(
    wavelength,
    flux,
    continuum,
    save_path,
    figsize=(12, 5),
    dpi=150,
    color_flux='blue',
    color_continuum='orange',
    show_residual=True,
    title=None
):
    """
    绘制光谱和 continuum 拟合结果。
    
    Parameters
    ----------
    wavelength : array-like
        波长数组
    flux : array-like
        流量数组
    continuum : array-like
        continuum 拟合值数组
    save_path : str
        图像保存路径
    figsize : tuple
        图像尺寸
    dpi : int
        分辨率
    color_flux : str
        光谱曲线颜色
    color_continuum : str
        continuum 曲线颜色
    show_residual : bool
        是否显示残差子图
    title : str, optional
        图像标题
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    wavelength = np.asarray(wavelength)
    flux = np.asarray(flux)
    continuum = np.asarray(continuum)
    
    if show_residual:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.5),
                                        gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None
    
    # 主图：光谱 + continuum
    ax1.plot(wavelength, flux, color=color_flux, linewidth=0.8, alpha=0.7, label='Spectrum')
    ax1.plot(wavelength, continuum, color=color_continuum, linewidth=1.5, label='Continuum')
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Flux')
    ax1.legend(loc='upper right')
    
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('Continuum Fitting')
    
    # 残差子图
    if show_residual and ax2 is not None:
        residual = flux - continuum
        ax2.plot(wavelength, residual, color='gray', linewidth=0.8, alpha=0.8)
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Residual')
        ax2.set_title('Residual Spectrum')
    
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return fig


def plot_residual_with_features(
    wavelength,
    residual,
    df_features,
    save_path,
    feature_type='emission',
    figsize=(12, 5),
    dpi=150,
    color_residual='gray',
    color_features='red',
    alpha_line=0.7,
    title=None
):
    """
    绘制残差光谱和检测到的特征位置。
    
    Parameters
    ----------
    wavelength : array-like
        波长数组
    residual : array-like
        残差数组（flux - continuum）
    df_features : DataFrame
        检测结果 DataFrame
    save_path : str
        图像保存路径
    feature_type : str
        'emission' 或 'absorption'
    figsize : tuple
        图像尺寸
    dpi : int
        分辨率
    color_residual : str
        残差曲线颜色
    color_features : str
        特征标记颜色
    alpha_line : float
        特征线透明度
    title : str, optional
        图像标题
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    wavelength = np.asarray(wavelength)
    residual = np.asarray(residual)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制残差光谱
    ax.plot(wavelength, residual, color=color_residual, linewidth=0.8, alpha=0.8)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    if len(df_features) > 0:
        # 兼容新旧格式列名
        wl_col = '波长(Å)' if '波长(Å)' in df_features.columns else 'wavelength'
        
        for _, row in df_features.iterrows():
            center = row[wl_col]
            ax.axvline(center, color=color_features, linestyle='-', alpha=alpha_line, linewidth=1.5)
    
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Residual Flux')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Residual Spectrum with {feature_type} features: {len(df_features)}')
    
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return fig


def plot_feature_detection_summary(
    wavelength,
    flux,
    continuum,
    df_emission,
    df_absorption,
    save_path,
    figsize=(14, 8),
    dpi=150,
    show_gaussian=True,
    title=None
):
    """
    绘制特征检测综合图（光谱 + continuum + 发射线 + 吸收线）。
    
    Parameters
    ----------
    wavelength : array-like
        波长数组
    flux : array-like
        流量数组
    continuum : array-like
        continuum 拟合值数组
    df_emission : DataFrame
        发射线检测结果
    df_absorption : DataFrame
        吸收线检测结果
    save_path : str
        图像保存路径
    figsize : tuple
        图像尺寸
    dpi : int
        分辨率
    show_gaussian : bool
        是否绘制高斯轮廓
    title : str, optional
        图像标题
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    wavelength = np.asarray(wavelength)
    flux = np.asarray(flux)
    continuum = np.asarray(continuum)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]),
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    # ===== 上图：光谱 + continuum + 特征标记 =====
    ax1.plot(wavelength, flux, color='blue', linewidth=0.8, alpha=0.7, label='Spectrum')
    ax1.plot(wavelength, continuum, color='orange', linewidth=1.2, alpha=0.8, label='Continuum')
    
    # 绘制发射线（红色）
    if len(df_emission) > 0:
        wl_col = '波长(Å)' if '波长(Å)' in df_emission.columns else 'wavelength'
        fwhm_col = 'FWHM(Å)' if 'FWHM(Å)' in df_emission.columns else 'FWHM_A'
        amp_col = '幅度' if '幅度' in df_emission.columns else 'amplitude'
        
        for _, row in df_emission.iterrows():
            center = row[wl_col]
            ax1.axvline(center, color='red', linestyle='-', alpha=0.6, linewidth=1.2)
            
            if show_gaussian:
                fwhm = row[fwhm_col]
                amplitude = row[amp_col]
                sigma = fwhm / SIGMA_TO_FWHM
                wave_local = wavelength[(wavelength >= center - 3*sigma) & (wavelength <= center + 3*sigma)]
                if len(wave_local) > 0:
                    gaussian_profile = amplitude * np.exp(-0.5 * ((wave_local - center) / sigma) ** 2)
                    ax1.plot(wave_local, gaussian_profile, color='red', linewidth=0.8, alpha=0.4)
    
    # 绘制吸收线（蓝色）
    if len(df_absorption) > 0:
        wl_col = '波长(Å)' if '波长(Å)' in df_absorption.columns else 'wavelength'
        fwhm_col = 'FWHM(Å)' if 'FWHM(Å)' in df_absorption.columns else 'FWHM_A'
        amp_col = '深度' if '深度' in df_absorption.columns else 'amplitude'
        
        for _, row in df_absorption.iterrows():
            center = row[wl_col]
            ax1.axvline(center, color='blue', linestyle='-', alpha=0.6, linewidth=1.2)
            
            if show_gaussian:
                fwhm = row[fwhm_col]
                amplitude = row[amp_col]
                sigma = fwhm / SIGMA_TO_FWHM
                wave_local = wavelength[(wavelength >= center - 3*sigma) & (wavelength <= center + 3*sigma)]
                if len(wave_local) > 0:
                    gaussian_profile = -amplitude * np.exp(-0.5 * ((wave_local - center) / sigma) ** 2)
                    ax1.plot(wave_local, gaussian_profile, color='blue', linewidth=0.8, alpha=0.4)
    
    ax1.set_ylabel('Flux')
    
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=1, label='Spectrum'),
        plt.Line2D([0], [0], color='orange', linewidth=1.2, label='Continuum'),
        plt.Line2D([0], [0], color='red', linewidth=1.2, label=f'Emission ({len(df_emission)})'),
        plt.Line2D([0], [0], color='blue', linewidth=1.2, label=f'Absorption ({len(df_absorption)})'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title(f'Feature Detection: {len(df_emission)} emission + {len(df_absorption)} absorption lines')
    
    # ===== 下图：残差 + 特征位置 =====
    residual = flux - continuum
    ax2.plot(wavelength, residual, color='gray', linewidth=0.6, alpha=0.8)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    # 标记特征位置
    if len(df_emission) > 0:
        wl_col = '波长(Å)' if '波长(Å)' in df_emission.columns else 'wavelength'
        for _, row in df_emission.iterrows():
            ax2.axvline(row[wl_col], color='red', linestyle='-', alpha=0.5, linewidth=1)
    
    if len(df_absorption) > 0:
        wl_col = '波长(Å)' if '波长(Å)' in df_absorption.columns else 'wavelength'
        for _, row in df_absorption.iterrows():
            ax2.axvline(row[wl_col], color='blue', linestyle='-', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Residual')
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return fig
