import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.ndimage import gaussian_filter1d
from AstroAgent.agents.common.state import SpectroState
from AstroAgent.agents.multi_agents.utils.usage import safe_to_bool


# ===========================================================
# Plotting
# ===========================================================

def _get_figsize(state: SpectroState):
    """
    获取绘图尺寸。
    对于图像输入，根据图像尺寸计算；
    对于 FITS 输入，使用默认尺寸。
    """
    # 检查是否为 FITS 格式输入
    file_path = state.get('file_path', '')
    if file_path.lower().endswith('.fits'):
        # FITS 格式使用默认尺寸
        return (12, 4)
    
    # 图像格式：根据图像尺寸计算
    img = cv2.imread(file_path)
    if img is None:
        # 如果无法读取图像，返回默认尺寸
        return (12, 4)
    h, w = img.shape[:2]
    # 对h和w反复除以10，直到二者中的一个第一次小于10
    while h > 10 and w > 10:
        h //= 10
        w //= 10
    return (w, h)


def plot_spectrum_snr(state: SpectroState):
    """
    绘制双子图：上方为光谱 flux（含不确定性填充），下方为 Effective SNR。
    保存为 {file_name}_spectrum.png，并返回 fig。
    """
    wavelength = state['spectrum']['new_wavelength']
    flux = state['spectrum']['weighted_flux']
    flux_top = state['spectrum']['max_unresolved_flux']
    flux_bottom = state['spectrum']['min_unresolved_flux']
    effective_snr = state['spectrum']['effective_snr']

    h, w = _get_figsize(state)

    fig, axs = plt.subplots(2, 1, figsize=(h, 2*w))

    axs[0].plot(wavelength, flux, color='b', label=r'$\bar F$: signal extracted from picture')
    axs[0].fill_between(wavelength, flux_top, flux_bottom, alpha=0.4, color='gray', label='information lossed in Opencv processing')
    axs[0].set_ylabel('flux')
    axs[0].set_xlabel('wavelength')
    axs[0].legend()

    axs[1].plot(wavelength, effective_snr, c='orange', label=r'$SNR=\frac{{\bar F}_i}{\sigma_{i,j}}$')
    axs[1].set_ylabel('Effective SNR')
    axs[1].set_xlabel('wavelength')
    axs[1].legend(fontsize=15)

    fig.savefig(os.path.join(state['output_dir'], f"{state['file_name']}_spectrum.png"), bbox_inches='tight')
    return fig


def plot_spec_extract(state: SpectroState):
    """
    绘制单子图：光谱 flux 曲线 + 信息损失区域填充（来自 OpenCV 处理）。
    保存为 {file_name}_spec_extract.png。
    """
    wavelength = state['spectrum']['new_wavelength']
    flux = state['spectrum']['weighted_flux']
    flux_top = state['spectrum'].get('max_unresolved_flux', [])
    flux_bottom = state['spectrum'].get('min_unresolved_flux', [])

    h, w = _get_figsize(state)

    fig = plt.figure(figsize=(h, w))
    if flux_top and flux_bottom:
        plt.fill_between(wavelength, flux_bottom, flux_top,
                         color='#FFB6A6', alpha=0.5, linewidth=0,
                         label='information lost in OpenCV processing (pink #FFB6A6)')
    plt.plot(wavelength, flux, color='b', lw=1.5,
             label=r'$\bar{F}_i$: signal extracted from picture (blue)')
    plt.xlabel('wavelength')
    plt.ylabel('flux')
    plt.legend(fontsize=12)
    plt.savefig(
        os.path.join(state['output_dir'], f"{state['file_name']}_spec_extract.png"),
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()
    return fig


def plot_continuum(state: SpectroState):
    """
    绘制 continuum 曲线，保存到 state['continuum_path']。
    """
    continuum_wavelength = np.array(state['continuum']['wavelength'])
    continuum_flux = np.array(state['continuum']['flux'])

    h, w = _get_figsize(state)

    fig = plt.figure(figsize=(h, w))
    plt.plot(continuum_wavelength, continuum_flux, color='orange', label='Continuum')
    plt.xlabel('wavelength')
    plt.ylabel('flux')
    plt.legend(fontsize=12)
    plt.savefig(
        state['continuum_path'],
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()
    return fig

def plot_residual_spectrum(state: SpectroState):
    """
    绘制 residual_spectrum 曲线。
    """
    residual_spectrum_wavelength = np.array(state['residual_spectrum']['wavelength'])
    residual_spectrum_flux = np.array(state['residual_spectrum']['flux'])

    h, w = _get_figsize(state)

    fig = plt.figure(figsize=(h, w))
    plt.plot(residual_spectrum_wavelength, residual_spectrum_flux, color='orange', label='residual_spectrum')
    plt.xlabel('wavelength')
    plt.ylabel('flux')
    plt.legend(fontsize=12)
    path = os.path.join(state['output_dir'], f"{state['file_name']}_residual_spectrum.png")
    plt.savefig(
        path,
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()
    return fig


def plot_masked_spectrum(state: SpectroState):
    """
    绘制原始光谱与 masked 后的光谱对比图。
    保存为 {file_name}_masked_spectrum.png。
    """
    spec = state.get("spectrum", {})
    wavelengths = spec.get("new_wavelength", [])
    flux = spec.get("weighted_flux", [])

    h, w = _get_figsize(state)

    fig = plt.figure(figsize=(h, w))
    plt.plot(wavelengths, flux, label="Original Spectrum", alpha=0.5)
    plt.plot(
        state["cleaned_spectrum"]["wavelength"],
        state["cleaned_spectrum"]["flux"],
        label="Masked Spectrum",
        alpha=0.9
    )
    plt.xlabel('wavelength')
    plt.ylabel('flux')
    plt.legend(fontsize=12)
    path = os.path.join(
        state["output_dir"],
        f"{state['file_name']}_masked_spectrum.png"
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_cleaned_features(state: SpectroState, sigma_list: List, wavelength_label: bool=False):
    """
    绘制清洗后的峰谷特征图。
    包含原始光谱、不同 sigma 平滑曲线、峰值和谷值标记线。
    保存为 {file_name}_features.png。
    """
    h, w = _get_figsize(state)

    fig = plt.figure(figsize=(h, w))
    spec = state["spectrum"]
    wavelengths = np.array(spec["new_wavelength"])
    flux = np.array(spec["weighted_flux"])

    # 首先绘制 overlap_regions（最底层）
    overlap_regions = state.get('overlap_regions')
    if overlap_regions:
        y_min, y_max = np.min(flux), np.max(flux)
        for region_name, (left, right) in overlap_regions.items():
            plt.axvspan(left, right, color='gray', alpha=0.2, zorder=0)
        # 添加一个虚拟的 patch 用于 legend
        from matplotlib.patches import Patch
        overlap_patch = Patch(color='gray', alpha=0.2, label='overlapped arms')

    plt.plot(wavelengths, flux, label='original', c='k', alpha=0.7, zorder=1)

    for sigma in sigma_list:
        sigma_smooth = gaussian_filter1d(state['spectrum']['weighted_flux'], sigma=sigma)
        plt.plot(state['spectrum']['new_wavelength'], sigma_smooth, alpha=0.7, label=rf'$\sigma={sigma}$', zorder=1)

    # 获取 y 轴范围用于文本定位
    y_min, y_max = plt.ylim()
    text_y_position = y_max * 0.7  # 在顶部 98% 位置

    # 绘制 cleaned_peaks（红色实线）
    for peak_ in state.get('cleaned_peaks', []):
        plt.axvline(peak_['wavelength'], linestyle='-', c='red', alpha=0.5, zorder=2)
        if wavelength_label:
            plt.text(peak_['wavelength'], text_y_position, f'{peak_["wavelength"]:.2f}',
                     rotation=90, verticalalignment='top', horizontalalignment='center')

    # 绘制 cleaned_troughs（蓝色实线）
    for trough_ in state.get('troughs', []):
        if trough_['wavelength'] > 0:
            plt.axvline(trough_['wavelength'], linestyle='-', c='blue', alpha=0.5, zorder=2)
            if wavelength_label:
                plt.text(trough_['wavelength'], text_y_position, f'{trough_["wavelength"]:.2f}',
                         rotation=90, verticalalignment='top', horizontalalignment='center')

    # 绘制 wiped_peaks（红色虚线）
    for peak_ in state.get('wiped_peaks', []):
        plt.axvline(peak_['wavelength'], linestyle='--', c='red', alpha=0.3, zorder=2)

    # 绘制 wiped_troughs（蓝色虚线）
    for trough_ in state.get('wiped_troughs', []):
        if trough_['wavelength'] > 0:
            plt.axvline(trough_['wavelength'], linestyle='--', c='blue', alpha=0.3, zorder=2)

    # 添加 legend 条目
    plt.plot([], [], linestyle='-', c='red', alpha=0.5, label='peaks')
    plt.plot([], [], linestyle='-', c='blue', alpha=0.5, label='troughs')
    if state.get('wiped_peaks') or state.get('wiped_troughs'):
        plt.plot([], [], linestyle='--', c='red', alpha=0.3, label='supposed peaks')
        plt.plot([], [], linestyle='--', c='blue', alpha=0.3, label='supposed troughs')
    if overlap_regions:
        plt.plot([], [], color='gray', alpha=0.2, linewidth=10, label='overlapped arms')

    plt.ylabel('flux')
    plt.xlabel('wavelength')
    plt.legend(ncol=2)
    print(f"Plot {len(state.get('cleaned_peaks', []))} peaks, {len(state.get('cleaned_troughs', []))} troughs, "
          f"{len(state.get('wiped_peaks', []))} supposed peaks, {len(state.get('wiped_troughs', []))} supposed troughs.")

    fig.savefig(os.path.join(state['output_dir'], f"{state['file_name']}_features.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig