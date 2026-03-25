import os
import numpy as np
from astropy.io import fits
from astropy.table import Table

def read_and_merge_spectra(path: str):
    """
    读取 DESI FITS 文件，提取第一条光谱的 B/R/Z 波段数据，
    拼接并处理重叠区域，保存为新的 FITS 文件。
    
    Args:
        path: 输入 FITS 文件路径
        output_path: 输出 FITS 文件路径
    """
    
    # 读取 FITS 文件
    with fits.open(path) as hdul:
        # 获取 SPECTRA HDU
        spectra_hdu = hdul['SPECTRA']
        spectra_data = spectra_hdu.data
        
        # 提取第一条光谱的数据
        wave_b = spectra_data['WAVE_B'][0]  # B 波段波长
        flux_b = spectra_data['FLUX_B'][0]  # B 波段流量
        
        wave_r = spectra_data['WAVE_R'][0]  # R 波段波长
        flux_r = spectra_data['FLUX_R'][0]  # R 波段流量
        
        wave_z = spectra_data['WAVE_Z'][0]  # Z 波段波长
        flux_z = spectra_data['FLUX_Z'][0]  # Z 波段流量
    
    print(f"B band: {len(wave_b)} points, range [{wave_b[0]:.2f}, {wave_b[-1]:.2f}]")
    print(f"R band: {len(wave_r)} points, range [{wave_r[0]:.2f}, {wave_r[-1]:.2f}]")
    print(f"Z band: {len(wave_z)} points, range [{wave_z[0]:.2f}, {wave_z[-1]:.2f}]")
    
    # 拼接三个波段的数据
    all_wavelengths = []
    all_fluxes = []
    all_sources = []  # 用于追踪每个点来自哪个波段，处理重叠
    
    # 添加 B 波段
    for w, f in zip(wave_b, flux_b):
        all_wavelengths.append(w)
        all_fluxes.append(f)
        all_sources.append('B')
    
    # 添加 R 波段
    for w, f in zip(wave_r, flux_r):
        all_wavelengths.append(w)
        all_fluxes.append(f)
        all_sources.append('R')
    
    # 添加 Z 波段
    for w, f in zip(wave_z, flux_z):
        all_wavelengths.append(w)
        all_fluxes.append(f)
        all_sources.append('Z')
    
    # 转换为 numpy 数组
    all_wavelengths = np.array(all_wavelengths)
    all_fluxes = np.array(all_fluxes)
    all_sources = np.array(all_sources)
    
    # 按波长排序
    sort_idx = np.argsort(all_wavelengths)
    all_wavelengths = all_wavelengths[sort_idx]
    all_fluxes = all_fluxes[sort_idx]
    all_sources = all_sources[sort_idx]
    
    # 处理重叠区域：对相同波长的 flux 取平均
    unique_wavelengths = []
    merged_fluxes = []
    
    i = 0
    while i < len(all_wavelengths):
        current_wave = all_wavelengths[i]
        # 找到所有相同波长的点
        same_wave_mask = (all_wavelengths == current_wave)
        same_wave_fluxes = all_fluxes[same_wave_mask]
        same_wave_sources = all_sources[same_wave_mask]
        
        # 取平均
        avg_flux = np.mean(same_wave_fluxes)
        
        unique_wavelengths.append(current_wave)
        merged_fluxes.append(avg_flux)
        
        if len(same_wave_sources) > 1:
            sources_str = ', '.join(same_wave_sources)
            # print(f"重叠波长 {current_wave:.2f}: 来自 {sources_str}, 平均 flux = {avg_flux:.4f}")
        
        # 跳过已处理的相同波长点
        i += np.sum(same_wave_mask)
    
    unique_wavelengths = np.array(unique_wavelengths)
    merged_fluxes = np.array(merged_fluxes)

    return unique_wavelengths, merged_fluxes

def complete_the_spectra(wavelength, flux):
    DECAM_z_start = 8250.0
    DECAM_z_end = 10150
    if wavelength[-1] < DECAM_z_end:
        # Append a point at DECAM_z_end with flux 0
        # 对[DECAM_z_start, wavelength[-1]]区间，使用 power law，通过拟合得到 wavelength = DECAM_z_end 时对应的 flux
        slope = np.polyfit(np.log(wavelength), np.log(flux), 1)[0]
        wavelength = np.append(wavelength, DECAM_z_end)
        flux = np.append(flux, np.exp(np.polyval([slope, 0], np.log(DECAM_z_end))))
        return wavelength, flux
