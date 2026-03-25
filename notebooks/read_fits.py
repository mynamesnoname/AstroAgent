#!/usr/bin/env python3
"""
读取 DESI FITS 文件，提取并拼接 B/R/Z 三个波段的光谱数据。
"""

import os
import numpy as np
from astropy.io import fits
from astropy.table import Table


def read_and_merge_spectra(path: str, output_path: str):
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
            print(f"重叠波长 {current_wave:.2f}: 来自 {sources_str}, 平均 flux = {avg_flux:.4f}")
        
        # 跳过已处理的相同波长点
        i += np.sum(same_wave_mask)
    
    unique_wavelengths = np.array(unique_wavelengths)
    merged_fluxes = np.array(merged_fluxes)
    
    print(f"\n合并后光谱: {len(unique_wavelengths)} 个点")
    print(f"波长范围: [{unique_wavelengths[0]:.2f}, {unique_wavelengths[-1]:.2f}]")
    
    # 创建输出 FITS 文件
    # 创建二进制表
    col1 = fits.Column(name='WAVELENGTH', format='D', array=unique_wavelengths)
    col2 = fits.Column(name='FLUX', format='D', array=merged_fluxes)
    hdu = fits.BinTableHDU.from_columns([col1, col2])
    
    # 添加 header 信息
    hdu.header['COMMENT'] = 'Merged DESI spectrum from B/R/Z bands'
    hdu.header['COMMENT'] = 'Overlapping wavelengths averaged'
    hdu.header['NAXIS1'] = len(unique_wavelengths)
    
    # 写入文件
    hdu.writeto(output_path, overwrite=True)
    print(f"\n已保存到: {output_path}")


def main():
    # 输入文件路径
    input_path = '/home/wbc/code3/llm-spectro-agent_advance/test_set/DESI/all_spectra_per_wave.fits'
    
    # 输出文件路径
    output_dir = '/home/wbc/code3/llm-spectro-agent_advance/data/input'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'test.fits')
    
    # 执行读取和合并
    read_and_merge_spectra(input_path, output_path)


if __name__ == '__main__':
    main()
