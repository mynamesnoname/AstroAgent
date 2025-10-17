# src/tools.py
import json
from typing import Any, List, Union

def _calculate_redshift(obs_wavelength: Union[float, List[float]], rest_wavelength: Union[float, List[float]]) -> dict:
    """根据观测波长和谱线本征波长计算红移。"""
    try:
        if isinstance(obs_wavelength, (int, float)):
            z = obs_wavelength / rest_wavelength - 1
        else:
            z = [o / r - 1 for o, r in zip(obs_wavelength, rest_wavelength)]
        return z
    except Exception as e:
        return {"error": str(e)}

def _predict_obs_wavelength(redshift: Union[float, List[float]], rest_wavelength: Union[float, List[float]]) -> dict:
    """根据红移和谱线本征波长预测观测波长。"""
    try:
        if isinstance(redshift, (int, float)):
            obs = rest_wavelength * (1 + redshift)
        else:
            obs = [r * (1 + z) for z, r in zip(redshift, rest_wavelength)]
        return obs
    except Exception as e:
        return {"error": str(e)}
    
def _weighted_average(redshift: List[float], flux: List[float]) -> float:
    """
    计算加权平均红移值
    
    Args:
        redshift: 红移值列表
        flux: 对应的置信度列表（作为权重）
    
    Returns:
        加权平均红移值
    """
    if len(redshift) != len(flux):
        raise ValueError("redshift and flux must have the same length")
    
    if not redshift:
        raise ValueError("Input lists cannot be empty")
    
    # 计算加权总和
    weighted_sum = sum(z * conf for z, conf in zip(redshift, flux))
    
    # 计算权重总和
    total_weight = sum(flux)
    
    if total_weight == 0:
        raise ValueError("Sum of confidence levels cannot be zero")
    
    return weighted_sum / total_weight