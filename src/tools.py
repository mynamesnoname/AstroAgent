from typing import Any, List, Union
from pydantic import BaseModel

def _calculate_redshift(obs_wavelength: Union[float, List[float]], rest_wavelength: Union[float, List[float]]) -> Union[float, List[float]]:
    """根据观测波长和谱线本征波长计算红移。"""
    try:
        if isinstance(obs_wavelength, (int, float)):
            z = obs_wavelength / rest_wavelength - 1
        else:
            z = [o / r - 1 for o, r in zip(obs_wavelength, rest_wavelength)]
        return z
    except Exception as e:
        return {"error": str(e)}

def _predict_obs_wavelength(redshift: Union[float, List[float]], rest_wavelength: Union[float, List[float]]) -> Union[float, List[float]]:
    """根据红移和谱线本征波长预测观测波长。"""
    try:
        if isinstance(redshift, (int, float)):
            obs = rest_wavelength * (1 + redshift)
        else:
            obs = [r * (1 + z) for z, r in zip(redshift, rest_wavelength)]
        return obs
    except Exception as e:
        return {"error": str(e)}

class WeightedResult(BaseModel):
    weighted_average: float
    weighted_std: float

def _weighted_average_with_error(redshift: List[float], flux: List[float]) -> dict:
    if not redshift or not flux:
        raise ValueError("Input lists cannot be empty")
    if len(redshift) != len(flux):
        raise ValueError("Length mismatch")
    
    total_weight = sum(flux)
    if total_weight == 0:
        raise ValueError("Sum of weights cannot be zero")
    
    avg = sum(z * w for z, w in zip(redshift, flux)) / total_weight
    # 简单标准差计算
    variance = sum(w * (z - avg)**2 for z, w in zip(redshift, flux)) / total_weight
    std = variance**0.5
    
    result = WeightedResult(weighted_average=avg, weighted_std=std)
    return result.dict()  # 返回 dict，LLM 直接读取即可


# def _weighted_average(redshift: List[float], flux: List[float]) -> float:
#     """
#     计算加权平均红移值
    
#     Args:
#         redshift: 红移值列表
#         flux: 对应的流量列表（作为权重）
    
#     Returns:
#         加权平均红移值
#     """
#     if len(redshift) != len(flux):
#         raise ValueError("redshift and flux must have the same length")
    
#     if not redshift:
#         raise ValueError("Input lists cannot be empty")
    
#     # 计算加权总和
#     weighted_sum = sum(z * conf for z, conf in zip(redshift, flux))
    
#     # 计算权重总和
#     total_weight = sum(flux)
    
#     if total_weight == 0:
#         raise ValueError("Sum of weight cannot be zero")
    
#     return weighted_sum / total_weight