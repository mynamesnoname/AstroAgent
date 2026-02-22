from typing import Any, List, Union
from pydantic import BaseModel
import numpy as np


def calculate_redshift(
    obs_wavelength: Union[float, List[float]],
    rest_wavelength: Union[float, List[float]],
) -> Union[float, List[float]]:
    """
    z = (λ_obs / λ_rest) - 1
    """
    if isinstance(obs_wavelength, (int, float)):
        return obs_wavelength / rest_wavelength - 1

    if len(obs_wavelength) != len(rest_wavelength):
        raise ValueError("obs_wavelength and rest_wavelength must have the same length")

    return [o / r - 1 for o, r in zip(obs_wavelength, rest_wavelength)]


def predict_obs_wavelength(
    redshift: Union[float, List[float]],
    rest_wavelength: Union[float, List[float]],
) -> Union[float, List[float]]:
    """
    λ_obs = λ_rest · (1 + z)
    """
    if isinstance(redshift, (int, float)):
        return rest_wavelength * (1 + redshift)

    if len(redshift) != len(rest_wavelength):
        raise ValueError("redshift and rest_wavelength must have the same length")

    return [r * (1 + z) for z, r in zip(redshift, rest_wavelength)]


def qso_redshift_rms(
    wavelength_rest: float,
    a: float,            # Å / pixel
    tolerance: int,      # pixel tolerance
    rms_lambda: float,   # Å
) -> float:
    """
    σ_z = sqrt((a·t)^2 + σ_λ^2) / λ_rest
    """
    return np.sqrt((a * tolerance) ** 2 + rms_lambda ** 2) / wavelength_rest

# def _calculate_redshift(obs_wavelength: Union[float, List[float]], rest_wavelength: Union[float, List[float]]) -> Union[float, List[float]]:
#     """根据观测波长和谱线本征波长计算红移。"""
#     try:
#         if isinstance(obs_wavelength, (int, float)):
#             z = obs_wavelength / rest_wavelength - 1
#         else:
#             z = [o / r - 1 for o, r in zip(obs_wavelength, rest_wavelength)]
#         return z
#     except Exception as e:
#         return {"error": str(e)}

# def _predict_obs_wavelength(redshift: Union[float, List[float]], rest_wavelength: Union[float, List[float]]) -> Union[float, List[float]]:
#     """根据红移和谱线本征波长预测观测波长。"""
#     try:
#         if isinstance(redshift, (int, float)):
#             obs = rest_wavelength * (1 + redshift)
#         else:
#             obs = [r * (1 + z) for z, r in zip(redshift, rest_wavelength)]
#         return obs
#     except Exception as e:
#         return {"error": str(e)}
    
# def _QSO_rms(wavelength_rest: float, 
#     a: float,            # wavelength per pixel (Å/pix)
#     tolerance: int,      # 像素容差 t
#     rms_lambda: float    # 拟合波长 rms
#     ):
#     rms = np.sqrt((a * tolerance)**2 + rms_lambda**2) / wavelength_rest
#     return rms

# class WeightedResult(BaseModel):
#     weighted_average: float
#     weighted_std: float

# def _weighted_average_with_error(redshift: List[float], flux: List[float], a: float, tolerance: int) -> dict:
#     if not redshift or not flux:
#         raise ValueError("Input lists cannot be empty")
#     if len(redshift) != len(flux):
#         raise ValueError("Length mismatch")
    
#     total_weight = sum(flux)
#     if total_weight == 0:
#         raise ValueError("Sum of weights cannot be zero")
    
#     avg = sum(z * w for z, w in zip(redshift, flux)) / total_weight
#     # 简单标准差计算
#     variance = sum(w * (z - avg)**2 for z, w in zip(redshift, flux)) / total_weight
#     std = variance**0.5
    
#     result = WeightedResult(weighted_average=avg, weighted_std=std)
#     return result.dict()  # 返回 dict，LLM 直接读取即可

# def _galaxy_weighted_average_with_error(
#     wavelength_obs: List[float],
#     wavelength_rest: List[float],
#     flux: List[float],
#     a: float,            # wavelength per pixel (Å/pix)
#     tolerance: int,      # 像素容差 t
#     rms_lambda: float    # 拟合波长 rms
# ) -> dict:
#     """
#     计算加权红移及误差
    
#     参数:
#     - wavelength_obs: 观测波长 λ_obs
#     - wavelength_rest: 静止系波长 λ_rest
#     - flux: 每条谱线 flux，用作权重
#     - a: 像素到波长比例 (Å/pix)
#     - tolerance: 像素容差 t
#     - rms_lambda: 拟合波长残差 (Å)
    
#     返回:
#     - dict 包含 weighted_average 和 weighted_std
#     """
#     # 输入检查
#     if not wavelength_obs or not wavelength_rest or not flux:
#         raise ValueError("Input lists cannot be empty")
#     if len(wavelength_obs) != len(wavelength_rest) or len(wavelength_obs) != len(flux):
#         raise ValueError("Length mismatch among wavelength_obs, wavelength_rest, flux")

#     wavelength_obs = np.array(wavelength_obs, dtype=float)
#     wavelength_rest = np.array(wavelength_rest, dtype=float)
#     flux = np.array(flux, dtype=float)

#     # 单条谱线红移
#     redshifts = wavelength_obs / wavelength_rest - 1.0

#     # 单条谱线波长不确定度
#     sigma_lambda = np.sqrt((a * tolerance)**2 + rms_lambda**2)
#     sigma_z = sigma_lambda / wavelength_rest  # 每条线的 z 误差

#     # 加权平均红移 (flux 作为权重)
#     total_weight = np.sum(flux)
#     if total_weight == 0:
#         raise ValueError("Sum of flux weights cannot be zero")

#     weighted_avg = np.sum(redshifts * flux) / total_weight

#     # flux 权重的加权标准差
#     weighted_var = np.sum((flux**2) * (sigma_z**2)) / (total_weight**2)
#     weighted_std = np.sqrt(weighted_var)

#     result = WeightedResult(weighted_average=weighted_avg, weighted_std=weighted_std)
#     return result.dict()  # 返回 dict，LLM 可直接读取

