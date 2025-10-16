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
