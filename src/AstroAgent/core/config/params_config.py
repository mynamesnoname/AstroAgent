import os

from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from AstroAgent.core.config._utils import getenv_int, getenv_int_list, getenv_float, getenv_optional_float, safe_to_bool


# ------------------------
# 分析参数配置
# ------------------------

class ParamsConfig(BaseModel):
    dataset: str
    arm_name: Optional[List[str]]
    arm_wavelength_range: Optional[List[List[float]]]
    snr_threshold_upper: Optional[float]
    snr_threshold_lower: Optional[float]
    ocr: str
    continuum_smoothing: int
    sigma_list: List[int]
    tol_pixels: int
    prom_threshold_peaks: float
    prom_threshold_troughs: float
    num_peaks: int
    num_troughs: int
    label: bool
    max_debate_rounds: int

    @classmethod
    def from_env(cls) -> "ParamsConfig":
        arm_name = cls._parse_arm_name()
        arm_wavelength_range = cls._parse_arm_wavelength_range()

        # -------- 核心一致性校验 --------
        if arm_name is None and arm_wavelength_range is None:
            pass  # ✅ 允许
        elif arm_name is None or arm_wavelength_range is None:
            raise ValueError(
                "ARM_NAME and ARM_WAVELENGTH_RANGE must both be set or both be None."
            )
        elif len(arm_name) != len(arm_wavelength_range):
            raise ValueError(
                "ARM_NAME and ARM_WAVELENGTH_RANGE must have the same length."
            )

        return cls(
            dataset=os.getenv("DATASET") or "DESI",
            arm_name=arm_name,
            arm_wavelength_range=arm_wavelength_range,
            snr_threshold_upper=getenv_optional_float("SNR_THRESHOLD_UPPER"),
            snr_threshold_lower=getenv_optional_float("SNR_THRESHOLD_LOWER"),
            ocr=os.getenv("OCR") or "paddle",
            continuum_smoothing=getenv_int("CONTINUUM_SMOOTHING", 100),
            sigma_list=getenv_int_list("SIGMA_LIST", [2, 4, 16]),
            tol_pixels=getenv_int("TOL_PIXELS", 10),
            prom_threshold_peaks=getenv_float("PROM_THRESHOLD_PEAKS", 0.01),
            prom_threshold_troughs=getenv_float("PROM_THRESHOLD_TROUGHS", 0.05),
            num_peaks=getenv_int("PEAKS_NUMBER", 10),
            num_troughs=getenv_int("TROUGHS_NUMBER", 15),
            label=safe_to_bool(os.getenv("LABEL")),
            max_debate_rounds=getenv_int("MAX_DEBATE_ROUNDS", 3),
        )

    # ------------------------
    # 解析 ARM_NAME
    # ------------------------
    @staticmethod
    def _parse_arm_name():
        raw = os.getenv("ARM_NAME")
        if raw is None or raw.strip() == "":
            return None
        return [name.strip() for name in raw.split(",")]

    # ------------------------
    # 解析 ARM_WAVELENGTH_RANGE
    # ------------------------
    @staticmethod
    def _parse_arm_wavelength_range():
        raw = os.getenv("ARM_WAVELENGTH_RANGE")
        if raw is None or raw.strip() == "":
            return None

        result = []
        for rng in raw.split(","):
            parts = rng.strip().split("-")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid wavelength range format: {rng}"
                )
            try:
                result.append([float(parts[0]), float(parts[1])])
            except ValueError:
                raise ValueError(
                    f"Invalid numeric wavelength range: {rng}"
                )

        return result