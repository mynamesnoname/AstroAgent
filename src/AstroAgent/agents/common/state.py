import json
import numpy as np
from dataclasses import field
from typing import Any, Dict, Optional, List, Union
from matplotlib.figure import Figure
import re
from langgraph.graph import MessagesState

# @dataclass
class SpectroState(MessagesState):
    """
    LangGraph Agent 的光谱上下文状态。
    支持：
    - 自动 JSON 解析
    - 安全 set / append 操作
    - 自动创建未声明字段或列表
    """

    # ===========================
    # 🔹 原始输入
    # ===========================
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    output_dir: Optional[str] = None
    crop_path: Optional[str] = None
    spec_extract_path: Optional[str] = None
    continuum_path: Optional[str] = None

    # ===========================
    # 🔹 图像识别与像素映射信息
    # ===========================
    axis_info: Optional[dict] = None
    OCR_detected_ticks: Optional[List[Dict[str, Union[float, int, List[int]]]]] = None
    tick_pixel_raw: Optional[List[Dict[str, Union[int, float, None]]]] = None
    chart_border: Optional[dict] = None
    tick_pixel_remap: Optional[List[Dict[str, Union[int, float, None]]]] = None
    pixel_to_value: Optional[dict] = None

    # ===========================
    # 🔹 光谱提取与特征数据
    # ===========================
    curve_points: Optional[List[List[int]]] = None
    curve_gray_values: Optional[Union[List[float], np.ndarray]] = None
    spectrum: Optional[Dict[str, Union[List[float], float]]] = None
    peak_groups: Optional[List[List[Dict[str, Any]]]] = None
    trough_groups: Optional[List[List[Dict[str, Any]]]] = None
    cleaned_spectrum: Optional[Dict[str, Union[List[float], float]]] = None
    continuum: Optional[Dict[str, Any]] = None
    residual_spectrum: Optional[Dict[str, Union[List[float], float]]] = None
    approved_peaks: Optional[List[Dict[str, float]]] = None
    approved_troughs: Optional[List[Dict[str, float]]] = None
    peaks: Optional[List[Dict[str, float]]] = None  
    troughs: Optional[List[Dict[str, float]]] = None
    # ROI_peaks: Optional[List[Dict[str, float]]] = None
    # ROI_troughs: Optional[List[Dict[str, float]]] = None
    # merged_peaks: Optional[List[Dict[str, float]]] = None
    # merged_troughs: Optional[List[Dict[str, float]]] = None
    cleaned_peaks: Optional[List[Dict[str, float]]] = None
    wiped_peaks: Optional[List[Dict[str, float]]] = None
    cleaned_troughs: Optional[List[Dict[str, float]]] = None
    overlap_regions: Optional[Dict[str, List[float]]] = None
    # ===========================
    # 🔹 可视化对象
    # ===========================
    spectrum_fig: Optional[Figure] = None
    features_fig: Optional[Figure] = None

    # ===========================
    # 🔹 LLM 解释与分析历史
    # ===========================
    # visual_interpretation: Optional[List] = field(default=None)
    qualitative_analysis: Optional[Dict[str, Any]] = field(default_factory=dict)
    preliminary_classification: Optional[str] = None
    preliminary_classification_with_absention: Optional[str] = None
    preliminary_classification_monkey: Optional[str] = None
    possible_object: Optional[List] = field(default=None)
    Lyalpha_candidate: Optional[List] = field(default=None)
    rule_analysis_QSO: Optional[List] = field(default_factory=list)
    rule_analysis_galaxy: Optional[List] = field(default_factory=list)
    # other Analysts
    auditing_history_QSO: Optional[List] = field(default_factory=list)
    refining_history_QSO: Optional[List] = field(default_factory=list)
    auditing_history_galaxy: Optional[List] = field(default_factory=list)
    refining_history_galaxy: Optional[List] = field(default_factory=list)
    # rule_analysis: Optional[List] = field(default_factory=list)
    debate_rounds: Optional[int] = None
    count: Optional[int] = None
    # auditing_history: Optional[List] = field(default_factory=list)
    # refine_history: Optional[List] = field(default_factory=list)
    summary: Optional[str] = None
    in_brief: Optional[Dict[str, float]] = None