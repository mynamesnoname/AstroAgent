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
    image_path: Optional[str] = None
    image_name: Optional[str] = None
    output_dir: Optional[str] = None
    crop_path: Optional[str] = None
    spec_extract_path: Optional[str] = None
    continuum_path: Optional[str] = None
    sigma_list: List[Union[int, float]] = field(default_factory=list)
    band_name: Optional[List[str]] = None
    band_wavelength: Optional[List[List[float]]] = None
    prompt: Optional[dict] = None

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
    peaks: Optional[List[Dict[str, float]]] = None
    troughs: Optional[List[Dict[str, float]]] = None
    ROI_peaks: Optional[List[Dict[str, float]]] = None
    ROI_troughs: Optional[List[Dict[str, float]]] = None
    merged_peaks: Optional[List[Dict[str, float]]] = None
    merged_troughs: Optional[List[Dict[str, float]]] = None
    continuum: Optional[Dict[str, Any]] = None
    cleaned_peaks: Optional[List[Dict[str, float]]] = None
    wiped_peaks: Optional[List[Dict[str, float]]] = None
    cleaned_troughs: Optional[List[Dict[str, float]]] = None
    # ===========================
    # 🔹 可视化对象
    # ===========================
    spectrum_fig: Optional[Figure] = None
    features_fig: Optional[Figure] = None

    # ===========================
    # 🔹 LLM 解释与分析历史
    # ===========================
    visual_interpretation: Optional[List] = field(default=None)
    preliminary_classification: Optional[str] = None
    preliminary_classification_with_confusion: Optional[str] = None
    possible_object: Optional[List] = field(default=None)
    Lyalpha_candidate: Optional[List] = field(default=None)
    rule_analysis_QSO: Optional[List] = field(default_factory=list)
    rule_analysis_galaxy: Optional[List] = field(default_factory=list)
    # other Analysts
    auditing_history_QSO: Optional[List] = field(default_factory=list)
    refine_history_QSO: Optional[List] = field(default_factory=list)
    auditing_history_galaxy: Optional[List] = field(default_factory=list)
    refine_history_galaxy: Optional[List] = field(default_factory=list)
    # rule_analysis: Optional[List] = field(default_factory=list)
    debate_rounds: Optional[int] = None
    count: Optional[int] = None
    # auditing_history: Optional[List] = field(default_factory=list)
    # refine_history: Optional[List] = field(default_factory=list)
    summary: Optional[str] = None
    in_brief: Optional[Dict[str, float]] = None

    # =====================================================
    # 🔧 通用方法
    # =====================================================
    def _parse_value(self, value: Any) -> Any:
        """自动解析 JSON 字符串"""
        if isinstance(value, str):
            cleaned = re.sub(r'```json\s*|\s*```', '', value).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return value
        return value

    def set(self, name: str, value: Any) -> str:
        """
        安全设置（覆盖或创建）变量。
        - 自动解析 JSON
        - 自动创建未声明字段
        """
        value = self._parse_value(value)

        try:
            setattr(self, name, value)
            return f"✅ Set variable '{name}' to {value!r}."
        except Exception as e:
            return f"⚠️ Failed to set '{name}': {e}"

    def append(self, name: str, value: Any) -> str:
        """
        安全追加元素。
        - 自动创建新字段为 list
        - 若字段为 None，则初始化为空 list
        - 若非 list 字段，则返回警告
        """
        value = self._parse_value(value)

        # 字段不存在 → 创建
        if not hasattr(self, name):
            setattr(self, name, [value])
            return f"✅ Created list '{name}' and appended {value!r}."

        current = getattr(self, name)
        # None → 初始化为 list
        if current is None:
            current = []
            setattr(self, name, current)

        # 非 list → 报警
        if not isinstance(current, list):
            return f"⚠️ Cannot append to non-list field '{name}'. Current type: {type(current).__name__}"

        current.append(value)
        setattr(self, name, current)
        return f"✅ Appended {value!r} to '{name}'."

