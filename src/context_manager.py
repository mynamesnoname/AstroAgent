import json
import numpy as np
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, List, Union

@dataclass
class SpectroContext:
    """
    存储视觉和分析阶段的中间结果。
    """
    # 原始输入
    image_path: Optional[str] = None
    crop_path: Optional[str] = None

    # 使用 field() 提供更灵活的默认值
    axis_info: Optional[dict] = field(default=None)
    OCR_detected_ticks: Optional[List[Dict[str, Union[float, int, List[int]]]]] = field(default=None)
    tick_pixel_raw: List[Dict[str, Union[int, float, None]]] = field(default=None)
    chart_border: Optional[dict] = field(default=None)
    tick_pixel_remap: List[Dict[str, Union[int, float, None]]] = field(default=None)
    pixel_to_value: Optional[dict] = field(default=None)
    curve_points: Optional[List[list[int]]] = field(default=None)
    curve_gray_values: Union[List[float], np.ndarray] = field(default=None)
    spectrum: Dict[str, Union[List[float], float]] = field(default=None)
    peaks: List[dict[float]] = field(default=None)
    troughs: List[dict[float]] = field(default=None)

    visual_interpretation: Optional[str] = field(default=None)
    preliminary_classification: Optional[str] = field(default=None)

    # 历史记录
    rule_analysis: Optional[dict] = field(default=None)

    def set(self, name: str, value: Any):
        """
        动态设置变量。若输入是 JSON 字符串，自动解析。
        支持 dataclass 字段和额外变量。
        """
        # 1. 尝试解析 JSON 字符串
        if isinstance(value, str):
            # 清理 Markdown 代码块标记
            import re
            cleaned_value = re.sub(r'```json\s*|\s*```', '', value).strip()
            
            try:
                value = json.loads(cleaned_value)
            except json.JSONDecodeError:
                # 如果不是 JSON，保持原字符串
                pass

        # 2. 检查是否为 dataclass 字段
        if hasattr(self, name) and name in [field.name for field in fields(self)]:
            # 直接设置 dataclass 字段
            setattr(self, name, value)
            return f"Variable '{name}' stored in context."
        else:
            return f"Variable '{name}' not found in context."
        
    def append(self, name: str, value: Any):
        """
        动态添加变量。若输入是 JSON 字符串，自动解析。
        支持 dataclass 字段和额外变量。
        """
        # 1. 尝试解析 JSON 字符串
        if isinstance(value, str):
            # 清理 Markdown 代码块标记
            import re
            cleaned_value = re.sub(r'```json\s*|\s*```', '', value).strip()
            
            try:
                value = json.loads(cleaned_value)
            except json.JSONDecodeError:
                # 如果不是 JSON，保持原字符串
                pass

        # 2. 检查是否为 dataclass 字段
        if hasattr(self, name) and name in [field.name for field in fields(self)]:
            # 如果 name 对应的列表是 None，初始化为空列表
            if getattr(self, name) is None:
                setattr(self, name, [])
            # 添加新值到列表
            getattr(self, name).append(value)
            return f"Variable '{name}' stored in context."
        else:
            return f"Variable '{name}' not found in context."
        
        

