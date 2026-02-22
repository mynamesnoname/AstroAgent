# tool_protocol.py
from typing import Any, Optional, Dict
from pydantic import BaseModel


class ToolError(BaseModel):
    type: str                 # 错误类型（InvalidInput / RuntimeError / etc）
    message: str              # 给 LLM / 人看的错误信息
    hint: Optional[str] = None  # 可选：如何修正输入


class ToolResult(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[ToolError] = None
