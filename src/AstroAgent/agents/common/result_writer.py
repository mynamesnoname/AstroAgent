# src/result_writer.py

import os
import json
from typing import Optional, Dict, Any
from datetime import datetime

from .state import SpectroState


class ResultWriter:
    """
    ResultWriter
    -------------
    将一次 workflow 的 SpectroState 写成可消费的结果文件。
    - 只读 state
    - 不修改分析逻辑
    - 可安全在循环中重复调用
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        overwrite: bool = True,
        encoding: str = "utf-8"
    ):
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.encoding = encoding

    # =========================
    # 🚀 Public API
    # =========================

    def write(self, state: SpectroState) -> None:
        """
        写出一次 workflow 的所有结果
        """
        if not state:
            return

        output_dir = self._resolve_output_dir(state)
        os.makedirs(output_dir, exist_ok=True)

        image_name = state.get("image_name", "unknown")

        self._write_rule_analysis(state, output_dir, image_name)
        self._write_summary(state, output_dir, image_name)
        self._write_in_brief(state, output_dir, image_name)
        self._write_snapshot(state, output_dir, image_name)

    # =========================
    # 📄 Artifact Writers
    # =========================

    def _write_rule_analysis(self, state: SpectroState, output_dir: str, image_name: str):
        path = os.path.join(output_dir, f"{image_name}_rule_analysis.md")
        if not self._can_write(path):
            return

        qso = state.get("rule_analysis_QSO") or []
        gal = state.get("rule_analysis_galaxy") or []

        blocks = []

        if qso:
            blocks.append("## QSO Rule Analysis\n")
            blocks.extend(self._stringify_list(qso))

        if gal:
            blocks.append("\n## Galaxy Rule Analysis\n")
            blocks.extend(self._stringify_list(gal))

        content = "\n\n".join(blocks).strip()

        if content:
            self._write_text(path, content)

    def _write_summary(self, state: SpectroState, output_dir: str, image_name: str):
        path = os.path.join(output_dir, f"{image_name}_summary.md")
        if not self._can_write(path):
            return

        summary = state.get("summary")
        if not summary:
            return

        self._write_text(path, summary)

    def _write_in_brief(self, state: SpectroState, output_dir: str, image_name: str):
        path = os.path.join(output_dir, f"{image_name}_in_brief.json")
        if not self._can_write(path):
            return

        in_brief = state.get("in_brief")
        if not isinstance(in_brief, dict):
            return

        payload = {
            "image_name": image_name,
            "timestamp": self._now(),
            "in_brief": in_brief,
        }

        self._write_json(path, payload)

    def _write_snapshot(self, state: SpectroState, output_dir: str, image_name: str):
        """
        保存一个安全的 state 快照（用于 debug / 复现）
        """
        path = os.path.join(output_dir, f"{image_name}_snapshot.json")
        if not self._can_write(path):
            return

        snapshot = self._serialize_state(state)
        self._write_json(path, snapshot)

    # =========================
    # 🧰 Helpers
    # =========================

    def _resolve_output_dir(self, state: SpectroState) -> str:
        return self.output_dir or state.get("output_dir") or "outputs"

    def _can_write(self, path: str) -> bool:
        return self.overwrite or not os.path.exists(path)

    def _write_text(self, path: str, content: str):
        with open(path, "w", encoding=self.encoding) as f:
            f.write(content)

    def _write_json(self, path: str, obj: Dict[str, Any]):
        with open(path, "w", encoding=self.encoding) as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _stringify_list(self, items):
        return [str(x) for x in items if x is not None]

    def _serialize_state(self, state: SpectroState) -> Dict[str, Any]:
        """
        只序列化 JSON-safe 的字段
        """
        safe = {}

        for k, v in state.items():
            if self._is_json_safe(v):
                safe[k] = v
            else:
                safe[k] = f"<non-serializable: {type(v).__name__}>"

        return {
            "image_name": state.get("image_name"),
            "timestamp": self._now(),
            "state": safe,
        }

    def _is_json_safe(self, v: Any) -> bool:
        try:
            json.dumps(v)
            return True
        except Exception:
            return False

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"
