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

        file_name = state.get("file_name", "unknown")

        self._write_rule_analysis(state, output_dir, file_name)
        self._write_summary(state, output_dir, file_name)
        self._write_in_brief(state, output_dir, file_name)
        self._write_snapshot(state, output_dir, file_name)

    def write_qualitative_analysis(self, state: SpectroState) -> None:
        """写出 qualitative_analysis.txt（在 qualitative_analysis 步骤结束后调用）"""
        output_dir, file_name = self._resolve_output_dir(state), state.get("file_name", "unknown")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{file_name}_qualitative_analysis.txt")
        content = json.dumps(state.get("qualitative_analysis") or {}, indent=2, ensure_ascii=False)
        self._write_text(path, content)

    def write_preliminary_classification(self, state: SpectroState) -> None:
        """写出 preliminary_classification.txt（在 qualitative_analysis 步骤结束后调用）"""
        output_dir, file_name = self._resolve_output_dir(state), state.get("file_name", "unknown")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{file_name}_preliminary_classification.txt")
        with open(path, "w", encoding=self.encoding) as f:
            for key in ("preliminary_classification", "preliminary_classification_monkey"):
                value = state.get(key)
                if value is None:
                    continue
                f.write(f"{'='*60}\n")
                f.write(f"  {key.upper()}\n")
                f.write(f"{'='*60}\n")
                f.write(value if isinstance(value, str)
                        else json.dumps(value, indent=2, ensure_ascii=False))
                f.write("\n\n")

    def write_rule_analysis_qso(self, state: SpectroState) -> None:
        """写出 rule_analysis_QSO.txt（在 QSO quantitative_analysis 结束后调用）"""
        output_dir, file_name = self._resolve_output_dir(state), state.get("file_name", "unknown")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{file_name}_rule_analysis_QSO.txt")
        with open(path, "w", encoding=self.encoding) as f:
            step_f_raw = (state.get("rule_analysis_QSO") or {}).get("step_F")
            if step_f_raw:
                f.write(f"{'='*60}\n  STEP_F (RAW SYNTHESIS)\n{'='*60}\n")
                f.write(step_f_raw if isinstance(step_f_raw, str)
                        else json.dumps(step_f_raw, indent=2, ensure_ascii=False))
                f.write("\n\n")

    def write_rule_analysis_lrg(self, state: SpectroState) -> None:
        """写出 rule_analysis_LRG.txt（在 LRG/BGS quantitative_analysis 结束后调用）"""
        e = state.get("rule_analysis_LRG") or {}
        extract = state.get("extract_LRG") or {}
        if not e and not extract:
            return
        output_dir, file_name = self._resolve_output_dir(state), state.get("file_name", "unknown")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{file_name}_rule_analysis_LRG.txt")
        with open(path, "w", encoding=self.encoding) as f:
            # raw step outputs
            for key, value in e.items():
                f.write(f"{'='*60}\n  {key.upper()} (RAW)\n{'='*60}\n")
                f.write(value if isinstance(value, str)
                        else json.dumps(value, indent=2, ensure_ascii=False))
                f.write("\n\n")
            # structured extract
            step_f_extract = extract.get("step_F")
            if step_f_extract:
                f.write(f"{'='*60}\n  STEP_F (STRUCTURED EXTRACT)\n{'='*60}\n")
                f.write(json.dumps(step_f_extract, indent=2, ensure_ascii=False))
                f.write("\n\n")

    def write_f_a_summaries(
        self, state: SpectroState, summaries: list, label: str = "QSO"
    ) -> None:
        """
        将 Step F-a 各假设的结构化摘要写出为 .txt 文件。
        文件名格式：{file_name}_{label}_f_a_summaries.txt
        每条摘要以 JSON 格式缩进输出，并附有序号标题。

        Parameters
        ----------
        state    : SpectroState
        summaries: list of dict  — QSO/ELG/LRG_step_F_extract 返回的结果列表
        label    : str           — 文件名中的类型标签，如 "QSO"、"ELG"、"LRG"
        """
        if not summaries:
            return
        output_dir = self._resolve_output_dir(state)
        os.makedirs(output_dir, exist_ok=True)
        file_name = state.get("file_name", "unknown")
        path = os.path.join(output_dir, f"{file_name}_{label}_f_a_summaries.txt")
        total = len(summaries)
        with open(path, "w", encoding=self.encoding) as f:
            f.write(f"{'='*60}\n  {label} STEP F-a SUMMARIES  ({total} hypotheses)\n{'='*60}\n\n")
            for idx, summary in enumerate(summaries, start=1):
                f.write(f"--- Hypothesis #{idx} / {total} ---\n")
                f.write(json.dumps(summary, indent=2, ensure_ascii=False))
                f.write("\n\n")

    def write_brute_force_matching(self, state: SpectroState) -> None:
        """
        将 brute_force_matching（QSO/ELG 模式）和 brute_force_matching_lrg_bgs（LRG/BGS 模式）
        分别写出为两个 .txt 文件。
        文件名格式：
          {file_name}_brute_force_matching.txt
          {file_name}_brute_force_matching_lrg_bgs.txt
        """
        output_dir = self._resolve_output_dir(state)
        os.makedirs(output_dir, exist_ok=True)
        file_name = state.get("file_name", "unknown")

        _matching_items = (
            ("brute_force_matching_qso_elg",         "BRUTE FORCE MATCHING (QSO / ELG)"),
            ("brute_force_matching_lrg_bgs", "BRUTE FORCE MATCHING (LRG / BGS)"),
        )
        for state_key, title in _matching_items:
            data = state.get(state_key)
            if not data:
                continue
            suffix = state_key  # 直接用 state key 作为文件名后缀
            path = os.path.join(output_dir, f"{file_name}_{suffix}.txt")
            with open(path, "w", encoding=self.encoding) as f:
                f.write(f"{'='*60}\n  {title}\n{'='*60}\n\n")
                for idx, entry in enumerate(data, start=1):
                    f.write(f"--- Match #{idx} ---\n")
                    for key, value in entry.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")

    def write_verdict(self, state: SpectroState) -> None:
        """写出 auditing_verdict.txt（在 AnalysisAuditor.auditing_verdict 结束后调用）"""
        verdict = state.get('verdict')
        if not verdict:
            return
        output_dir, file_name = self._resolve_output_dir(state), state.get('file_name', 'unknown')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{file_name}_verdict.txt")
        with open(path, 'w', encoding=self.encoding) as f:
            f.write(f"{'='*60}\n  AUDITING VERDICT\n{'='*60}\n")
            f.write(verdict if isinstance(verdict, str)
                    else json.dumps(verdict, indent=2, ensure_ascii=False))
            f.write('\n')

    def write_critique(self, state: SpectroState) -> None:
        """写出 auditing_critique.txt（在 AnalysisAuditor.auditing_critique 结束后调用）"""
        critique = state.get('critique')
        if not critique:
            return
        output_dir, file_name = self._resolve_output_dir(state), state.get('file_name', 'unknown')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{file_name}_auditing_critique.txt")
        with open(path, 'w', encoding=self.encoding) as f:
            f.write(f"{'='*60}\n  AUDITING CRITIQUE\n{'='*60}\n\n")
            f.write(critique if isinstance(critique, str)
                    else json.dumps(critique, indent=2, ensure_ascii=False))
            f.write('\n')

    def write_patched_verdict(self, state: SpectroState) -> None:
        """写出 refining_patch.txt（在 RefinementAssistant.refining_patch 结束后调用）"""
        patched = state.get('patched_verdict')
        if not patched:
            return
        output_dir, file_name = self._resolve_output_dir(state), state.get('file_name', 'unknown')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{file_name}_refining_patch.txt")
        with open(path, 'w', encoding=self.encoding) as f:
            f.write(f"{'='*60}\n  REFINING PATCH (PATCHED VERDICT)\n{'='*60}\n\n")
            f.write(patched if isinstance(patched, str)
                    else json.dumps(patched, indent=2, ensure_ascii=False))
            f.write('\n')

    def write_final_report(self, state: SpectroState) -> None:
        """写出 final_report.txt（在 RefinementAssistant.refining_final 结束后调用）"""
        report = state.get('final_report')
        if not report:
            return
        output_dir, file_name = self._resolve_output_dir(state), state.get('file_name', 'unknown')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{file_name}_final_report.txt")
        with open(path, 'w', encoding=self.encoding) as f:
            f.write(f"{'='*60}\n  FINAL ANALYSIS REPORT\n{'='*60}\n\n")
            f.write(report)
            f.write('\n')

    def write_verdict_extract(self, state: SpectroState) -> None:
        """写出 verdict_extract.txt（在 AnalysisAuditor.verdict_extract 结束后调用）"""
        items = state.get('verdict_extract')
        if not items:
            return
        output_dir, file_name = self._resolve_output_dir(state), state.get('file_name', 'unknown')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{file_name}_verdict_extract.txt")
        with open(path, 'w', encoding=self.encoding) as f:
            f.write(f"{'='*60}\n  VERDICT EXTRACT  ({len(items)} item(s))\n{'='*60}\n\n")
            for i, item in enumerate(items, 1):
                f.write(f"--- Result #{i} ---\n")
                f.write(f"  Source_path      : {item.get('Source_path', '?')}\n")
                f.write(f"  Physical_type    : {item.get('Physical_type', '?')}\n")
                f.write(f"  Suggested_redshift: {item.get('Suggested_redshift', '?')}\n")
                f.write(f"  Confidence       : {item.get('Confidence', '?')}\n")
                f.write(f"  Hypothesis       : {item.get('Hypothesis', '?')}\n")
                pairs = item.get('Adopted_pairs') or []
                if pairs:
                    f.write("  Adopted_pairs    :\n")
                    for p in pairs:
                        f.write(f"    {p.get('line','?')} → {p.get('obs_wavelength','?')} Å  (z={p.get('z','?')})\n")
                f.write(f"  Key_evidence     : {item.get('Key_evidence', '?')}\n")
                doubts = item.get('Remaining_doubts') or []
                f.write(f"  Remaining_doubts : {'; '.join(doubts) if doubts else 'none'}\n")
                f.write("\n")

    def write_rule_analysis_elg(self, state: SpectroState) -> None:
        """写出 rule_analysis_ELG.txt（在 ELG quantitative_analysis 结束后调用）"""
        e = state.get("rule_analysis_ELG") or {}
        extract = state.get("extract_ELG") or {}
        if not e and not extract:
            return
        output_dir, file_name = self._resolve_output_dir(state), state.get("file_name", "unknown")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{file_name}_rule_analysis_ELG.txt")
        with open(path, "w", encoding=self.encoding) as f:
            # raw step outputs
            for key, value in e.items():
                f.write(f"{'='*60}\n  {key.upper()} (RAW)\n{'='*60}\n")
                f.write(value if isinstance(value, str)
                        else json.dumps(value, indent=2, ensure_ascii=False))
                f.write("\n\n")
            # structured extract
            step_f_extract = extract.get("step_F")
            if step_f_extract:
                f.write(f"{'='*60}\n  STEP_F (STRUCTURED EXTRACT)\n{'='*60}\n")
                f.write(json.dumps(step_f_extract, indent=2, ensure_ascii=False))
                f.write("\n\n")

    # =========================
    # 📄 Artifact Writers
    # =========================

    def _write_rule_analysis(self, state: SpectroState, output_dir: str, file_name: str):
        # rule_analysis_QSO is now a dict written directly by RuleAnalyst.py as .txt files.
        # This .md output is superseded; skip to avoid empty files.
        return

    def _write_summary(self, state: SpectroState, output_dir: str, file_name: str):
        path = os.path.join(output_dir, f"{file_name}_summary.md")
        if not self._can_write(path):
            return

        summary = state.get("summary")
        if not summary:
            return

        self._write_text(path, summary)

    def _write_in_brief(self, state: SpectroState, output_dir: str, file_name: str):
        path = os.path.join(output_dir, f"{file_name}_in_brief.json")
        if not self._can_write(path):
            return

        in_brief = state.get("in_brief")
        if not isinstance(in_brief, dict):
            return

        payload = {
            "file_name": file_name,
            "timestamp": self._now(),
            "in_brief": in_brief,
        }

        self._write_json(path, payload)

    def _write_snapshot(self, state: SpectroState, output_dir: str, file_name: str):
        """
        保存一个安全的 state 快照（用于 debug / 复现）
        """
        path = os.path.join(output_dir, f"{file_name}_snapshot.json")
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
            "file_name": state.get("file_name"),
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
