import json
import asyncio
import logging

from AstroAgent.agents.common.state import SpectroState
from AstroAgent.agents.common.base_agent import BaseAgent
from AstroAgent.core.runtime.runtime_container import RuntimeContainer


class SynthesisHost(BaseAgent):
    """
    综合主持人：从 refining_final 报告和 verdict_extract 中提取结构化摘要信息
    Synthesis Host: Extract structured summary from final_report & verdict_extract
    """
    agent_name = "SynthesisHost"

    def __init__(self, runtime: RuntimeContainer):
        super().__init__(runtime)

    async def run(self, state: SpectroState) -> SpectroState:
        """并发提取 primary + 2nd 两个 JSON"""
        if state.get('in_brief') is None:
            state['in_brief'] = {}

        if not state.get('final_report'):
            print("[SynthesisHost] No final_report available, skipping extraction.")
            return state

        primary_result, second_result = await asyncio.gather(
            self._extract_primary(state),
            self._extract_2nd(state),
            return_exceptions=True,
        )

        # ── 写入 primary 字段 ──
        if isinstance(primary_result, Exception):
            logging.warning(f"[SynthesisHost] _extract_primary failed: {primary_result}")
            for key in ('type', 'score', 'redshift', 'redshift_rms', 'lines', 'human'):
                state['in_brief'][key] = None
        elif isinstance(primary_result, dict):
            state['in_brief']['type'] = primary_result.get('type')
            try:
                state['in_brief']['score'] = int(primary_result.get('score')) if primary_result.get('score') is not None else None
            except (ValueError, TypeError):
                state['in_brief']['score'] = None
            try:
                state['in_brief']['redshift'] = float(primary_result.get('redshift')) if primary_result.get('redshift') is not None else None
            except (ValueError, TypeError):
                state['in_brief']['redshift'] = None
            try:
                state['in_brief']['redshift_rms'] = float(primary_result.get('redshift_rms')) if primary_result.get('redshift_rms') is not None else None
            except (ValueError, TypeError):
                state['in_brief']['redshift_rms'] = None
            state['in_brief']['lines'] = primary_result.get('lines')
            state['in_brief']['human'] = primary_result.get('human')
        else:
            for key in ('type', 'score', 'redshift', 'redshift_rms', 'lines', 'human'):
                state['in_brief'][key] = None

        # ── 写入 2nd 字段 ──
        if isinstance(second_result, Exception):
            logging.warning(f"[SynthesisHost] _extract_2nd failed: {second_result}")
            for key in ('type_2nd', 'redshift_2nd', 'redshift_rms_2nd', 'lines_2nd'):
                state['in_brief'][key] = None
        elif isinstance(second_result, dict):
            state['in_brief']['type_2nd'] = second_result.get('type_2nd')
            try:
                state['in_brief']['redshift_2nd'] = float(second_result.get('redshift_2nd')) if second_result.get('redshift_2nd') is not None else None
            except (ValueError, TypeError):
                state['in_brief']['redshift_2nd'] = None
            try:
                state['in_brief']['redshift_rms_2nd'] = float(second_result.get('redshift_rms_2nd')) if second_result.get('redshift_rms_2nd') is not None else None
            except (ValueError, TypeError):
                state['in_brief']['redshift_rms_2nd'] = None
            state['in_brief']['lines_2nd'] = second_result.get('lines_2nd')
        else:
            for key in ('type_2nd', 'redshift_2nd', 'redshift_rms_2nd', 'lines_2nd'):
                state['in_brief'][key] = None

        print("[SynthesisHost] in_brief extraction complete.")
        return state

    # =========================
    # 提取函数
    # =========================

    async def _extract_primary(self, state: SpectroState):
        """从 final_report 一次性提取 type/score/redshift/redshift_rms/lines/human"""
        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name="in_brief_primary",
            final_report=state['final_report'],
        )
        result = await self.call_llm_with_context(
            system_prompt, user_prompt,
            parse_json=True,
            description="in_brief_primary",
            want_tools=False,
        )
        return result

    async def _extract_2nd(self, state: SpectroState):
        """从 verdict_extract 提取第2顺位裁决信息"""
        verdict_extract = state.get('verdict_extract')
        if not verdict_extract or len(verdict_extract) < 2:
            return {"type_2nd": None, "redshift_2nd": None, "redshift_rms_2nd": None, "lines_2nd": None}

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name="in_brief_2nd",
            verdict_extract=verdict_extract,
        )
        result = await self.call_llm_with_context(
            system_prompt, user_prompt,
            parse_json=True,
            description="in_brief_2nd",
            want_tools=False,
        )
        return result
