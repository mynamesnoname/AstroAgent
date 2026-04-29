import json
import os
import numpy as np
import logging

from AstroAgent.agents.common.state import SpectroState
from AstroAgent.agents.common.base_agent import BaseAgent
from AstroAgent.agents.common.result_writer import ResultWriter
from AstroAgent.core.runtime.runtime_container import RuntimeContainer


class AnalysisAuditor(BaseAgent):
    """
    审查分析师：跨类型综合裁决，从 QSO/ELG/LRG 路径的定量分析摘要中选出最符合物理语义的结论
    AnalysisAuditor: Cross-type verdict — selects the most physically consistent
    classification from QSO/ELG/LRG extract summaries.
    """
    agent_name = "AnalysisAuditor"

    def __init__(self, runtime: RuntimeContainer):
        super().__init__(runtime)
        self._writer = ResultWriter()

    @staticmethod
    def _all_paths_inconclusive(state: SpectroState) -> bool:
        """Return True when every extract path has only null-Hypothesis entries (or is absent)."""
        for key in ('extract_QSO', 'extract_ELG', 'extract_LRG'):
            items = (state.get(key) or {}).get('step_F') or []
            for item in items:
                if item.get('Hypothesis') is not None:
                    return False   # at least one valid hypothesis found
        return True

    async def run(self, state: SpectroState) -> SpectroState:
        if self._all_paths_inconclusive(state):
            print("[AnalysisAuditor] All paths inconclusive — skipping verdict/critique/patch.")
            return state
        await self.auditing_verdict(state)
        self._writer.write_verdict(state)
        # print(f"Auditing verdict: {state['verdict']}")
        await self.verdict_extract(state)
        # print(f"Verdict extract: {state['verdict_extract']}")
        await self.auditing_critique(state)
        self._writer.write_critique(state)
        return state

    async def auditing_verdict(self, state: SpectroState) -> SpectroState:
        """Cross-type verdict: select the best hypothesis from extract_QSO/ELG/LRG."""
        function_name = "auditing_verdict"

        continuum_description = state['continuum']['description']
        feature_description = state['qualitative_analysis']['lines']
        wl_left = state['spectrum']['new_wavelength'][0]
        wl_right = state['spectrum']['new_wavelength'][-1]
        peaks = state['peaks']
        troughs = state['troughs']

        # Collect per-path extracts; use the step_F key produced by generic_extract.
        # If a path was not run, the dict will be empty — pass None so the template
        # can detect the absence cleanly.
        def _get_extract(state_key: str):
            data = state.get(state_key) or {}
            step_f = data.get('step_F')
            return step_f if step_f else None

        extract_QSO = _get_extract('extract_QSO')
        extract_ELG = _get_extract('extract_ELG')
        extract_LRG = _get_extract('extract_LRG')

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            continuum_description=continuum_description,
            feature_description=feature_description,
            wl_left=wl_left,
            wl_right=wl_right,
            extract_QSO=extract_QSO,
            extract_ELG=extract_ELG,
            extract_LRG=extract_LRG,
            peaks=peaks,
            troughs=troughs,
        )

        result = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parse_json=False,
            description="Auditing verdict (cross-type)",
            want_tools=False,
        )

        state['verdict'] = result
        print(f"Auditing verdict:\n{result}")
        self._writer.write_verdict(state)
        return state

    async def verdict_extract(self, state: SpectroState) -> SpectroState:
        """Extract structured List[Dict] from the free-text auditing_verdict output."""
        function_name = "verdict_extract"

        if not state.get('verdict'):
            print("[verdict_extract] verdict is empty, skipping.")
            return state

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            verdict=state['verdict'],
        )

        result = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parse_json=True,
            description="Verdict extract (structured)",
            want_tools=False,
        )

        # result may be a list directly or wrapped in a dict — normalise
        if isinstance(result, list):
            state['verdict_extract'] = result
        elif isinstance(result, dict):
            # some LLMs wrap the array under a key; try common keys first
            for key in ("result", "data", "items", "verdicts"):
                if key in result and isinstance(result[key], list):
                    state['verdict_extract'] = result[key]
                    break
            else:
                state['verdict_extract'] = [result]   # single-element fallback
        else:
            state['verdict_extract'] = []

        # ── Post-processing filter ──────────────────────────────────────────
        # Drop the 2nd entry when it is not genuinely distinct from the 1st:
        #   • Confidence of 2nd is "low", OR
        #   • |Δz| < 0.05  (same physical interpretation, different bookkeeping)
        VE = state['verdict_extract']
        if len(VE) == 2:
            z1 = VE[0].get("Suggested_redshift") or 0.0
            z2 = VE[1].get("Suggested_redshift") or 0.0
            c2 = (VE[1].get("Confidence") or "low").lower()
            drop = False
            reason = ""
            if c2 == "low":
                drop, reason = True, f"2nd Confidence=low"
            elif abs(z1 - z2) < 0.05:
                drop, reason = True, f"|Δz|={abs(z1-z2):.3f} < 0.05"
            if drop:
                print(f"  [filter] Dropping 2nd verdict entry ({reason})")
                state['verdict_extract'] = VE[:1]
        # ───────────────────────────────────────────────────────────────────

        print(f"Verdict extract ({len(state['verdict_extract'])} item(s)):")
        for i, item in enumerate(state['verdict_extract'], 1):
            print(f"  [{i}] {item.get('Source_path','?')} | {item.get('Confidence','?')} | z={item.get('Suggested_redshift','?')}")
        return state

    async def auditing_critique(self, state: SpectroState) -> SpectroState:
        """Critique the top-1 verdict: surface physical loopholes and weak points."""
        function_name = "auditing_critique"

        ve = state.get('verdict_extract') or []
        if not ve:
            print("[auditing_critique] verdict_extract is empty, skipping.")
            return state

        primary_verdict   = ve[0]
        secondary_verdict = ve[1] if len(ve) > 1 else None

        continuum_description = state['continuum']['description']
        feature_description   = state['qualitative_analysis']['lines']
        wl_left  = state['spectrum']['new_wavelength'][0]
        wl_right = state['spectrum']['new_wavelength'][-1]

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            continuum_description=continuum_description,
            feature_description=feature_description,
            wl_left=wl_left,
            wl_right=wl_right,
            primary_verdict=primary_verdict,
            secondary_verdict=secondary_verdict,
        )

        result = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parse_json=False,
            description="Auditing critique",
            want_tools=False,
        )

        state['critique'] = result
        print(f"Auditing critique:\n{result}")
        return state


# ============================================================================
# DISABLED / LEGACY METHODS — not called in current pipeline
# ============================================================================

# --- auditing (legacy QSO-only debate round) ---
#     async def auditing(self, state: SpectroState) -> SpectroState:
#         function_name = "auditing"
#
#         num_peaks = self.runtime.configs.params.num_peaks
#         num_troughs = self.runtime.configs.params.num_troughs
#
#         peak_list = [
#             {
#                 "wavelength": pe.get('wavelength'),
#                 "flux": pe.get('mean_flux'),
#                 "width": pe.get('width_mean'),
#                 "width_in_km_s": pe.get('width_in_km_s'),
#                 "prominance": pe.get('max_prominence'),
#                 "seen_in_max_global_smoothing_scale_sigma": pe.get('max_global_sigma_seen', None),
#                 "seen_in_max_local_smoothing_scale_sigma": pe.get('max_roi_sigma_seen', None),
#                 "describe": pe.get('describe')
#             }
#             for pe in state.get('cleaned_peaks', [])[:num_peaks]
#         ]
#         trough_list = [
#             {
#                 "wavelength": tr.get('wavelength'),
#                 "flux": tr.get('mean_flux'),
#                 "width": tr.get('width_mean'),
#                 "seen_in_scales_of_sigma": tr.get('seen_in_scales_of_sigma')
#             }
#             for tr in state.get('cleaned_troughs', [])[:num_troughs]
#         ]
#
#         from AstroAgent.agents.multi_agents.utils.usage import find_overlap_regions, get_wiped_lines
#         rms = state.get("pixel_to_value", {}).get("x", {}).get("rms", 0)
#         tol = self.runtime.configs.params.tol_wavelength
#         rule_analysis = state['rule_analysis_QSO']
#         wl_left = state['spectrum']['new_wavelength'][0]
#         wl_right = state['spectrum']['new_wavelength'][-1]
#         arm_name = self.runtime.configs.params.arm_name
#         arm_wavelength_range = self.runtime.configs.params.arm_wavelength_range
#         if arm_name:
#             overlap_regions = find_overlap_regions(arm_name, arm_wavelength_range)
#             wiped_peaks = get_wiped_lines(state, overlap_regions)
#         debate_history = []
#         num_complete_rounds = min(
#             len(state['auditing_history_QSO']),
#             len(state['refining_history_QSO'])
#         )
#         if num_complete_rounds >= 1:
#             for i in range(num_complete_rounds):
#                 debate_history.append({
#                     "auditing": state['auditing_history_QSO'][i],
#                     "response": state['refining_history_QSO'][i],
#                 })
#         system_prompt, user_prompt = self.runtime.prompt_manager.load(
#             state=state,
#             agent_name=self.agent_name,
#             function_name=function_name,
#             wl_left=wl_left,
#             wl_right=wl_right,
#             peak_list=peak_list,
#             trough_list=trough_list,
#             overlap_regions=overlap_regions if arm_name else None,
#             wiped_peaks=wiped_peaks if arm_name else None,
#             rule_analysis=rule_analysis,
#             debate_history=debate_history,
#             rms=rms,
#             tol=tol
#         )
#         response = await self.call_llm_with_context(
#             system_prompt=system_prompt,
#             user_prompt=user_prompt,
#             parse_json=True,
#             description="Auditing",
#             want_tools=True
#         )
#         state['auditing_history_QSO'].append(response)
#         return state
