import json
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.ndimage import gaussian_filter1d

from AstroAgent.agents.common.state import SpectroState
from AstroAgent.agents.common.base_agent import BaseAgent
from AstroAgent.core.runtime.runtime_container import RuntimeContainer

from AstroAgent.agents.common.utils import find_overlap_regions, get_wiped_lines


class RefinementAssistant(BaseAgent):
    """
    改进者：回应审查并改进分析
    Refinement Assistant: Respond to auditing and improve analysis
    """
    agent_name = "RefinementAssistant"

    def __init__(self, runtime: RuntimeContainer):
        super().__init__(runtime)

    
    async def run(self, state: SpectroState) -> SpectroState:
        await self.refining(state)
        return state

    async def refining(self, state: SpectroState) -> SpectroState:
        function_name = "refining"

        num_peaks = self.runtime.configs.params.num_peaks
        num_troughs = self.runtime.configs.params.num_troughs

        # We offer a limited number of features to LLM
        peak_list = [
            {
                "wavelength": pe.get('wavelength'),
                "flux": pe.get('mean_flux'),
                "width": pe.get('width_mean'),
                "width_in_km_s": pe.get('width_in_km_s'),
                "prominance": pe.get('max_prominence'),
                "seen_in_max_global_smoothing_scale_sigma": pe.get('max_global_sigma_seen', None),
                "seen_in_max_local_smoothing_scale_sigma": pe.get('max_roi_sigma_seen', None),
                "describe": pe.get('describe')
            }
            for pe in state.get('cleaned_peaks', [])[:num_peaks]
        ]
        trough_list = [
            {
                "wavelength": tr.get('wavelength'),
                "flux": tr.get('mean_flux'),
                "width": tr.get('width_mean'),
                "seen_in_scales_of_sigma": tr.get('seen_in_scales_of_sigma')
            }
            for tr in state.get('cleaned_troughs', [])[:num_troughs]
        ]

        rms = state["pixel_to_value"]["x"]["rms"]
        a_x = state['pixel_to_value']['x']['a']
        tol = self.runtime.configs.params.tol_pixels
        rule_analysis = state['rule_analysis_QSO']
        wl_left = state['spectrum']['new_wavelength'][0]
        wl_right = state['spectrum']['new_wavelength'][-1]

        arm_name = self.runtime.configs.params.arm_name
        arm_wavelength_range = self.runtime.configs.params.arm_wavelength_range

        if arm_name: 
            overlap_regions = find_overlap_regions(arm_name, arm_wavelength_range)
            wiped_peaks = get_wiped_lines(state, overlap_regions)

        rule_analysis = state['rule_analysis_QSO']

        debate_history = []
        num_complete_rounds = min(
            len(state['auditing_history_QSO']),
            len(state['refining_history_QSO'])
        )
        if num_complete_rounds >= 1:
            for i in range(num_complete_rounds):
                auditing_history = state['auditing_history_QSO'][i] 
                response_history = state['refining_history_QSO'][i]
                history = {
                    "auditing": auditing_history,
                    "response": response_history
                }
                debate_history.append(history)
        
        latest_auditing = state['auditing_history_QSO'][-1]

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            wl_left=wl_left,
            wl_right=wl_right,
            peak_list=peak_list,
            trough_list=trough_list,
            overlap_regions=overlap_regions if arm_name else None,
            wiped_peaks=wiped_peaks if arm_name else None,
            rule_analysis=rule_analysis,
            debate_history=debate_history,
            a=a_x,
            rms=rms,
            tol=tol,
            latest_auditing=latest_auditing
        )

        response = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parse_json=True,
            description="Auditing",
            want_tools=True
        )

        state['refining_history_QSO'].append(response)

        return state
