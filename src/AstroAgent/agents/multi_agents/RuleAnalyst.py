import json
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.ndimage import gaussian_filter1d

from AstroAgent.agents.common.state import SpectroState
from AstroAgent.agents.common.base_agent import BaseAgent
from AstroAgent.core.runtime.runtime_container import RuntimeContainer


from AstroAgent.agents.common.utils import (
    plot_cleaned_features, safe_to_bool, 
    find_overlap_regions, get_overlap_window,
    get_Ly_alpha_candidates, get_wiped_lines
)

# ---------------------------------------------------------
# 2. Rule-based Analyst — Responsible for rule-based physical analysis
# ---------------------------------------------------------
class RuleAnalyst(BaseAgent):
    """
    规则驱动型分析师：基于给定的物理与谱线知识进行定性分析
    Rule-based Analyst: Responsible for rule-based physical analysis
    """

    agent_name = "RuleAnalyst"

    def __init__(self, runtime: RuntimeContainer):
        super().__init__(runtime)

    async def run(self, state: SpectroState) -> SpectroState:
        await self.qualitative_analysis(state)
        await self.quantitative_analysis(state)

        q = state['rule_analysis_QSO']
        out_path = os.path.join(state['output_dir'], f'{state["image_name"]}_rule_analysis_QSO.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(q))
        return state


    async def qualitative_analysis(self, state: SpectroState) -> SpectroState:
        await self._check_arm_noise(state)
        await self._cleaning(state)
        await self._describe_continuum(state)
        await self._describe_lines(state)
        await self._describe_quality(state)

        sigma_list = self.runtime.configs.params.sigma_list
        num_peaks = self.runtime.configs.params.num_peaks
        num_troughs = self.runtime.configs.params.num_troughs
        plot_cleaned_features(state, sigma_list, num_peaks, num_troughs)

        qualitative_analysis_path = os.path.join(state['output_dir'], f'{state['image_name']}_qualitative_analysis.txt')
        with open(qualitative_analysis_path, 'w', encoding='utf-8') as f:
            json_str = json.dumps(state['qualitative_analysis'], indent=2, ensure_ascii=False)
            f.write(json_str)

        await self._preliminary_classification(state)
        await self._preliminary_classification_with_absention(state)
        await self._preliminary_classification_with_absention_monkey(state)
        
        return state
    
    async def _check_arm_noise(self, state: SpectroState) -> SpectroState:
        """
        使用 LLM 检测光谱中是否存在因为仪器的 arm 接缝导致的噪声。
        Use LLM to detect whether there is arm noise in the spectrum caused by the arm seam of the instrument.
        """
        function_name = "check_arm_noise"

        arm_name = self.runtime.configs.params.arm_name
        if arm_name:
            arm_wavelength_range = self.runtime.configs.params.arm_wavelength_range
            # Find the overlap region, output nearby infos (2*len(overlap region)) to make LLMs 
            # have a better understanding of the spectrum.
            overlap_regions = get_overlap_window(state, arm_name, arm_wavelength_range)
            system_prompt, user_prompt = self.runtime.prompt_manager.load(
                state=state,
                agent_name=self.agent_name,
                function_name=function_name,
                arm_name = arm_name, 
                overlap_regions = overlap_regions
            )
            # print('check_arm_noise:')
            # print(system_prompt)
            # print(user_prompt)
            result = await self.call_llm_with_context(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_json=True,
                description="arm noise",
                want_tools=False
            )
        else:
            result = {
                    "arm_noise": 'false',
                    "arm_noise_wavelength": None
                }
        state['qualitative_analysis']['arm_noise'] = result
        return state

    async def _cleaning(self, state: SpectroState) -> SpectroState:
        """
        根据 _check_arm_noise 的结果，把收到噪声影响的峰和谷过滤出去。
        Filter out the peaks and valleys affected by noise according to the result of _check_arm_noise.
        """
        arm_noise_analysis = state['qualitative_analysis']['arm_noise']
        if not safe_to_bool(arm_noise_analysis.get('arm_noise', False)):
            # No arm noise detected, no need to filter
            state['cleaned_peaks'] = state['merged_peaks']
            state['cleaned_troughs'] = state['merged_troughs']
        else:
            # Extract the noise position given by LLM
            arm_noise_wl = arm_noise_analysis.get('arm_noise_wavelength', [])
            arm_noise_wl = np.array(arm_noise_wl)

            wavelength = np.array(state['spectrum']['new_wavelength'])
            # peaks
            peaks = state['merged_peaks']
            cleaned_peaks = []
            wiped_peaks = []
            for p in peaks:
                wl = p['wavelength']
                width = p['width_mean']

                distance = abs(wl - arm_noise_wl)
                # If any distance is less than or equal to width, it is considered to be in the noise region
                if np.any(distance <= width):
                    is_artifact = True
                else:
                    is_artifact = False
                if not is_artifact:
                    if p['width_in_km_s'] is not None and p['wavelength'] > wavelength[0]: # Use p['wavelength'] > wavelength[0] to prevent issues at the sequence start (such as border not cleared properly leading to non-physical values at the sequence start)
                        if p['width_in_km_s'] > 2000:
                            p['describe'] = 'Broad line'
                        elif p['width_in_km_s'] < 1000:
                            p['describe'] = 'Narrow line'
                        else:
                            p['describe'] = 'Medium-width line'
                        cleaned_peaks.append(p)
                else:
                    wiped_peaks.append(p)
            state['cleaned_peaks'] = cleaned_peaks
            state['wiped_peaks'] = wiped_peaks
            # troughs
            cleaned_troughs = []
            for t in state['merged_troughs']:
                wl = t['wavelength']
                distance = abs(wl - arm_noise_wl)
                if np.any(distance <= width):
                    is_artifact = True
                else:
                    is_artifact = False
                if not is_artifact:
                    if t['width_in_km_s'] is not None and t['wavelength'] > wavelength[0]:
                        if t['width_in_km_s'] > 2000:
                            t['describe'] = 'Broad valley'
                        elif t['width_in_km_s'] < 1000:
                            t['describe'] = 'Narrow valley'
                        else:
                            t['describe'] = 'Medium-width valley'
                    else:
                        t['describe'] = 'Unprocessed'
                    cleaned_troughs.append(t)
            state['cleaned_troughs'] = cleaned_troughs
        return state

    async def _describe_continuum(self, state: SpectroState) -> SpectroState:
        function_name = 'describe_continuum'

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name
        )
        # print('describe_continuum')
        # print(system_prompt)
        # print(user_prompt)
        result = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=state['continuum_path'],
            parse_json=True,
            description="Describe continuum",
            want_tools=False
        )
        state['qualitative_analysis']['continuum'] = result
        return state
        
    async def _describe_lines(self, state: SpectroState) -> SpectroState:
        function_name = 'describe_lines'

        num_peaks = self.runtime.configs.params.num_peaks
        num_troughs = self.runtime.configs.params.num_troughs
        peaks = state['cleaned_peaks'][:num_peaks]
        troughs = state['cleaned_troughs'][:num_troughs]

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            peaks=peaks,
            troughs=troughs
        )
        # print('describe_lines')
        # print(system_prompt)
        # print(user_prompt)
        result = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=state['spec_extract_path'],
            parse_json=True,
            description="Describe lines",
            want_tools=False
        )
        state['qualitative_analysis']['lines'] = result
        return state

    async def _describe_quality(self, state: SpectroState) -> SpectroState:
        function_name = 'describe_quality'

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
        )
        # print('describe_quality')
        # print(system_prompt)
        # print(user_prompt)
        result = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=state['spec_extract_path'],
            parse_json=True,
            description="Describe quality",
            want_tools=False
        )
        state['qualitative_analysis']['quality'] = result
        return state
    
    async def _preliminary_classification(self, state: SpectroState) -> SpectroState:
        function_name = 'preliminary_classification'
        continuum_description = state['qualitative_analysis']['continuum']
        dataset = self.runtime.configs.params.dataset
        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            dataset=dataset,
            continuum_description=continuum_description
        )
        # print('preliminary_classification')
        # print(system_prompt)
        # print(user_prompt)
        result = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=None,
            parse_json=True,
            description="Preliminary classification",
            want_tools=False
        )
        state['preliminary_classification'] = result
        return state

    async def _preliminary_classification_with_absention(self, state: SpectroState) -> SpectroState:
        function_name = 'preliminary_classification_with_absention'
        continuum_description = state['qualitative_analysis']['continuum']
        dataset = self.runtime.configs.params.dataset

        arm_name = self.runtime.configs.params.arm_name
        arm_wavelength_range = self.runtime.configs.params.arm_wavelength_range
        wavelength = np.array(state['spectrum']['new_wavelength'])
        if arm_name and arm_wavelength_range:
            overlap = find_overlap_regions(arm_name, arm_wavelength_range)
        else:
            overlap = None
        if overlap:
            mask = np.zeros_like(wavelength, dtype=bool)
            for key, interval in overlap.items():
                a, b = interval
                band_mask = (wavelength >= a) & (wavelength <= b)            
            mask = mask | band_mask
            mask = ~mask
        else:
            mask = np.ones_like(wavelength, dtype=bool)
        snr = np.array(state['spectrum']['effective_snr'])
        mask_ = ~np.isinf(snr)  # Exclude positive and negative infinite values
        mask_[0] = False  # Also exclude the first value
        mask_[-1] = False  # Also exclude the last value
        mask = mask & mask_
        snr_ok = np.abs(snr[mask])
        snr_max = np.max(snr_ok)

        snr_threshold_upper = self.runtime.configs.params.snr_threshold_upper
        snr_threshold_lower = self.runtime.configs.params.snr_threshold_lower

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            dataset=dataset,
            continuum_description=continuum_description,
            snr_max=snr_max,
            snr_threshold_upper=snr_threshold_upper,
            snr_threshold_lower=snr_threshold_lower
        )
        # print('preliminary_classification_with_absention')
        # print(system_prompt)
        # print(user_prompt)
        result = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=None,
            parse_json=True,
            description="Preliminary classification_with_absention",
            want_tools=False
        )
        state['preliminary_classification_with_absention'] = result
        return state

    async def _preliminary_classification_with_absention_monkey(self, state: SpectroState) -> SpectroState:
        function_name = 'preliminary_classification_with_absention_monkey'
        preliminary_classification_with_absention = state['preliminary_classification_with_absention']
        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            preliminary_classification_with_absention=preliminary_classification_with_absention
        )
        # print('preliminary_classification_with_absention_monkey')
        # print(system_prompt)
        # print(user_prompt)
        result = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=None,
            parse_json=True,
            description="Preliminary classification_with_absention_monkey",
            want_tools=False
        )
        state['preliminary_classification_monkey'] = result
        return state

    async def quantitative_analysis(self, state: SpectroState) -> SpectroState:
        """QSO spectrum analysis"""

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

        get_Ly_alpha_candidates(state, peak_list)

        await self.step_1_QSO(state, peak_list, trough_list)
        await self.step_2_QSO(state, peak_list, trough_list)
        await self.step_3_QSO(state, peak_list, trough_list)
        await self.step_4_QSO(state, peak_list, trough_list)

        return state

    async def step_1_QSO(self, state: SpectroState, peak_list, trough_list) -> SpectroState:
        function_name = "step_1_QSO"

        qualitative_analysis = state.get('qualitative_analysis', [])
        wl_left = state['spectrum']['new_wavelength'][0]
        wl_right = state['spectrum']['new_wavelength'][-1]
        sigma_list = self.runtime.configs.params.sigma_list
        a_x = state['pixel_to_value']['x']['a']
        tol_wavelength = a_x * self.runtime.configs.params.tol_pixels
        lyalpha_candidates = state['Lyalpha_candidates']

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            qualitative_analysis=qualitative_analysis,
            peak_list=peak_list,
            trough_list=trough_list,
            wl_left=wl_left,
            wl_right=wl_right,
            sigma_list=sigma_list,
            tol_wavelength=tol_wavelength,
            lyalpha_candidates=lyalpha_candidates
        )

        result = await self.call_llm_with_context(
            system_prompt, 
            user_prompt, 
            parse_json=True, 
            description="Step 1",
            want_tools=True
        )

        state['rule_analysis_QSO'].append(result)
        return state
    
    async def step_2_QSO(self, state: SpectroState, peak_list, trough_list):
        function_name = "step_2_QSO"

        qualitative_analysis = state.get('qualitative_analysis', [])
        wl_left = state['spectrum']['new_wavelength'][0]
        wl_right = state['spectrum']['new_wavelength'][-1]
        sigma_list = self.runtime.configs.params.sigma_list
        a_x = state['pixel_to_value']['x']['a']
        tol_wavelength = a_x * self.runtime.configs.params.tol_pixels

        arm_name = self.runtime.configs.params.arm_name
        arm_wavelength_range = self.runtime.configs.params.arm_wavelength_range

        if arm_name: 
            overlap_regions = find_overlap_regions(arm_name, arm_wavelength_range)
            wiped_peaks = get_wiped_lines(state, overlap_regions)

        history = state['rule_analysis_QSO']

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            history=history,
            tol_wavelength=tol_wavelength,
            wiped_peaks=wiped_peaks if arm_name else None,
            overlap_regions=overlap_regions if arm_name else None,
            qualitative_analysis=qualitative_analysis,
            peak_list=peak_list,
            trough_list=trough_list,
            wl_left=wl_left,
            wl_right=wl_right,
            sigma_list=sigma_list,
        )

        result = await self.call_llm_with_context(
            system_prompt, 
            user_prompt, 
            parse_json=True, 
            description="Step 2",
            want_tools=True
        )

        state['rule_analysis_QSO'].append(result)
        return state

    async def step_3_QSO(self, state: SpectroState, peak_list, trough_list):
        function_name = "step_3_QSO"

        qualitative_analysis = state.get('qualitative_analysis', [])
        wl_left = state['spectrum']['new_wavelength'][0]
        wl_right = state['spectrum']['new_wavelength'][-1]
        sigma_list = self.runtime.configs.params.sigma_list
        a_x = state['pixel_to_value']['x']['a']
        tol_wavelength = a_x * self.runtime.configs.params.tol_pixels

        history = state['rule_analysis_QSO']

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            history=history,
            tol_wavelength=tol_wavelength,
            qualitative_analysis=qualitative_analysis,
            peak_list=peak_list,
            trough_list=trough_list,
            wl_left=wl_left,
            wl_right=wl_right,
            sigma_list=sigma_list,
        )

        result = await self.call_llm_with_context(
            system_prompt, 
            user_prompt, 
            parse_json=True, 
            description="Step 3",
            want_tools=True
        )

        state['rule_analysis_QSO'].append(result)
        return state

    async def step_4_QSO(self, state: SpectroState, peak_list, trough_list):
        function_name = "step_4_QSO"

        qualitative_analysis = state.get('qualitative_analysis', [])
        wl_left = state['spectrum']['new_wavelength'][0]
        wl_right = state['spectrum']['new_wavelength'][-1]
        sigma_list = self.runtime.configs.params.sigma_list
        a_x = state['pixel_to_value']['x']['a']
        tol_wavelength = a_x * self.runtime.configs.params.tol_pixels

        history = state['rule_analysis_QSO']

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            history=history,
            tol_wavelength=tol_wavelength,
            qualitative_analysis=qualitative_analysis,
            peak_list=peak_list,
            trough_list=trough_list,
            wl_left=wl_left,
            wl_right=wl_right,
            sigma_list=sigma_list,
        )

        result = await self.call_llm_with_context(
            system_prompt, 
            user_prompt, 
            parse_json=True, 
            description="Step 3",
            want_tools=True
        )

        state['rule_analysis_QSO'].append(result)
        return state
