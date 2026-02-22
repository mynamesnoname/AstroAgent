import json
import os
import logging
import sys

from AstroAgent.agents.common.state import SpectroState
from AstroAgent.agents.common.base_agent import BaseAgent
from AstroAgent.core.runtime.runtime_container import RuntimeContainer


class SynthesisHost(BaseAgent):
    """
    综合主持人：整合和总结多个分析结果，生成最终报告
    Synthesis Host: Integrate and summarize multiple analysis results to generate a final report
    """
    agent_name = "SynthesisHost"

    def __init__(self, runtime: RuntimeContainer):
        super().__init__(runtime)

    async def run(self, state: SpectroState) -> SpectroState:
        await self.summary(state)
        await self.in_brief(state)
        supplimentary_classification_json = state['preliminary_classification_with_absention']
        state['summary'] += f"""

# Supplementary materials: The preliminary classification of the spectrum:
{supplimentary_classification_json}
"""
        return state

    async def summary(self, state: SpectroState) -> SpectroState:
        """
        综合分析结果并生成最终报告
        Integrate and summarize multiple analysis results to generate a final report
        """
        function_name = "summary"

        visual_interpretation = state.get('visual_interpretation', {})
        preliminary_classification_with_absention = state['preliminary_classification_monkey']['type']
        preliminary_classification = state['preliminary_classification']['type']        
        rule_analysis_QSO = state.get('rule_analysis_QSO', {})
        auditing_QSO = state.get('auditing_history_QSO', [])
        refining_QSO = state.get('refining_history_QSO', [])

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name=function_name,
            visual_interpretation=visual_interpretation,
            preliminary_classification_with_absention=preliminary_classification_with_absention,
            preliminary_classification=preliminary_classification,
            rule_analysis_QSO=rule_analysis_QSO,
            auditing_QSO=auditing_QSO,
            refining_QSO=refining_QSO
        )

        response = await self.call_llm_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parse_json=True,
            description="summary",
            want_tools=False
        )

        state['summary'] = response

        return state

    
    async def in_brief(self, state):
        """
        生成简洁的报告摘要
        Generate a brief report summary
        """

        # type with absention
        state['in_brief']['type_with_absention'] = state['preliminary_classification_monkey']['type']
        
        # type forced        
        state['in_brief']['type_forced'] = state['preliminary_classification']['type']

        await self.in_brief_redshift(state)
        await self.in_brief_rms(state)
        await self.in_brief_lines(state)
        await self.in_brief_score(state)
        await self.in_brief_human(state)
        return state

    async def in_brief_redshift(self, state):
        summary = state['summary']

        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name="in_brief_redshift",
            summary=summary
        )

        response_redshift = await self.call_llm_with_context(
            system_prompt, 
            user_prompt, 
            parse_json=True, 
            description="总结",
            want_tools=False
        )
        state['in_brief']['redshift'] = response_redshift
        return state
    
    async def in_brief_rms(self, state):
        summary = state['summary']
        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name="in_brief_rms",
            summary=summary
        )
        response_rms = await self.call_llm_with_context(
            system_prompt, 
            user_prompt, 
            parse_json=True, 
            description="总结",
            want_tools=False
        )
        state['in_brief']['rms'] = response_rms
        return state

    async def in_brief_lines(self, state):
        summary = state['summary']
        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name="in_brief_lines",
            summary=summary
        )

        response_lines = await self.call_llm_with_context(
            system_prompt, 
            user_prompt, 
            parse_json=True, 
            description="总结", 
            want_tools=False
        )
        state['in_brief']['lines'] = response_lines
        return state

    async def in_brief_score(self, state):
        summary = state['summary']
        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name="in_brief_score",
            summary=summary
        )
        
        response_score = await self.call_llm_with_context(
            system_prompt, 
            user_prompt, 
            parse_json=True, 
            description="总结", 
            want_tools=False
        )
        state['in_brief']['score'] = response_score
        return state
    
    async def in_brief_human(self, state):
        summary = state['summary']
        system_prompt, user_prompt = self.runtime.prompt_manager.load(
            state=state,
            agent_name=self.agent_name,
            function_name="in_brief_human",
            summary=summary
        )

        response_human = await self.call_llm_with_context(
            system_prompt, 
            user_prompt, 
            parse_json=True, 
            description="总结", 
            want_tools=False
        )
        state['in_brief']['human'] = response_human
        return state
