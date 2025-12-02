import asyncio
import numpy as np
import os
import json

from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Set, Union

from langgraph.graph import StateGraph, END, START

from src.mcp_manager import MCPManager
from src.utils import getenv_int
from src.context_manager import SpectroState
from src.astro_agents import (
    SpectralVisualInterpreter as CN_SpectralVisualInterpreter,
    SpectralRuleAnalyst as CN_SpectralRuleAnalyst,
    SpectralAnalysisAuditor as CN_SpectralAnalysisAuditor,
    SpectralRefinementAssistant as CN_SpectralRefinementAssistant,
    SpectralSynthesisHost as CN_SpectralSynthesisHost
)
from src.astro_agents_EN import (
    SpectralVisualInterpreter as EN_SpectralVisualInterpreter,
    SpectralRuleAnalyst as EN_SpectralRuleAnalyst,
    SpectralAnalysisAuditor as EN_SpectralAnalysisAuditor,
    SpectralRefinementAssistant as EN_SpectralRefinementAssistant,
    SpectralSynthesisHost as EN_SpectralSynthesisHost
)

# damn


class WorkflowOrchestrator:
    """工作流编排器 - 管理整个智能体交互流程"""
    
    def __init__(self, config_file: str = "mcp_config.json"):
        # 加载环境变量
        load_dotenv()
        
        language = os.getenv('LANGUAGE')
        
        if language == "CN":
            self.agent_classes = {
                'SpectralVisualInterpreter': CN_SpectralVisualInterpreter,
                'SpectralRuleAnalyst': CN_SpectralRuleAnalyst,
                'SpectralAnalysisAuditor': CN_SpectralAnalysisAuditor,
                'SpectralRefinementAssistant': CN_SpectralRefinementAssistant,
                'SpectralSynthesisHost': CN_SpectralSynthesisHost
            }
        elif language == "EN":
            self.agent_classes = {
                'SpectralVisualInterpreter': EN_SpectralVisualInterpreter,
                'SpectralRuleAnalyst': EN_SpectralRuleAnalyst,
                'SpectralAnalysisAuditor': EN_SpectralAnalysisAuditor,
                'SpectralRefinementAssistant': EN_SpectralRefinementAssistant,
                'SpectralSynthesisHost': EN_SpectralSynthesisHost
            }
        else:
            raise ValueError(f"Language {language} is not supported")
        
        # 初始化MCP管理器
        self.mcp_manager = MCPManager(config_file)
        
        # 初始化所有智能体
        self.spectro_agents = self._initialize_agents()
        
        # 创建状态图
        self.workflow = self._create_workflow()
        
        self.max_debate_rounds = getenv_int('MAX_DEBATE_ROUNDS', 3)
        
        print("🚀 工作流编排器初始化完成")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """初始化所有智能体"""
        spectro_agents = {
            'Spectral_Visual_Interpreter': self.agent_classes['SpectralVisualInterpreter'](self.mcp_manager),
            'Spectral_Rule_Analyst': self.agent_classes['SpectralRuleAnalyst'](self.mcp_manager),
            'Spectral_Analysis_Auditor': self.agent_classes['SpectralAnalysisAuditor'](self.mcp_manager),
            'Spectral_Refinement_Assistant': self.agent_classes['SpectralRefinementAssistant'](self.mcp_manager),
            'Spectral_Synthesis_Host': self.agent_classes['SpectralSynthesisHost'](self.mcp_manager)
        }
        
        print(f"初始化了 {len(spectro_agents)} 个智能体")
        return spectro_agents
    
    def _check_cancel(self):
        """检查是否需要取消分析"""
        if self.cancel_checker and callable(self.cancel_checker):
            if self.cancel_checker():
                raise asyncio.CancelledError("分析已被用户取消")

    async def _visual_interpreter_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 1: Visual Interpreter')
        result = await self.spectro_agents["Spectral_Visual_Interpreter"].run(state, plot=True)
        self._check_cancel()
        return result
    
    async def _rule_analyst_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 2: Rule Analyst')
        result = await self.spectro_agents["Spectral_Rule_Analyst"].run(state)
        self._check_cancel()
        return result
    
    async def _analysis_auditor_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        count = state['count'] if isinstance(state, dict) else state.count
        if count == 0:
            print('Stage 3: Debate')
            current_round = (count + 1) // 2 + ((count + 1) % 2 > 0)
            print(f"🤔 Spectro analyse debate: 开始第 {count+1} 次发言, 当前轮数={current_round}, 最大轮数={self.max_debate_rounds}")
            print(f"⚖️ 开始光谱辩论 - 审查分析师 (第{current_round}轮)")
        result = await self.spectro_agents["Spectral_Analysis_Auditor"].run(state)
        # ✅ 修正：在返回的result上递增count
        if isinstance(result, dict):
            result['count'] = result.get('count', 0) + 1
        else:
            result.count += 1
        self._check_cancel()
        return result

    async def _refinement_assistant_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        result = await self.spectro_agents["Spectral_Refinement_Assistant"].run(state)
        # ✅ 修正：在返回的result上递增count
        if isinstance(result, dict):
            result['count'] = result.get('count', 0) + 1
        else:
            result.count += 1
        self._check_cancel()
        return result
    
    async def _synthesis_host_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 4: Synthesis Host')
        result = await self.spectro_agents["Spectral_Synthesis_Host"].run(state)
        self._check_cancel()
        return result
    
    def _should_continue_debate(self, state) -> str:
        """判断是否继续风险辩论"""
        count = state['count'] if isinstance(state, dict) else state.count
        
        # 计算当前轮数：每2次发言为1轮
        current_round = (count + 1) // 2 + ((count + 1) % 2 > 0)
        
        if current_round <= self.max_debate_rounds:
            print(f"🤔 Spectro analyse debate: 开始第 {count+1} 次发言, 当前轮数={current_round}, 最大轮数={self.max_debate_rounds}")
            # ✅ 修正：正确的轮换逻辑
            if count % 2 == 1:  # 奇数：刚执行完auditor，下一步是assistant
                print(f"🖋️ 继续光谱辩论 - 完善分析师 (第{current_round}轮)")
                return "refinement_assistant"
            else:  # 偶数：刚执行完assistant，下一步是auditor
                print(f"⚖️ 继续光谱辩论 - 审查分析师 (第{current_round}轮)")
                return "analysis_auditor"
        else:
            print(f"🏁 光谱辩论结束({self.max_debate_rounds}轮完成)，进入总结")
            return "synthesis_host"
    
    def _create_workflow(self) -> StateGraph:
        
        workflow = StateGraph(SpectroState)
        
        # 添加节点
        workflow.add_node("visual_interpreter", self._visual_interpreter_node)
        workflow.add_node("rule_analyst", self._rule_analyst_node)
        workflow.add_node("analysis_auditor", self._analysis_auditor_node)
        workflow.add_node("refinement_assistant", self._refinement_assistant_node)
        workflow.add_node("synthesis_host", self._synthesis_host_node)

        workflow.add_edge(START, 'visual_interpreter')
        workflow.set_entry_point("visual_interpreter")
        workflow.add_edge("visual_interpreter", "rule_analyst")
        workflow.add_edge("rule_analyst", "analysis_auditor")
        workflow.add_conditional_edges(
            "analysis_auditor",
            self._should_continue_debate,
            {
                "refinement_assistant": "refinement_assistant",
                "synthesis_host": "synthesis_host"
            }
        )
        workflow.add_conditional_edges(
            "refinement_assistant",
            self._should_continue_debate,
            {
                "analysis_auditor": "analysis_auditor",
                "synthesis_host": "synthesis_host"
            }
        )
        workflow.add_edge("synthesis_host", END)
        # workflow.add_edge("analysis_auditor", 'refinement_assistant')
        # workflow.add_edge("refinement_assistant", END)
        
        return workflow.compile()
    
    async def initialize(self) -> bool:
        """初始化MCP连接"""
        try:
            success = await self.mcp_manager.initialize()
            if success:
                print("✅ 工作流编排器初始化成功")
            else:
                print("⚠️ MCP连接失败，将在无工具模式下运行")
            return success
        except Exception as e:
            print(f"❌ 工作流编排器初始化失败: {e}")
            return False

    async def run_analysis_single(self, cancel_checker=None) -> SpectroState:
        """运行完整的交易分析流程"""
        print("🚀 Start MCP LLM Spectro Agent")
        # 存储取消检查器
        self.cancel_checker = cancel_checker

        input_dir = os.getenv('INPUT_DIR')
        output_dir = os.getenv('OUTPUT_DIR')
        image_name = os.getenv('IMAGE_NAME')
        image_path = os.path.join(input_dir, f'{image_name}.png')
        cropped_path = os.path.join(output_dir, f'{image_name}_cropped.png')
        spec_extract_path = os.path.join(output_dir, f'{image_name}_spec_extract.png')
        continuum_path = os.path.join(output_dir, f'{image_name}_continuum.png')

        BAND_NAME = os.getenv('BAND_NAME', None)
        if BAND_NAME is not None:
            BAND_NAME = BAND_NAME.split(',')
        # BAND_WAVELENGTH=3600-5800,5760-7620,7520-9824
        BAND_WAVELENGTH = os.getenv('BAND_WAVELENGTH', None)
        if BAND_WAVELENGTH is not None:
            BAND_WAVELENGTH = [list(map(float, band.split('-'))) for band in BAND_WAVELENGTH.split(',')]
        print(f"🔍 分析设置 - 波段名称: {BAND_NAME}, 波段波长: {BAND_WAVELENGTH}")

        prompts_path = os.getenv('PROMPTS_PATH')
        with open(prompts_path, 'r', encoding='utf-8') as f:
            PROMPTS = json.load(f)


        # 初始化状态
        initial_state = SpectroState(
            image_path=image_path,
            image_name=image_name,
            output_dir=output_dir,
            crop_path=cropped_path, 
            spec_extract_path=spec_extract_path,
            continuum_path=continuum_path,
            band_name=BAND_NAME,
            band_wavelength=BAND_WAVELENGTH,
            prompt=PROMPTS,
            count=0,
            visual_interpretation=[],
            possible_object=[],
            rule_analysis_QSO=[],
            rule_analysis_galaxy=[],
            auditing_history_QSO=[], 
            refine_history_QSO=[], 
            auditing_history_galaxy=[], 
            refine_history_galaxy=[], 
            in_brief = {}
        )
        try:
            # 检查取消状态
            self._check_cancel()
            
            # 运行工作流
            workflow_result = await self.workflow.ainvoke(initial_state)
            
            # LangGraph返回字典，需要转换为AgentState对象
            if isinstance(workflow_result, dict):
                # 创建新的AgentState对象并复制数据
                final_state = SpectroState(
                    image_name  = workflow_result.get('image_name', None),
                    image_path  = workflow_result.get('image_path', None),
                    output_dir  = workflow_result.get('output_dir', None),
                    crop_path   = workflow_result.get('crop_path', None),
                    spec_extract_path = workflow_result.get('spec_extract_path', None),
                    max_debate_rounds = workflow_result.get('max_debate_rounds', None),
                    sigma_list  = workflow_result.get('sigma_list', None),
                    axis_info   = workflow_result.get('axis_info', None),
                    OCR_detected_ticks  = workflow_result.get('OCR_detected_ticks', None),
                    tick_pixel_raw      = workflow_result.get('tick_pixel_raw', None),
                    chart_border        = workflow_result.get('chart_border', None),
                    tick_pixel_remap    = workflow_result.get('tick_pixel_remap', None),
                    pixel_to_value      = workflow_result.get('pixel_to_value', None),
                    curve_points        = workflow_result.get('curve_points', None),
                    curve_gray_values = workflow_result.get('curve_gray_values', None),
                    spectrum         = workflow_result.get('spectrum', None),
                    peaks            = workflow_result.get('peaks', None),
                    troughs          = workflow_result.get('troughs', None),
                    spectrum_fig     = workflow_result.get('spectrum_fig', None),
                    features_fig     = workflow_result.get('features_fig', None),
                    visual_interpretation       = workflow_result.get('visual_interpretation', None),
                    preliminary_classification  = workflow_result.get('preliminary_classification', None),
                    rule_analysis_QSO           = workflow_result.get('rule_analysis_QSO', None),
                    auditing_history_QSO        = workflow_result.get('auditing_history_QSO', None),
                    refine_history_QSO          = workflow_result.get('refine_history_QSO', None),
                    rule_analysis_galaxy           = workflow_result.get('rule_analysis_galaxy', None),
                    auditing_history_galaxy        = workflow_result.get('auditing_history_galaxy', None),
                    refine_history_galaxy          = workflow_result.get('refine_history_galaxy', None),
                    summary                     = workflow_result.get('summary', None),
                    in_brief                    = workflow_result.get('in_brief', None)
                )
            else:
                final_state = workflow_result

            try:
                # 安全提取 rule_analysis
                rule_list = final_state.get('rule_analysis_QSO')
                rule_list_2 = final_state.get('rule_analysis_galaxy')
                if not isinstance(rule_list, (list, tuple)):
                    rule_list = []
                rule_analysis = "\n\n".join(str(item) for item in rule_list if item is not None)
                rule_analysis_2 = "\n\n".join(str(item) for item in rule_list_2 if item is not None)

                # 安全提取 summary
                summary = final_state.get('summary', '')
                if summary is None:
                    summary = ''

                output_dir = final_state['output_dir']
                image_name = final_state['image_name']

                # ✅ 用 open 写文本

                md_path = os.path.join(output_dir, f'{image_name}_rule_analysis.md')
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(rule_analysis + rule_analysis_2)

                summary_path = os.path.join(output_dir, f'{image_name}_summary.md')
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)

                print("✅ 分析日志和总结已保存")
                
            except Exception as e:
                print(f"⚠️ 保存分析结果时出错: {e}")
                # 可选：继续抛出或记录
                
            print("✅ 分析流程完成")
            return final_state
            
        except asyncio.CancelledError as e:
            print(f"⚠️ 分析流程已取消: {e}")
            return initial_state
            
        except Exception as e:
            print(f"❌ 分析流程失败: {e}")
            return initial_state