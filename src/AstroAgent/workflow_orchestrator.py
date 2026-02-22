import asyncio

from typing import Dict, Any, List, Optional, Set, Union

from langgraph.graph import StateGraph, END, START

from AstroAgent.core.runtime.runtime_container import RuntimeContainer
from AstroAgent.agents.common.state import SpectroState

#########################
# To be done
from AstroAgent.agents.multi_agents.VisualInterpreter import VisualInterpreter
from AstroAgent.agents.multi_agents.RuleAnalyst import RuleAnalyst
from AstroAgent.agents.multi_agents.AnalysisAuditor import AnalysisAuditor
from AstroAgent.agents.multi_agents.RefinementAssistant import RefinementAssistant
from AstroAgent.agents.multi_agents.SynthesisHost import SynthesisHost
#########################


class WorkflowOrchestrator:
    """
    工作流编排器：管理整个智能体交互流程
    Workflow orchestrator: manage the entire agent interaction workflow
    """


    # 定义语言到代理类的映射
    AGENT_CLASSES = {
        'VisualInterpreter': VisualInterpreter,
        'RuleAnalyst': RuleAnalyst,
        'AnalysisAuditor': AnalysisAuditor,
        'RefinementAssistant': RefinementAssistant,
        'SynthesisHost': SynthesisHost
    }
    
    def __init__(
            self, 
            configs: Dict[str, Any],
        ):

        self.configs = configs

        # language = self.configs.language  # 默认使用 'CN'

        # self.agent_classes = self.AGENT_CLASSES.get(language, self.AGENT_CLASSES["CN"])

        # 初始化运行时容器
        self.runtime = RuntimeContainer(self.configs)
        
        # 初始化所有智能体
        self.spectro_agents = self._initialize_agents()
        
        # 创建状态图
        self.workflow = self._create_workflow()
        
        self.max_debate_rounds = self.configs.params.max_debate_rounds
        
        print("🚀 工作流编排器初始化完成")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """
        初始化所有智能体
        Initialize all agents
        """
        spectro_agents = {
            '_Visual_Interpreter': self.AGENT_CLASSES['VisualInterpreter'](self.runtime),
            '_Rule_Analyst': self.AGENT_CLASSES['RuleAnalyst'](self.runtime),
            '_Analysis_Auditor': self.AGENT_CLASSES['AnalysisAuditor'](self.runtime),
            '_Refinement_Assistant': self.AGENT_CLASSES['RefinementAssistant'](self.runtime),
            '_Synthesis_Host': self.AGENT_CLASSES['SynthesisHost'](self.runtime)
        }
        # print(f"初始化了 {len(spectro_agents)} 个智能体")
        print(f"Initialized {len(spectro_agents)} agents")
        return spectro_agents
    
    def _check_cancel(self):
        """
        检查是否需要取消分析
        Check if analysis needs to be canceled
        """
        if self.cancel_checker and callable(self.cancel_checker):
            if self.cancel_checker():
                raise asyncio.CancelledError("分析已被用户取消 Analysis canceled by user")

    async def _visual_interpreter_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 1: Visual Interpreter')
        result = await self.spectro_agents["_Visual_Interpreter"].run(state, plot=True)
        self._check_cancel()
        return result
    
    async def _rule_analyst_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 2: Rule Analyst')
        result = await self.spectro_agents["_Rule_Analyst"].run(state)
        self._check_cancel()
        return result
    
    async def _analysis_auditor_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        count = state['count'] if isinstance(state, dict) else state.count
        if count == 0:
            print('Stage 3: Debate')
            current_round = (count + 1) // 2 + ((count + 1) % 2 > 0)
            print(f"Spectro analyse debate: Starting the {count+1}th statement, current round={current_round}, max rounds={self.max_debate_rounds}")
            print(f"Starting spectro debate - Analysis Auditor (Round {current_round})")
        result = await self.spectro_agents["_Analysis_Auditor"].run(state)
        if isinstance(result, dict):
            result['count'] = result.get('count', 0) + 1
        else:
            result.count += 1
        self._check_cancel()
        return result

    async def _refinement_assistant_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        result = await self.spectro_agents["_Refinement_Assistant"].run(state)
        if isinstance(result, dict):
            result['count'] = result.get('count', 0) + 1
        else:
            result.count += 1
        self._check_cancel()
        return result
    
    async def _synthesis_host_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 4: Synthesis Host')
        result = await self.spectro_agents["_Synthesis_Host"].run(state)
        self._check_cancel()
        return result
    
    def _should_continue_debate(self, state) -> str:
        count = state['count'] if isinstance(state, dict) else state.count

        current_round = (count + 1) // 2 + ((count + 1) % 2 > 0)
        
        if current_round <= self.max_debate_rounds:
            print(f"Spectro analyse debate: Starting the {count+1}th statement, current round={current_round}, max rounds={self.max_debate_rounds}")
            if count % 2 == 1:  
                print(f"Continuing spectro debate - Refinement Assistant (Round {current_round})")
                return "refinement_assistant"
            else: 
                print(f"Continuing spectro debate - Analysis Auditor (Round {current_round})")
                return "analysis_auditor"
        else:
            print(f"Spectro analyse debate: Debate completed ({self.max_debate_rounds} rounds), entering synthesis")
            return "synthesis_host"
    
    def _create_workflow(self) -> StateGraph:
        
        workflow = StateGraph(SpectroState)
        
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
        
        return workflow.compile()
    

    async def run_analysis_single(self, state, cancel_checker=None) -> SpectroState:
        
        print("🚀 Start MCP LLM Spectro Agent")
        # 存储取消检查器
        self.cancel_checker = cancel_checker

        # 初始化状态
        initial_state = state
        try:
            # 检查取消状态
            self._check_cancel()
            
            # 运行工作流
            workflow_result = await self.workflow.ainvoke(initial_state)
            final_state = workflow_result
                
            print("✅ 分析流程完成")
            return final_state
            
        except asyncio.CancelledError as e:
            print(f"⚠️ 分析流程已取消: {e}")
            return initial_state
            
        except Exception as e:
            print(f"❌ 分析流程失败: {e}")
            return initial_state