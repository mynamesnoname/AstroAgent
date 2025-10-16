import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from .agent_states import AgentState
from .mcp_manager import MCPManager

class BaseAgent(ABC):
    """基础智能体类 - 所有智能体的基类"""
    
    def __init__(self, agent_name: str, mcp_manager: MCPManager, role_description: str = ""):
        self.agent_name = agent_name
        self.mcp_manager = mcp_manager
        
        # 获取LLM实例
        self.llm = mcp_manager.llm
        
        # 不在初始化时获取工具，而是在使用时动态获取
        self.available_tools = []
        
        # 延迟创建智能体实例，等到MCP工具初始化完成后再创建
        self.agent = None
        
        print(f"智能体 {agent_name} 初始化完成，MCP工具: {'启用' if self.mcp_enabled else '禁用'}")
    
    async def call_llm_with_context(state, request, progress_tracker):
        # 将系统和上下文组合成一个系统消息
        system_level_prompt = f"""{system_prompt}

        {context_prompt}"""
