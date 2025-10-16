from typing import Any, Dict, List, Optional
import json
import os
from langchain.chat_models import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPManager():
    def __init__(self, config_file:str, llm_type:str):

        # 读取 MCP server 的配置文件
        self.config = self._load_config(config_file)

        # 初始化大模型
        self.llm, self.visual_llm = self._init_llm(llm_type)

        # MCP客户端和工具
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: List = []
        self.tools_by_server: Dict[str, List] = {}

        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []
        
        print("MCP管理器初始化完成")

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"配置文件加载成功: {config_file}")
            return config
        except FileNotFoundError:
            print(f"⚠️ 配置文件未找到: {config_file}，使用默认配置")
            return {"servers": {}, "agent_permissions": {}}
        except json.JSONDecodeError as e:
            print(f"❌ 配置文件格式错误: {e}")

    def _init_llm(self) -> ChatOpenAI:
        """初始化大模型 - 从环境变量加载配置"""
        # 大模型配置只从环境变量加载
        llm_api_key = os.getenv("CHAT_LLM_API_KEY", "your_api_key_here")
        visual_llm_api_key = os.getenv("VISUAL_LLM_API_KEY", "your_api_key_here")
        base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        chat_model_name = os.getenv("LLM_MODEL", "qwen3-max")
        visual_model_name = os.getenv("LLM_MODEL", "qwen-vl-max-latest")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "None"))

        print(f"[LLM INIT] Loaded from env -> chat_LLM_MODEL={chat_model_name}, visual_LLM_MODEL={visual_model_name}, LLM_TEMPERATURE={temperature}, LLM_MAX_TOKENS={max_tokens}, LLM_BASE_URL={base_url}")
        
        llm = ChatOpenAI(
            model=chat_model_name,
            api_key=llm_api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )

        visual_llm = ChatOpenAI(
            model=visual_model_name,
            api_key=visual_llm_api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )

        try:
            print(
                f"[LLM INIT] ChatOpenAI config -> model={getattr(llm, 'model', getattr(llm, 'model_name', None))}, "
                f"temperature={getattr(llm, 'temperature', None)}, max_tokens={getattr(llm, 'max_tokens', None)}"
            )
        except Exception as _:
            pass
        
        print(f"大模型初始化完成: {chat_model_name}, {visual_model_name} @ {base_url}")
        return llm, visual_llm